import random

import torch
from torch_geometric.utils import remove_self_loops, coalesce, scatter


class MotiFiestaDisc(torch.nn.Module):
    """subgraph representation learning for unsupervised motif discovery.

    ego-net sampling (esan ego+ policy):
      for each anchor node, collect all nodes within h hops (bfs).
      anchor is marked with a binary flag before gin — this is esan ego+,
      strictly more expressive than ego (unmarked) and 1-wl.

    wl_hops augmentation (dss approximation):
      node features are augmented with k-hop neighbor sums on the full graph
      before subgraph extraction. this injects global graph structure into each
      subgraph's gin computation, approximating the dss-gnn l2 cross-subgraph
      information sharing module (bevilacqua et al. 2022).

    joint structural representation (srinivasan & ribeiro 2020):
      gin runs on the induced subgraph. the embedding of each anchor is a
      function of the joint structure of its ego-net, not independent per-node.

    forward returns a list of per-level dicts:
      z_sub:       (n_sub, hidden_dim) subgraph embeddings
      scores:      (n_sub,) motif membership scores in (0, 1)
      node_to_sub: (n_nodes,) maps each node to its highest-scoring subgraph
      flat_nodes:  (total_flat,) global node index for each flat position
      flat_subs:   (total_flat,) subgraph index for each flat position
      sub_batch:   (n_sub,) graph id for each subgraph
    """

    def __init__(self, n_features=25, hidden_dim=32, gin_layers=2,
                 walk_lens=(1, 2, 3), wl_hops=1, pool='mean', **_):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.walk_lens = list(walk_lens) if hasattr(walk_lens, '__iter__') else [walk_lens]
        self.wl_hops = wl_hops
        self.pool = pool

        # input: wl-augmented features + 4-dim anchor distance one-hot (0/1/2/3+) + 1-dim ego+ flag
        wl_dim = n_features * (1 + wl_hops)
        in_dim = wl_dim + 4 + 1

        self.gin = torch.nn.ModuleList()
        for i in range(gin_layers):
            d_in = in_dim if i == 0 else hidden_dim
            self.gin.append(torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            ))

        # per-level score nets: each hop level scores independently (matches motifiesta design)
        self.score_nets = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, 1) for _ in self.walk_lens
        ])

        # attention pooling: per-node logit for weighted subgraph embedding
        if pool == 'attn':
            self.attn_net = torch.nn.Linear(hidden_dim, 1)

        # decoder for warmup rec_loss: reconstruct mean degree distribution from z_sub
        self.x_decoder = torch.nn.Linear(hidden_dim, n_features)

    def _wl_augment(self, x, edge_index):
        """k-hop wl neighbor sum augmentation on the full graph."""
        src, dst = edge_index[0], edge_index[1]
        parts = [x]
        cur = x
        for _ in range(self.wl_hops):
            nbr = torch.zeros_like(cur).scatter_add(
                0, dst.unsqueeze(1).expand(-1, x.size(1)), cur[src]
            )
            parts.append(nbr)
            cur = nbr
        return torch.cat(parts, dim=-1)

    def _build_adj(self, edge_index, n_nodes):
        adj = [[] for _ in range(n_nodes)]
        for u, v in zip(edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist()):
            adj[u].append(v)
        return adj

    def _h_hop_ego(self, anchor, adj, h):
        """bfs from anchor to depth h; returns sorted node list."""
        visited = {anchor}
        frontier = {anchor}
        for _ in range(h):
            nxt = {w for v in frontier for w in adj[v] if w not in visited}
            visited.update(nxt)
            frontier = nxt
            if not frontier:
                break
        return sorted(visited)

    def _sample_anchors(self, batch, max_per_graph=200):
        """sample anchor nodes once per graph, shared across all hop levels.

        returns dict {graph_id: [anchor_nodes]}.
        """
        batch_cpu = batch.cpu()
        anchor_dict = {}
        for g in batch_cpu.unique():
            g_int = g.item()
            nodes = (batch_cpu == g).nonzero(as_tuple=False).squeeze(-1).tolist()
            anchor_dict[g_int] = random.sample(nodes, min(max_per_graph, len(nodes)))
        return anchor_dict

    def _sample_subgraphs(self, adj, batch, hop, anchor_dict=None, max_per_graph=200):
        """ego-net sampling: h-hop ego-net per anchor, uniform anchor selection.

        returns list of (anchor, sorted_node_list, graph_id).
        if anchor_dict is provided (shared anchors), deduplicates by anchor identity
        so all hop levels operate on the same anchor set.
        otherwise deduplicates by ego-net node set.
        """
        batch_cpu = batch.cpu()
        all_subgraphs = []
        for g in batch_cpu.unique():
            g_int = g.item()
            nodes = (batch_cpu == g).nonzero(as_tuple=False).squeeze(-1).tolist()
            if anchor_dict is not None:
                # shared anchors: deduplicate by anchor identity, not ego-net
                seen_anchors = set()
                for anchor in anchor_dict[g_int]:
                    if anchor not in seen_anchors:
                        seen_anchors.add(anchor)
                        all_subgraphs.append((anchor, self._h_hop_ego(anchor, adj, hop), g_int))
            else:
                anchors = random.sample(nodes, min(max_per_graph, len(nodes)))
                seen = set()
                for anchor in anchors:
                    node_list = self._h_hop_ego(anchor, adj, hop)
                    key = frozenset(node_list)
                    if key not in seen:
                        seen.add(key)
                        all_subgraphs.append((anchor, node_list, g_int))
        return all_subgraphs

    def _embed_subgraphs(self, x_aug, adj, subgraph_list, device, hop=1):
        """induced subgraph gin: build batched graph → ego+ flag → gin → pool.

        x_target is the mean of the outermost shell (nodes at distance exactly `hop`
        from their anchor). this gives each hop level a distinct reconstruction target.
        falls back to full-subgraph mean for any subgraph with no shell nodes.
        """
        if not subgraph_list:
            emp = torch.zeros(0, dtype=torch.long, device=device)
            return (torch.zeros(0, self.hidden_dim, device=device), emp, emp, [],
                    torch.zeros(0, self.n_features))

        n_sub = len(subgraph_list)
        x_aug_cpu = x_aug.detach().cpu()
        ei_src, ei_dst, anchors = [], [], []
        sub_assign, flat_nodes = [], []
        node_offset = 0

        sg_data = []  # (n_nodes, local_edge_list, node_features_np) per subgraph
        for s_idx, (au, nodes, _g) in enumerate(subgraph_list):
            node_set = set(nodes)
            local_idx = {v: i for i, v in enumerate(nodes)}
            local_edges = []
            for v in nodes:
                lv = local_idx[v]
                for w in adj[v]:
                    if w in node_set:
                        lw = local_idx[w]
                        ei_src.append(lv + node_offset)
                        ei_dst.append(lw + node_offset)
                        if lv < lw:
                            local_edges.append((lv, lw))
            nf = x_aug_cpu[torch.tensor(nodes, dtype=torch.long)].numpy()[:, :self.n_features]
            sg_data.append((len(nodes), local_edges, nf))
            anchors.append(local_idx[au] + node_offset)
            sub_assign.extend([s_idx] * len(nodes))
            flat_nodes.extend(nodes)
            node_offset += len(nodes)

        N = node_offset
        src = torch.tensor(ei_src, dtype=torch.long) if ei_src else torch.zeros(0, dtype=torch.long)
        dst = torch.tensor(ei_dst, dtype=torch.long) if ei_src else torch.zeros(0, dtype=torch.long)

        reach = torch.zeros(N, dtype=torch.long)
        umask = torch.empty(N, dtype=torch.bool)
        dmask = torch.empty(N, dtype=torch.bool)

        def bfs(anchor_list):
            d = torch.full((N,), 3, dtype=torch.long)
            d[torch.tensor(anchor_list, dtype=torch.long)] = 0
            if src.numel() == 0:
                return d
            for hop in range(1, 3):
                reach.zero_()
                reach.scatter_add_(0, dst, (d[src] == hop - 1).long())
                torch.gt(reach, 0, out=umask)
                torch.gt(d, hop, out=dmask)
                umask.logical_and_(dmask)
                d.masked_fill_(umask, hop)
            return d

        d = bfs(anchors)
        dist_oh = torch.zeros(N, 4)
        dist_oh.scatter_(1, d.unsqueeze(1), 1.0)

        # ego+: binary flag on the anchor node, persists through gin message passing
        anchor_flag = torch.zeros(N, 1)
        anchor_flag[torch.tensor(anchors, dtype=torch.long)] = 1.0

        nodes_t = torch.tensor(flat_nodes, dtype=torch.long)
        X = torch.cat([x_aug_cpu[nodes_t].float(), dist_oh, anchor_flag], dim=-1).to(device)

        ei = torch.stack([src, dst]).to(device) if ei_src else \
             torch.zeros(2, 0, dtype=torch.long, device=device)

        H = X
        for mlp in self.gin:
            agg = torch.zeros_like(H)
            if ei.size(1) > 0:
                agg.scatter_add_(0, ei[1].unsqueeze(1).expand(-1, H.size(1)), H[ei[0]])
            H = mlp(H + agg)

        sub_t = torch.tensor(sub_assign, dtype=torch.long, device=device)
        if self.pool == 'anchor':
            anchors_t = torch.tensor(anchors, dtype=torch.long, device=device)
            z_sub = H[anchors_t]
        elif self.pool == 'attn':
            # per-node attention logits → softmax within each subgraph → weighted sum
            # max subtraction done on cpu to avoid mps scatter_reduce('max') limitation
            logits  = self.attn_net(H).squeeze(-1).clamp(-10, 10)           # (N,) clamped for gradient stability
            logits_cpu = logits.cpu()
            sub_cpu    = sub_t.cpu()
            max_l   = torch.zeros(n_sub).scatter_reduce_(
                0, sub_cpu, logits_cpu, reduce='amax', include_self=True
            )[sub_cpu].to(device)
            exp_l   = torch.exp(logits - max_l)
            sum_e   = scatter(exp_l, sub_t, dim=0, dim_size=n_sub, reduce='sum')[sub_t]
            attn    = (exp_l / sum_e).unsqueeze(-1)                        # (N, 1)
            z_sub   = scatter(H * attn, sub_t, dim=0, dim_size=n_sub, reduce='sum')
        else:
            z_sub = scatter(H, sub_t, dim=0, dim_size=n_sub, reduce='mean')

        # shell-specific reconstruction target: mean features of nodes at distance exactly `hop`
        # (the outermost shell). falls back to full-subgraph mean for subgraphs with no shell nodes.
        x_orig   = x_aug_cpu[nodes_t][:, :self.n_features].float()
        sub_t_cpu = sub_t.cpu()
        shell_mask = (d == hop)  # (N,) — outermost shell nodes
        if shell_mask.any():
            shell_feat  = x_orig[shell_mask]
            shell_subs  = sub_t_cpu[shell_mask]
            x_shell     = torch.zeros(n_sub, self.n_features)
            shell_count = torch.zeros(n_sub)
            x_shell.scatter_add_(0, shell_subs.unsqueeze(1).expand_as(shell_feat), shell_feat)
            shell_count.scatter_add_(0, shell_subs, torch.ones(shell_mask.sum()))
            has_shell = shell_count > 0
            x_shell[has_shell] /= shell_count[has_shell].unsqueeze(1)
            x_full    = scatter(x_orig, sub_t_cpu, dim=0, dim_size=n_sub, reduce='mean')
            x_target  = torch.where(has_shell.unsqueeze(1), x_shell, x_full)
        else:
            x_target = scatter(x_orig, sub_t_cpu, dim=0, dim_size=n_sub, reduce='mean')

        return z_sub, nodes_t.to(device), sub_t, sg_data, x_target

    def forward(self, x, edge_index, batch, **_):
        n = x.size(0)
        device = x.device
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes=n)

        x_aug = self._wl_augment(x.float(), edge_index)
        adj   = self._build_adj(edge_index, n)

        # sample anchors once so all levels share the same set → aligned z_sub across levels
        anchor_dict = self._sample_anchors(batch)

        # first pass: collect raw embeddings from all levels
        raw = []
        for hop in self.walk_lens:
            subgraph_list = self._sample_subgraphs(adj, batch, hop=hop, anchor_dict=anchor_dict)
            sub_batch = (torch.tensor([g for _, _, g in subgraph_list], dtype=torch.long, device=device)
                         if subgraph_list else torch.zeros(0, dtype=torch.long, device=device))
            z_sub, flat_nodes, flat_subs, sg_data, x_target = self._embed_subgraphs(
                x_aug, adj, subgraph_list, device, hop=hop
            )
            raw.append((z_sub, flat_nodes, flat_subs, sub_batch, sg_data, x_target))

        levels = []
        for lvl, (z_sub, flat_nodes, flat_subs, sub_batch, sg_data, x_target) in enumerate(raw):
            n_sub = z_sub.size(0)
            scores = (torch.sigmoid(self.score_nets[lvl](z_sub).squeeze(-1))
                      if n_sub > 0 else torch.zeros(0, device=device))
            node_to_sub = torch.zeros(n, dtype=torch.long, device=device)
            if flat_nodes.numel() > 0:
                order = torch.argsort(scores[flat_subs])
                node_to_sub.scatter_(0, flat_nodes[order], flat_subs[order])
            levels.append({
                'z_sub':       z_sub,
                'scores':      scores,
                'node_to_sub': node_to_sub,
                'flat_nodes':  flat_nodes,
                'flat_subs':   flat_subs,
                'sub_batch':   sub_batch,
                'sg_data':     sg_data,
                'x_target':    x_target,
            })

        return levels

    @staticmethod
    def _distance_density(X, X_ref, k):
        """distance to k-th nearest neighbor (ported from main branch)."""
        from scipy.spatial import KDTree
        knn = KDTree(X_ref.cpu().detach().numpy())
        R, _ = knn.query(X.cpu().detach().numpy(), k=k)
        return torch.tensor(R[:, k - 1], dtype=torch.float32, requires_grad=False)

    def freq_loss(self, levels_pos, levels_neg, beta=1.0, lam=1.0, k=30, **_):
        """knn density contrast (ported from main branch MotiFiestaModel.freq_loss).

        f_pos[i] = distance to k-th nearest pos neighbor (low when pos cluster is tight)
        f_neg[i] = distance to k-th nearest neg neighbor (low when neg is nearby)
        loss = -s * exp(-beta * (f_pos - f_neg)) + lam * s²
        """
        device = next(self.parameters()).device
        n_levels = min(len(levels_pos), len(levels_neg))
        if n_levels == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for t in range(n_levels):
            x_pos = torch.nn.functional.normalize(levels_pos[t]['z_sub'], dim=-1)
            x_neg = torch.nn.functional.normalize(levels_neg[t]['z_sub'].detach(), dim=-1)
            s     = levels_pos[t]['scores']

            if x_pos.size(0) < 2 or x_neg.size(0) < 1:
                continue
            k_eff = min(k, x_pos.size(0) - 1, x_neg.size(0))
            if k_eff < 1:
                continue

            f_pos = self._distance_density(x_pos, x_pos, k_eff + 1)
            f_neg = self._distance_density(x_pos, x_neg, k_eff)
            f_pos, f_neg = f_pos.to(device), f_neg.to(device)

            tot_loss = tot_loss + (-s * torch.exp(-beta * (f_pos - f_neg))).mean()
            tot_loss = tot_loss + lam * s.pow(2.0).mean()
            n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active

    def rec_loss(self, levels_pos):
        """warmup loss: reconstruct mean degree distribution from z_sub (ported from main branch).

        trains gin to produce embeddings that reflect the degree structure of each subgraph.
        also naturally bounds gin output magnitudes for stable freq_loss training.
        """
        device = next(self.parameters()).device
        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0
        for lvl_dict in levels_pos:
            z   = lvl_dict['z_sub']
            tgt = lvl_dict['x_target'].to(device)
            tot_loss = tot_loss + torch.nn.functional.mse_loss(self.x_decoder(z), tgt)
            n_active += 1
        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active
