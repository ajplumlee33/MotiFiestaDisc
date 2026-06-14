import random

import torch
from torch_geometric.utils import remove_self_loops, coalesce, scatter


class MotiFiestaDisc(torch.nn.Module):
    """subgraph representation learning for unsupervised motif discovery.

    snowball sampling → induced subgraph extraction → anchor distance labeling
    → GIN on induced subgraph → mean pool → per-level score_nets.

    joint structural representation: GIN runs on the *induced* subgraph of sampled
    nodes. this is a joint representation per Srinivasan & Ribeiro 2020 —
    the representation of each anchor is a function of the joint structure of its
    sampled neighborhood, not independent per-node features.

    anchor distance labeling: each node in the induced subgraph is labeled by its
    BFS distance from the anchor (0,1,2,3+), giving every node a structural role
    relative to the center. this goes beyond 1-WL expressiveness.
    """

    def __init__(self, n_features=25, hidden_dim=32, gin_layers=2,
                 walk_lens=(4, 8, 12), n_walks=4, wl_hops=1, **_):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.walk_lens = list(walk_lens) if hasattr(walk_lens, '__iter__') else [walk_lens]
        self.n_walks = n_walks
        self.wl_hops = wl_hops

        # input to GIN: WL-augmented features + 4-dim anchor distance one-hot (hop 0/1/2/3+)
        wl_dim = n_features * (1 + wl_hops)
        in_dim = wl_dim + 4

        # GIN: standard sum aggregation with MLP per layer
        self.gin = torch.nn.ModuleList()
        for i in range(gin_layers):
            d_in = in_dim if i == 0 else hidden_dim
            self.gin.append(torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            ))

        self.score_nets = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, 1) for _ in self.walk_lens
        ])

    def _wl_augment(self, x, edge_index):
        """k-hop WL neighbor sum augmentation on the full graph."""
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

    def _sample_subgraphs(self, adj, batch, walk_len, max_per_graph=200):
        """snowball sampling. returns list of (anchor, sorted_node_list)."""
        batch_cpu = batch.cpu()
        all_subgraphs = []
        for g in batch_cpu.unique():
            node_mask = batch_cpu == g
            nodes = node_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            seen = set()
            for anchor in nodes:
                if len(seen) >= max_per_graph:
                    break
                for _ in range(self.n_walks):
                    visited_list = [anchor]
                    visited_set = {anchor}
                    for _ in range(walk_len):
                        cur = random.choice(visited_list)
                        nbrs = adj[cur]
                        if not nbrs:
                            break
                        nxt = random.choice(nbrs)
                        if nxt not in visited_set:
                            visited_list.append(nxt)
                            visited_set.add(nxt)
                    key = frozenset(visited_set)
                    if key not in seen:
                        seen.add(key)
                        all_subgraphs.append((anchor, sorted(visited_set)))
        return all_subgraphs

    def _embed_subgraphs(self, x_aug, adj, subgraph_list, device):
        """induced subgraph GIN: build batched graph → anchor flag → GIN → mean pool."""
        if not subgraph_list:
            emp = torch.zeros(0, dtype=torch.long, device=device)
            return torch.zeros(0, self.hidden_dim, device=device), emp, emp

        n_sub = len(subgraph_list)
        x_aug_cpu = x_aug.detach().cpu()
        ei_src, ei_dst, anchors = [], [], []
        sub_assign, flat_nodes = [], []
        node_offset = 0

        for s_idx, (au, nodes) in enumerate(subgraph_list):
            node_set = set(nodes)
            local_idx = {v: i for i, v in enumerate(nodes)}
            for v in nodes:
                lv = local_idx[v]
                for w in adj[v]:
                    if w in node_set:
                        ei_src.append(lv + node_offset)
                        ei_dst.append(local_idx[w] + node_offset)
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

        dist_oh = torch.zeros(N, 4)
        dist_oh.scatter_(1, bfs(anchors).unsqueeze(1), 1.0)

        nodes_t = torch.tensor(flat_nodes, dtype=torch.long)
        X_wl = x_aug_cpu[nodes_t].float()

        X  = torch.cat([X_wl, dist_oh], dim=-1).to(device)
        ei = torch.stack([src, dst]).to(device) if ei_src else \
             torch.zeros(2, 0, dtype=torch.long, device=device)

        H = X
        for mlp in self.gin:
            agg = torch.zeros_like(H)
            if ei.size(1) > 0:
                agg.scatter_add_(0, ei[1].unsqueeze(1).expand(-1, H.size(1)), H[ei[0]])
            H = mlp(H + agg)

        sub_t = torch.tensor(sub_assign, dtype=torch.long, device=device)
        z_sub = scatter(H, sub_t, dim=0, dim_size=n_sub, reduce='mean')
        return z_sub, nodes_t.to(device), sub_t

    def forward(self, x, edge_index, _batch, **_):
        n = x.size(0)
        device = x.device
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes=n)

        x_aug = self._wl_augment(x.float(), edge_index)
        adj   = self._build_adj(edge_index, n)

        all_z, all_scores, all_ei, all_internals, all_mh = [], [], [], [], []

        for lvl, wl in enumerate(self.walk_lens):
            subgraph_list = self._sample_subgraphs(adj, _batch, walk_len=wl)
            z_sub, flat_nodes, flat_subs = self._embed_subgraphs(
                x_aug, adj, subgraph_list, device
            )
            scores = torch.sigmoid(self.score_nets[lvl](z_sub).squeeze(-1))

            node_to_sub = torch.zeros(n, dtype=torch.long, device=device)
            if flat_nodes.numel() > 0:
                order = torch.argsort(scores[flat_subs])
                node_to_sub.scatter_(0, flat_nodes[order], flat_subs[order])

            all_z.append(z_sub)
            all_scores.append(scores)
            all_ei.append(edge_index)
            all_mh.append(node_to_sub)
            all_internals.append({
                'z_sub': z_sub,
                'scores': scores,
                'node_to_sub': node_to_sub,
            })

        merge_info = {'node_to_sub': all_mh}
        return all_z, all_scores, all_ei, None, merge_info, all_internals

    def freq_loss(self, internals_pos, internals_neg, pp, beta=1, lam=1.0, k=30):
        """kNN density contrast. GIN and score_nets train jointly via score gradient."""
        device = next(self.parameters()).device
        n_levels = min(len(internals_pos), len(internals_neg), len(pp))
        if n_levels == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for t in range(n_levels):
            # kNN distances: detached, on CPU (O(n²), no gradient needed)
            x_pos = internals_pos[t]['z_sub'].detach().cpu()
            x_neg = internals_neg[t]['z_sub'].detach().cpu()
            s = internals_pos[t]['scores']  # gradient flows through GIN here

            if x_pos.size(0) < 2 or x_neg.size(0) < 1:
                continue
            k_eff = min(k, x_pos.size(0) - 1, x_neg.size(0))
            if k_eff < 1:
                continue

            X_sq  = (x_pos ** 2).sum(1, keepdim=True)
            Xr_sq = (x_neg ** 2).sum(1, keepdim=True)
            rho_pos = (X_sq + X_sq.T  - 2.0 * (x_pos @ x_pos.T)).clamp(min=0).topk(
                k_eff + 1, largest=False).values[:, 1:].mean(1).sqrt()
            rho_neg = (X_sq + Xr_sq.T - 2.0 * (x_pos @ x_neg.T)).clamp(min=0).topk(
                k_eff, largest=False).values.mean(1).sqrt()
            delta_f = rho_pos - rho_neg

            neg_center = x_neg.mean(0)
            struct_dist = (x_pos - neg_center).norm(dim=-1)
            struct_w = struct_dist / struct_dist.mean().clamp(min=1e-8)

            delta_f = delta_f.to(device)
            struct_w = struct_w.to(device)

            tot_loss = tot_loss + (
                -struct_w * s.clamp(min=1e-6) *
                torch.exp(-beta * delta_f.clamp(min=-10, max=10))
            ).mean()
            tot_loss = tot_loss + lam * s.clamp(min=1e-6).pow(2.0).mean()
            n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / max(n_active, 1)
