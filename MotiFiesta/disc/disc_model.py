import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, coalesce, scatter
from sklearn.neighbors import KDTree


class MotiFiestaDisc(torch.nn.Module):
    """set-based induced subgraph representation for unsupervised motif discovery.

    sampling (set-based, not node-based):
      for each graph, sample n_samples random connected k-subsets via bfs from a
      random start node, with k ~ uniform(3, k_max). no bijection between nodes and
      subgraphs — escapes frasca et al. 2022 theorem 6, which bounds all node-based
      (bijection) policies to 3-wl.

    joint representation (srinivasan & ribeiro 2020):
      gin runs on the full induced subgraph g[s]. message passing within g[s] means
      each h_v is influenced by all other nodes in s — z(s) is a genuinely joint
      function of the set, not a decomposable mean of independent node embeddings.

    training:
      graph ae warmup: for each induced subgraph g[s], predict whether each node pair
      (u,v) is connected via bce(sigmoid(h_u · h_v), adj[u,v]). clique subgraphs have
      dense adjacency → gin learns high mutual inner products for clique nodes. background
      subgraphs are sparse → low inner products. no masking, no decoder parameters — just
      the inner product between gin node embeddings. gives gin strong structural gradient.

      freq phase: adds freq_loss (score_net) on top of graph ae — score_net learns which
      subgraph types appear more in pos vs neg density regions of the embedding space.

    forward returns a single-element list (one level) with:
      z_sub:      (n_sub, hidden_dim) joint subgraph embeddings
      H:          (total_flat, hidden_dim) node-level gin embeddings (flat across subgraphs)
      scores:     (n_sub,) motif membership scores (raw logits)
      node_to_sub:(n_nodes,) maps each node to its highest-scoring subgraph
      flat_nodes: (total_flat,) global node index per flat position
      flat_subs:  (total_flat,) subgraph index per flat position
      sub_batch:  (n_sub,) graph id per subgraph
      sg_data:    list of (n_nodes, local_edges, node_feats) per subgraph
      x_target:   (n_sub, n_features) mean node features per subgraph
    """

    def __init__(self, n_features=25, hidden_dim=64, gin_layers=2,
                 k_max=12, n_samples=50, wl_hops=1, pool='mean', dist_label=False, **_):
        super().__init__()
        self.n_features  = n_features
        self.hidden_dim  = hidden_dim
        self.k_max       = k_max
        self.n_samples   = n_samples
        self.wl_hops     = wl_hops
        self.pool        = pool
        self.dist_label  = dist_label

        # dist_label adds 5 one-hot distance bins (0,1,2,3,4+) to gin input
        in_dim = n_features + (5 if dist_label else 0)

        self.gin = torch.nn.ModuleList()
        for i in range(gin_layers):
            d_in = in_dim if i == 0 else hidden_dim
            self.gin.append(torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            ))

        # 3 score heads matching motifiesta's level 2/3/4 (k=3-4, 5-8, 9+)
        # mlp scorer: hidden_dim → hidden_dim//2 → 1
        def _score_mlp():
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, 1),
            )
            torch.nn.init.zeros_(mlp[2].weight)
            torch.nn.init.zeros_(mlp[2].bias)
            return mlp
        self.score_net_s = _score_mlp()
        self.score_net_m = _score_mlp()
        self.score_net_l = _score_mlp()
        self.x_decoder   = torch.nn.Linear(hidden_dim, n_features)

        if pool == 'attn':
            self.attn_net = torch.nn.Linear(hidden_dim, 1)

    def _build_adj(self, edge_index, n_nodes):
        adj = [[] for _ in range(n_nodes)]
        for u, v in zip(edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist()):
            adj[u].append(v)
        return adj

    def _bfs_k(self, start, adj, k):
        """bfs from start collecting k nodes; returns (nodes, dists) in bfs order."""
        visited = [start]
        dists   = [0]
        seen    = {start}
        qi = 0
        while len(visited) < k and qi < len(visited):
            v = visited[qi]
            d = dists[qi]
            qi += 1
            nbrs = [w for w in adj[v] if w not in seen]
            random.shuffle(nbrs)
            for w in nbrs:
                if len(visited) >= k:
                    break
                seen.add(w)
                visited.append(w)
                dists.append(d + 1)
        return visited, dists

    def _sample_induced(self, adj, batch):
        """sample n_samples random connected k-subsets per graph, k ~ uniform(3, k_max)."""
        batch_cpu = batch.cpu()
        subgraphs = []
        for g in batch_cpu.unique():
            g_int = g.item()
            nodes = (batch_cpu == g).nonzero(as_tuple=False).squeeze(-1).tolist()
            k_hi  = min(self.k_max, len(nodes))
            for _ in range(self.n_samples):
                k       = random.randint(3, k_hi)
                s       = random.choice(nodes)
                nl, dl  = self._bfs_k(s, adj, k)
                subgraphs.append((nl, dl, g_int))
        return subgraphs

    def _gin_forward(self, X, ei):
        """single gin pass on batched node features X with edge index ei."""
        H = X
        for mlp in self.gin:
            agg = torch.zeros_like(H)
            if ei.size(1) > 0:
                agg.scatter_add_(0, ei[1].unsqueeze(1).expand(-1, H.size(1)), H[ei[0]])
            H = mlp(H + agg)
        return H

    def _pool(self, H, sub_t, n_sub, device):
        """pool node embeddings H into subgraph embeddings z_sub."""
        if self.pool == 'attn':
            logits     = self.attn_net(H).squeeze(-1).clamp(-10, 10)
            logits_cpu = logits.cpu()
            sub_cpu    = sub_t.cpu()
            max_l = torch.zeros(n_sub).scatter_reduce_(
                0, sub_cpu, logits_cpu, reduce='amax', include_self=True
            )[sub_cpu].to(device)
            exp_l = torch.exp(logits - max_l)
            sum_e = scatter(exp_l, sub_t, dim=0, dim_size=n_sub, reduce='sum')[sub_t]
            attn  = (exp_l / sum_e).unsqueeze(-1)
            return scatter(H * attn, sub_t, dim=0, dim_size=n_sub, reduce='sum')
        else:
            return scatter(H, sub_t, dim=0, dim_size=n_sub, reduce='mean')

    def _embed_induced(self, x_aug, adj, subgraph_list, device):
        """gin on batched induced subgraphs g[s]; returns z_sub, node embeddings H, and supporting tensors."""
        if not subgraph_list:
            emp = torch.zeros(0, dtype=torch.long, device=device)
            return (torch.zeros(0, self.hidden_dim, device=device),
                    torch.zeros(0, self.hidden_dim, device=device),
                    emp, emp, emp, emp, emp, emp, [], torch.zeros(0, self.n_features))

        n_sub = len(subgraph_list)
        x_cpu = x_aug.detach().cpu()
        ei_src, ei_dst = [], []
        sub_assign, flat_nodes, local_pos_list = [], [], []
        flat_dists = []
        edge_subs_l, edge_lsrc_l, edge_ldst_l = [], [], []
        node_offset = 0
        sg_data = []

        for s_idx, (nodes, dists, _g) in enumerate(subgraph_list):
            node_set  = set(nodes)
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
                            edge_subs_l.append(s_idx)
                            edge_lsrc_l.append(lv)
                            edge_ldst_l.append(lw)

            nf = x_cpu[torch.tensor(nodes, dtype=torch.long)].numpy()[:, :self.n_features]
            sg_data.append((len(nodes), local_edges, nf))

            local_pos_list.extend(range(len(nodes)))
            sub_assign.extend([s_idx] * len(nodes))
            flat_nodes.extend(nodes)
            flat_dists.extend(dists)
            node_offset += len(nodes)

        src = torch.tensor(ei_src, dtype=torch.long) if ei_src else torch.zeros(0, dtype=torch.long)
        dst = torch.tensor(ei_dst, dtype=torch.long) if ei_src else torch.zeros(0, dtype=torch.long)

        nodes_t = torch.tensor(flat_nodes, dtype=torch.long)
        X       = x_cpu[nodes_t].to(device)

        if self.dist_label:
            dist_t  = torch.tensor(flat_dists, dtype=torch.long).clamp(max=4)
            dist_oh = F.one_hot(dist_t, num_classes=5).float().to(device)
            X       = torch.cat([X, dist_oh], dim=-1)
        ei      = (torch.stack([src, dst]).to(device) if ei_src
                   else torch.zeros(2, 0, dtype=torch.long, device=device))
        sub_t   = torch.tensor(sub_assign,    dtype=torch.long, device=device)
        local_t = torch.tensor(local_pos_list, dtype=torch.long, device=device)

        edge_subs = (torch.tensor(edge_subs_l, dtype=torch.long, device=device)
                     if edge_subs_l else torch.zeros(0, dtype=torch.long, device=device))
        edge_lsrc = (torch.tensor(edge_lsrc_l, dtype=torch.long, device=device)
                     if edge_lsrc_l else torch.zeros(0, dtype=torch.long, device=device))
        edge_ldst = (torch.tensor(edge_ldst_l, dtype=torch.long, device=device)
                     if edge_ldst_l else torch.zeros(0, dtype=torch.long, device=device))

        H     = self._gin_forward(X, ei)
        z_sub = self._pool(H, sub_t, n_sub, device)

        x_target = scatter(x_cpu[nodes_t].float(), sub_t.cpu(), dim=0, dim_size=n_sub, reduce='mean')

        return z_sub, H, nodes_t.to(device), sub_t, local_t, edge_subs, edge_lsrc, edge_ldst, sg_data, x_target

    def _sample_ego(self, adj, batch):
        """deterministic ego-subgraphs: each node v → induced subgraph on {v} ∪ neighbors(v)."""
        batch_cpu = batch.cpu()
        subgraphs = []
        for v in range(len(adj)):
            nodes = [v] + [w for w in adj[v]]
            if len(nodes) < 3:
                continue
            if len(nodes) > self.k_max:
                nodes = nodes[:self.k_max]
            # seed v has distance 0; all 1-hop neighbors have distance 1
            dists = [0] + [1] * (len(nodes) - 1)
            subgraphs.append((nodes, dists, batch_cpu[v].item()))
        return subgraphs

    @staticmethod
    def _bucket(k):
        """map subgraph size k to level index: 0=small(3-4), 1=medium(5-8), 2=large(9+)."""
        if k <= 4: return 0
        if k <= 8: return 1
        return 2

    def _sample_walk_pairs(self, adj, batch):
        """sample n_samples//2 pairs of bfs subgraphs per graph.
        each pair shares the same seed and same k; emitted consecutively so
        indices (2i, 2i+1) form a pair. returns (subgraphs, n_pairs).
        """
        batch_cpu = batch.cpu()
        subgraphs = []
        n_pairs   = 0
        for g in batch_cpu.unique():
            g_int = g.item()
            nodes = (batch_cpu == g).nonzero(as_tuple=False).squeeze(-1).tolist()
            k_hi  = min(self.k_max, len(nodes))
            for _ in range(self.n_samples // 2):
                k        = random.randint(3, k_hi)
                s        = random.choice(nodes)
                nl1, dl1 = self._bfs_k(s, adj, k)
                nl2, dl2 = self._bfs_k(s, adj, k)
                subgraphs.append((nl1, dl1, g_int))
                subgraphs.append((nl2, dl2, g_int))
                n_pairs += 1
        return subgraphs, n_pairs

    def forward(self, x, edge_index, batch, ego=False, walk_pairs=False, **_):
        n      = x.size(0)
        device = x.device
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes=n)

        adj = self._build_adj(edge_index, n)

        if ego:
            subgraph_list = self._sample_ego(adj, batch)
            n_pairs = 0
        elif walk_pairs:
            subgraph_list, n_pairs = self._sample_walk_pairs(adj, batch)
        else:
            subgraph_list = self._sample_induced(adj, batch)
            n_pairs = 0

        sub_batch_all = (torch.tensor([g for _, _, g in subgraph_list], dtype=torch.long, device=device)
                         if subgraph_list else torch.zeros(0, dtype=torch.long, device=device))

        z_sub, H, flat_nodes, flat_subs, local_t, edge_subs, edge_lsrc, edge_ldst, sg_data, x_target = (
            self._embed_induced(x.float(), adj, subgraph_list, device)
        )

        n_sub = z_sub.size(0)
        if n_sub > 0:
            k_sizes    = torch.tensor([d[0] for d in sg_data], dtype=torch.float, device=device)
            k_norm     = (k_sizes - 3.0) / max(self.k_max - 3, 1)
            bucket_ids = torch.tensor([self._bucket(d[0]) for d in sg_data],
                                      dtype=torch.long, device=device)
            # normalize before score_net — matches freq_loss knn normalization,
            # prevents score from tracking norm magnitude instead of structural direction
            z_sub_n = F.normalize(z_sub, dim=-1)
            scores  = torch.zeros(n_sub, device=device)
            for b, snet in enumerate([self.score_net_s, self.score_net_m, self.score_net_l]):
                mask = (bucket_ids == b)
                if mask.any():
                    scores[mask] = torch.sigmoid(snet(z_sub_n[mask]).squeeze(-1))
        else:
            k_norm = scores = bucket_ids = torch.zeros(0, device=device)

        node_to_sub = torch.zeros(n, dtype=torch.long, device=device)
        if flat_nodes.numel() > 0:
            order = torch.argsort(scores[flat_subs])
            node_to_sub.scatter_(0, flat_nodes[order], flat_subs[order])

        return [{
            'z_sub':      z_sub,
            'H':          H,
            'k_norm':     k_norm,
            'bucket_ids': bucket_ids,
            'local_t':    local_t,
            'edge_subs':  edge_subs,
            'edge_lsrc':  edge_lsrc,
            'edge_ldst':  edge_ldst,
            'scores':     scores,
            'node_to_sub':node_to_sub,
            'flat_nodes': flat_nodes,
            'flat_subs':  flat_subs,
            'sub_batch':  sub_batch_all,
            'sg_data':    sg_data,
            'x_target':   x_target,
            'n_pairs':    n_pairs,
        }]

    @staticmethod
    def _kde_density(X, X_ref, sigma=0.5):
        """gaussian kde on normalized (unit-sphere) embeddings.
        returns log density: larger = denser.
        """
        X_n   = F.normalize(X,     dim=-1)
        Xr_n  = F.normalize(X_ref, dim=-1)
        dists = torch.cdist(X_n, Xr_n)                          # (N, M)
        log_k = -dists.pow(2) / (2 * sigma ** 2)               # log of unnorm kernel
        log_density = torch.logsumexp(log_k, dim=1)            # log sum of kernels
        return log_density

    @staticmethod
    def _knn_density(X, X_ref, s_ref=None, k=20):
        """log knn density on raw (unnormalized) embeddings.

        returns log density: larger = denser. uses full euclidean distance.
        no sigma tuning: adapts to local scale. log form is numerically stable.
        s_ref: optional (m,) score weights. returns log(sum(s/R)/sum(s)).
        without s_ref: returns -log(mean(R)).
        """
        k_eff = min(k, X_ref.size(0) - 1)
        if k_eff < 1:
            return torch.zeros(X.size(0), device=X.device)
        tree = KDTree(X_ref.cpu().numpy())
        R, idx = tree.query(X.cpu().numpy(), k=k_eff)   # (N, k_eff)
        if s_ref is not None:
            s_np = s_ref.cpu().numpy()
            nbr_scores = s_np[idx]                                      # (N, k_eff)
            log_w = np.log(nbr_scores + 1e-8)                          # (N, k_eff)
            log_r = np.log(R + 1e-8)                                   # (N, k_eff)
            # log of score-weighted inverse distance: log(sum(s/R) / sum(s))
            log_density = (np.logaddexp.reduce(log_w - log_r, axis=1)
                           - np.logaddexp.reduce(log_w, axis=1))
        else:
            log_density = -np.log(R + 1e-8).mean(axis=1)               # larger = denser
        return torch.tensor(log_density, dtype=torch.float32, device=X.device)

    def rec_loss(self, levels_pos, **_):
        """shell mse warmup: predict mean node features of each subgraph from z_sub."""
        device = next(self.parameters()).device
        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0
        for lvl in levels_pos:
            z   = lvl['z_sub']
            tgt = lvl['x_target'].to(device)
            tot_loss = tot_loss + F.mse_loss(self.x_decoder(z), tgt)
            n_active += 1
        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active

    def wwl_loss(self, levels_pos, n_sample=30, **_):
        """wwl warmup: mse between pairwise cosine similarities and wwl kernel matrix.

        ports the main-branch rec_loss: supervises pairwise metric structure of the
        embedding space to match wwl graph kernel similarity instead of per-subgraph
        mean features. captures topology via wl node coloring (4 iterations).
        subsamples n_sample subgraphs per level to keep o(n^2) wwl tractable.
        """
        import networkx as nx
        from igraph import Graph as IGraph
        from wwl import wwl as wwl_kernel

        device = next(self.parameters()).device
        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for lvl in levels_pos:
            z  = lvl['z_sub']
            sg = lvl['sg_data']
            n  = z.size(0)
            if n < 2:
                continue

            m   = min(n_sample, n)
            idx = torch.randperm(n)[:m].sort().values
            z_s = z[idx]
            sgs = [sg[i] for i in idx.tolist()]

            ig_graphs, nf_list = [], []
            for (n_nodes, edges, nf) in sgs:
                G = nx.Graph()
                G.add_nodes_from(range(n_nodes))
                G.add_edges_from(edges)
                ig_graphs.append(IGraph.from_networkx(G))
                nf_list.append(nf.astype(np.float32))

            K_true = torch.tensor(
                wwl_kernel(ig_graphs, nf_list, num_iterations=3),
                dtype=torch.float32, device=device,
            )

            z_n   = F.normalize(z_s, dim=-1)
            K_pred = z_n @ z_n.t()

            loss = F.mse_loss(K_pred, K_true)
            if not torch.isnan(loss):
                tot_loss = tot_loss + loss
                n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active

    def ae_loss(self, levels_pos, ae_eps=0.0, **_):
        """graph ae: reconstruct induced subgraph adjacency via inner product decoder.

        vectorized: pad node embeddings to (n_sub, k_max, D), compute all pairwise
        cosine similarities via bmm on normalized embeddings, bce vs actual adjacency
        on valid (non-padding, off-diagonal) pairs. normalizing before bmm decouples
        reconstruction from norm — prevents motif/bg norm gap that corrupts knn density.
        """
        device = next(self.parameters()).device
        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for lvl in levels_pos:
            H         = lvl['H']         # (total_flat, D)
            sub_t     = lvl['flat_subs'] # (total_flat,)
            local_t   = lvl['local_t']   # (total_flat,)
            edge_subs = lvl['edge_subs'] # (n_edges,)
            edge_lsrc = lvl['edge_lsrc'] # (n_edges,)
            edge_ldst = lvl['edge_ldst'] # (n_edges,)
            sg_data   = lvl['sg_data']
            if H is None or H.size(0) < 2:
                continue

            n_sub  = len(sg_data)
            k_max  = max(d[0] for d in sg_data)
            D      = H.size(-1)

            # scatter node embeddings into padded tensor — no python loop
            H_pad = torch.zeros(n_sub, k_max, D, device=device)
            H_pad[sub_t, local_t] = H

            # build adjacency via indexed scatter — no python loop
            adj_pad = torch.zeros(n_sub, k_max, k_max, device=device)
            if edge_subs.numel() > 0:
                adj_pad[edge_subs, edge_lsrc, edge_ldst] = 1.0
                adj_pad[edge_subs, edge_ldst, edge_lsrc] = 1.0

            # valid mask: both nodes exist and not on diagonal
            node_counts = torch.tensor([d[0] for d in sg_data],
                                       dtype=torch.long, device=device)
            arange = torch.arange(k_max, device=device)
            in_sub = arange.unsqueeze(0) < node_counts.unsqueeze(1)  # (n_sub, k_max)
            valid  = in_sub.unsqueeze(2) & in_sub.unsqueeze(1)       # (n_sub, k_max, k_max)
            diag   = torch.eye(k_max, dtype=torch.bool, device=device).unsqueeze(0)
            valid  = valid & ~diag

            # normalize before inner product — cosine similarity not raw dot product.
            # raw inner product rewards high norms for dense (clique) subgraphs,
            # creating a 10x motif/bg norm gap that corrupts knn density in freq phase.
            H_pad_n = F.normalize(H_pad, dim=-1)
            logits = torch.bmm(H_pad_n, H_pad_n.transpose(1, 2))  # (n_sub, k_max, k_max)

            target = adj_pad[valid]
            if ae_eps > 0.0:
                target = target * (1.0 - ae_eps) + (1.0 - target) * ae_eps
            loss = F.binary_cross_entropy_with_logits(
                logits[valid], target, reduction='mean'
            )
            if not torch.isnan(loss):
                tot_loss = tot_loss + loss
                n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active

    def walk_pair_loss(self, levels_pos, n_sample=30, **_):
        """walk-pair regression: mse between cosine similarity of paired subgraph embeddings
        and wwl kernel similarity. pairs are same-seed bfs walks with same k, emitted
        consecutively — indices (2i, 2i+1) form a pair. subsamples n_sample pairs for speed.
        """
        import networkx as nx
        from igraph import Graph as IGraph
        from wwl import wwl as wwl_kernel

        device = next(self.parameters()).device
        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for lvl in levels_pos:
            n_pairs = lvl.get('n_pairs', 0)
            if n_pairs < 1:
                continue
            z  = lvl['z_sub']    # (2*n_pairs, D)
            sg = lvl['sg_data']  # list of 2*n_pairs entries

            # subsample pairs to keep wwl tractable
            pair_idx = torch.randperm(n_pairs)[:min(n_sample, n_pairs)].tolist()

            # compute wwl similarity for each pair independently — 2 graphs per call
            # avoids computing the full (2*n_pairs)^2 kernel matrix
            wwl_vals = []
            for i in pair_idx:
                n1, e1, nf1 = sg[2 * i]
                n2, e2, nf2 = sg[2 * i + 1]
                G1 = nx.Graph(); G1.add_nodes_from(range(n1)); G1.add_edges_from(e1)
                G2 = nx.Graph(); G2.add_nodes_from(range(n2)); G2.add_edges_from(e2)
                K2 = wwl_kernel(
                    [IGraph.from_networkx(G1), IGraph.from_networkx(G2)],
                    [nf1.astype(np.float32), nf2.astype(np.float32)],
                    num_iterations=3,
                )
                wwl_vals.append(float(K2[0, 1]))
            wwl_sims  = torch.tensor(wwl_vals, dtype=torch.float32, device=device)
            pair_t    = torch.tensor(pair_idx, dtype=torch.long, device=device)
            i_idx     = pair_t * 2
            j_idx     = i_idx + 1

            z_n      = F.normalize(z, dim=-1)
            cos_sims = (z_n[i_idx] * z_n[j_idx]).sum(dim=-1)  # (n_pairs,)

            loss = F.mse_loss(cos_sims, wwl_sims)
            if not torch.isnan(loss):
                tot_loss = tot_loss + loss
                n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active

    def infonce_loss(self, levels_pos, tau=0.1, **_):
        """nt-xent contrastive loss using wl hash as structural type oracle.

        positive pairs: subgraphs with the same wl graph hash (topology only, 3 iterations).
        negatives: all other subgraphs in the batch.
        anchors with no positive in the batch are skipped.
        """
        import networkx as nx

        device = next(self.parameters()).device
        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for lvl in levels_pos:
            z   = lvl['z_sub']    # (n_sub, D)
            sg  = lvl['sg_data']  # list of (n_nodes, local_edges, nf)
            n   = z.size(0)
            if n < 2:
                continue

            # compute wl hash for each subgraph — topology only, no node features
            wl_hashes = []
            for (n_nodes, local_edges, _) in sg:
                G = nx.Graph()
                G.add_nodes_from(range(n_nodes))
                G.add_edges_from(local_edges)
                wl_hashes.append(nx.weisfeiler_lehman_graph_hash(G, iterations=3))

            unique_hashes = list(dict.fromkeys(wl_hashes))
            hash_to_int   = {h: i for i, h in enumerate(unique_hashes)}
            labels = torch.tensor([hash_to_int[h] for h in wl_hashes],
                                  dtype=torch.long, device=device)

            z_n  = F.normalize(z, dim=-1)
            sim  = z_n @ z_n.t() / tau  # (n, n)

            eye     = torch.eye(n, dtype=torch.bool, device=device)
            same    = labels.unsqueeze(0) == labels.unsqueeze(1)  # (n, n)
            pos_mask = same & ~eye
            has_pos  = pos_mask.any(dim=1)
            if not has_pos.any():
                continue

            sim_sel      = sim[has_pos]           # (n_pos, n)
            pos_sel      = pos_mask[has_pos]      # (n_pos, n)
            self_sel     = eye[has_pos]            # (n_pos, n)

            log_denom    = torch.logsumexp(sim_sel.masked_fill(self_sel, float('-inf')), dim=1)
            log_numer    = torch.logsumexp(sim_sel.masked_fill(~pos_sel, float('-inf')), dim=1)

            loss = -(log_numer - log_denom).mean()
            if not torch.isnan(loss):
                tot_loss = tot_loss + loss
                n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active

    def freq_loss(self, levels_pos, levels_neg, beta=1.0, lam=1.0, estimator='knn', sigma=0.5, score_weight=False, **_):
        """freq loss: stratified density ratio loss on detached embeddings.

        density is computed within each size bucket (s/m/l) separately, matching
        motifiesta's per-level design. this prevents size-frequency from confounding
        the topological frequency signal — a k=3 subgraph is only compared against
        other k=3-4 subgraphs, not against k=9+ subgraphs.

        estimator='knn': log knn density on raw embeddings (-log mean_R).
        estimator='kde': gaussian kde on normalized (unit-sphere) embeddings.
        loss = -s*(f_pos - f_neg) + lam*s^2, accumulated across active buckets.
        """
        device = next(self.parameters()).device
        n_levels = min(len(levels_pos), len(levels_neg))
        if n_levels == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for t in range(n_levels):
            z_pos      = levels_pos[t]['z_sub']
            z_neg      = levels_neg[t]['z_sub']
            s          = levels_pos[t]['scores']
            bids_pos   = levels_pos[t]['bucket_ids']
            bids_neg   = levels_neg[t]['bucket_ids']

            if z_pos.size(0) < 2 or z_neg.size(0) < 1:
                continue

            # normalize before knn so euclidean distance measures direction not norm magnitude
            x_pos = F.normalize(z_pos.detach(), dim=-1)
            x_neg = F.normalize(z_neg.detach(), dim=-1)
            s_neg = levels_neg[t]['scores']

            # stratify density by size bucket so topological frequency is not
            # confounded by size frequency
            for b in range(3):
                mask_pos = (bids_pos == b)
                mask_neg = (bids_neg == b)
                if mask_pos.sum() < 2 or mask_neg.sum() < 1:
                    continue

                xp = x_pos[mask_pos]
                xn = x_neg[mask_neg]
                sb = s[mask_pos]

                if estimator == 'kde':
                    f_pos = self._kde_density(xp, xp, sigma=sigma)
                    f_neg = self._kde_density(xp, xn, sigma=sigma)
                else:
                    if score_weight:
                        sn_b = s_neg[mask_neg]
                        f_pos = self._knn_density(xp, xp, s_ref=sb.detach())
                        f_neg = self._knn_density(xp, xn, s_ref=(1 - sn_b).detach())
                    else:
                        f_pos = self._knn_density(xp, xp)
                        f_neg = self._knn_density(xp, xn)

                diff = (f_pos - f_neg).clamp(-5, 5)
                tot_loss = tot_loss + (-sb * diff).mean()
                tot_loss = tot_loss + lam * sb.pow(2.0).mean()
                n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active

