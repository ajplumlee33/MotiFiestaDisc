import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce


def _mlp(in_dim, out_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.ELU(),
        torch.nn.Linear(out_dim, out_dim),
    )


def _greedy_contract(edge_index, scores, n_nodes):
    """greedy edge contraction ranked by score, vectorized.

    mutual-best rounds: selects all edges where both endpoints' best remaining
    incident edge is that edge. equivalent to sequential greedy.
    runs on cpu — mps scatter_ uses first-write-wins for duplicate indices which
    breaks the state tracking (taken/cluster). cpu has correct last-write-wins.
    """
    orig_device = scores.device
    edge_index = edge_index.cpu()
    scores = scores.cpu()
    device = torch.device('cpu')
    n_edges = edge_index.size(1)

    taken = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    cluster = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    next_id = 0
    selected_parts = []

    order = torch.argsort(scores, descending=True)
    u_ord = edge_index[0, order]
    v_ord = edge_index[1, order]
    ranks = torch.arange(n_edges, device=device)
    INF = n_edges

    active = torch.ones(n_edges, dtype=torch.bool, device=device)
    both_nodes = torch.cat([u_ord, v_ord])

    while active.any():
        eff_rank = torch.where(active, ranks, torch.full_like(ranks, INF))
        both_eff = torch.cat([eff_rank, eff_rank])

        # min-scatter via "last write wins": sort descending so inactive (INF) writes first,
        # then active worst→best; lowest rank is last and survives per node.
        # avoids scatter_reduce_(amin) which is unimplemented on MPS.
        desc = torch.argsort(both_eff, descending=True)
        best = torch.full((n_nodes,), INF, dtype=torch.long, device=device)
        best.scatter_(0, both_nodes[desc], both_eff[desc])

        sel_mask = active & (best[u_ord] == ranks) & (best[v_ord] == ranks)
        sel_orig = order[sel_mask]
        if sel_orig.numel() == 0:
            break

        sel_u = edge_index[0, sel_orig]
        sel_v = edge_index[1, sel_orig]
        n_sel = sel_orig.numel()
        ids = torch.arange(next_id, next_id + n_sel, device=device)
        cluster[sel_u] = ids
        cluster[sel_v] = ids
        taken[sel_u] = True
        taken[sel_v] = True
        selected_parts.append(sel_orig)
        next_id += n_sel

        active = active & ~taken[u_ord] & ~taken[v_ord]

    singletons = (~taken).nonzero(as_tuple=False).squeeze(-1)
    if singletons.numel() > 0:
        n_sing = singletons.numel()
        cluster[singletons] = torch.arange(next_id, next_id + n_sing, device=device)
        next_id += n_sing

    selected = torch.cat(selected_parts) if selected_parts else torch.zeros(0, dtype=torch.long, device=device)
    return cluster.to(orig_device), next_id, taken.to(orig_device), selected.to(orig_device)


def _knn_radius(X, X_ref, k):
    """euclidean distance to the k-th nearest neighbor in X_ref for each point in X.

    matches main branch distance_density (KDTree, euclidean). no normalization —
    magnitude encodes degree structure and must not be discarded.
    """
    with torch.no_grad():
        dists = torch.cdist(X, X_ref)
        knn, _ = dists.topk(k, dim=1, largest=False)
    return knn[:, k - 1]


class GINPool(torch.nn.Module):
    """one contraction level: GIN message passing → transform → score → greedy contract.

    per-level GIN aggregates from all neighboring supernodes at the current graph scale,
    giving each level structural awareness the main branch's linear transform lacks.
    rec_loss (geomloss WWL) supervises x_merged at each level; freq_loss trains
    score_net only (GIN is frozen to protect learned embeddings).
    """

    def __init__(self, dim, gin_layers=1):
        super().__init__()
        self.gin = torch.nn.ModuleList(
            [GINConv(_mlp(dim, dim)) for _ in range(gin_layers)]
        )
        self.transform = torch.nn.Linear(dim, dim)
        self.score_net = torch.nn.Linear(dim, 1)

    def forward(self, feats, edge_index, node_batch, dummy=False):
        """contract one level. returns None if no edges remain."""
        n_nodes = feats.size(0)
        device = feats.device
        dim = feats.size(1)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, n_nodes, n_nodes)
        if edge_index.size(1) == 0:
            return None

        # per-level message passing on the current (contracted) graph
        for gin_layer in self.gin:
            feats = F.elu(gin_layer(feats, edge_index))

        feats_t = self.transform(feats)
        if dummy:
            # random bernoulli skip — matches main branch's `if r > 0.5: skip` per edge.
            # truly random contractions (not biased by mutual-best on uniform scores).
            edge_score = torch.bernoulli(
                torch.full((edge_index.size(1),), 0.5, device=device)
            ).clamp(min=1e-6)
        else:
            edge_score = torch.sigmoid(
                self.score_net(feats_t[edge_index[0]] + feats_t[edge_index[1]]).squeeze(-1)
            )
        edge_batch = node_batch[edge_index[0]]

        new_cluster, n_new, _, selected = _greedy_contract(
            edge_index, edge_score.detach(), n_nodes
        )

        # scatter all nodes to their supernodes: contracted pairs sum feats_t[u]+feats_t[v],
        # singletons pass through feats_t[node]. no conditionals, no device syncs.
        supernode_emb = torch.zeros(n_new, dim, device=device).scatter_add(
            0, new_cluster.unsqueeze(1).expand(-1, dim), feats_t
        )

        # contracted supernode score = the selected edge's score; singletons score 0
        # (matches main branch EdgePooling: total_sigma accumulates contraction edges only)
        supernode_score = torch.zeros(n_new, device=device)
        contracted_ids = new_cluster[edge_index[0, selected]]
        supernode_score.scatter_(0, contracted_ids, edge_score[selected])

        x_edge = feats_t[edge_index[0]] + feats_t[edge_index[1]]

        return (supernode_emb, supernode_score, new_cluster, n_new,
                edge_score, edge_batch, x_edge, edge_index)


class MotiFiestaDisc(torch.nn.Module):
    """motif discovery model: input projection → stacked GINPool layers.

    each GINPool applies GIN message passing on the contracted graph at that scale,
    then scores edges, contracts greedily, and produces supernode embeddings.
    rec_loss (GeomLoss WWL): trains GIN + transform via OT-based kernel matching.
    freq_loss (kNN density contrast): trains score_net only — GIN frozen to protect
    the structural embeddings rec_loss builds.
    """

    def __init__(self, n_features=61, dim=32, depth=4, gin_layers=1,
                 steps=None, wl_hops=1, rec_kernel='cosine', **kwargs):
        super().__init__()
        if steps is not None:
            depth = steps
        self.dim = dim
        self.depth = depth
        self.wl_hops = wl_hops
        self.n_features = n_features
        self.rec_kernel = rec_kernel

        self.input_proj = torch.nn.Linear(n_features * (1 + wl_hops), dim)

        # per-level pool layers: each owns gin + transform + score_net
        self.pool_layers = torch.nn.ModuleList(
            [GINPool(dim, gin_layers=gin_layers) for _ in range(depth)]
        )

        self.layers = self.pool_layers  # backward compat alias

    @property
    def hidden_dim(self):
        return self.dim

    def _wl_augment(self, x, edge_index):
        """augment node features with k-hop neighbor sums."""
        src, dst = edge_index[0], edge_index[1]
        parts = [x]
        cur = x
        for _ in range(self.wl_hops):
            nbr = torch.zeros_like(x).scatter_add(
                0, dst.unsqueeze(1).expand(-1, x.size(1)), cur[src]
            )
            parts.append(nbr)
            cur = nbr
        return torch.cat(parts, dim=-1)

    def forward(self, x, edge_index, _batch, dummy=False):
        n = x.size(0)
        device = x.device

        # project augmented input features — per-level GINs handle message passing
        x_aug = self._wl_augment(x, edge_index)
        feats = F.elu(self.input_proj(x_aug)).clamp(-100, 100)

        cur_ei = edge_index
        cum_assign = torch.arange(n, device=device)
        node_batch = _batch

        xx = [feats]
        pp = [torch.ones(n, device=device)]
        ee = [edge_index]
        internals = [{
            'x_merged': feats,
            'x_edge': torch.zeros(0, self.dim, device=device),
            'edge_scores': torch.ones(n, device=device),
            'edge_scores_raw': torch.zeros(0, device=device),
            'edge_batch': torch.zeros(0, dtype=torch.long, device=device),
            'cum_assign': cum_assign.clone(),
        }]

        for pool in self.pool_layers:
            n_nodes = feats.size(0)
            result = pool(feats, cur_ei, node_batch, dummy=dummy)
            if result is None:
                break

            supernode_emb, supernode_score, new_cluster, n_new, \
                edge_score, edge_batch_t, x_edge, clean_ei = result

            feats = supernode_emb
            cum_assign = new_cluster[cum_assign]
            cur_ei = new_cluster[clean_ei]
            node_batch = node_batch.new_zeros(n_new).scatter_(0, new_cluster, node_batch)

            xx.append(feats)
            pp.append(supernode_score)
            ee.append(cur_ei)
            internals.append({
                'x_merged': supernode_emb,
                'x_edge': x_edge,
                'edge_scores': supernode_score,
                'edge_scores_raw': edge_score,
                'edge_batch': edge_batch_t,
                'cum_assign': cum_assign.clone(),
            })

            if n_new >= n_nodes:
                break

        merge_info = {
            'cumulative_assignments': [d['cum_assign'] for d in internals],
            'spotlights': None,
            'tree': None,
        }

        return xx, pp, ee, None, merge_info, internals

    def _wwl_gram(self, hop_feats, cum_assign, n_sup, perm=None):
        """WWL kernel gram matrix: K[i,j] = exp(-Wasserstein(spotlight_i_WL, spotlight_j_WL)).

        groups original-node WL features by spotlight, pads to fixed size, then computes
        all-pairs OT distances via GeomLoss SamplesLoss in one batched GPU call.
        perm: optional [k] index — subsample supernodes first (keeps cost O(k²)).
        """
        from geomloss import SamplesLoss
        device = hop_feats[0].device
        cum_cpu = cum_assign.cpu()

        if perm is not None:
            perm_cpu = perm.cpu()
            old_to_new = torch.full((n_sup,), -1, dtype=torch.long)
            old_to_new[perm_cpu] = torch.arange(len(perm_cpu))
            node_mask = old_to_new[cum_cpu] >= 0
            new_cum = old_to_new[cum_cpu[node_mask]]
            node_emb = torch.cat([h[node_mask.to(device)] for h in hop_feats], dim=-1)
            n_out = len(perm_cpu)
        else:
            new_cum = cum_cpu
            node_emb = torch.cat(hop_feats, dim=-1)
            n_out = n_sup

        # vectorized padding via sort + cummax within-group positions
        order = torch.argsort(new_cum, stable=True)
        sorted_cum = new_cum[order]
        sorted_emb = node_emb.cpu()[order]

        changes = torch.cat([torch.tensor([True]), sorted_cum[1:] != sorted_cum[:-1]])
        starts = torch.zeros(len(new_cum), dtype=torch.long)
        starts[changes] = changes.nonzero(as_tuple=False).squeeze(-1)
        within_pos = (torch.arange(len(new_cum)) - starts.cummax(0).values).clamp(max=7)

        max_size = min(int(within_pos.max().item()) + 1, 8) if len(within_pos) > 0 else 1
        feat_dim = node_emb.size(-1)
        padded = torch.zeros(n_out, max_size, feat_dim)
        pad_mask = torch.zeros(n_out, max_size, dtype=torch.bool)
        padded[sorted_cum, within_pos] = sorted_emb
        pad_mask[sorted_cum, within_pos] = True

        # GeomLoss all-pairs OT in one batched call
        padded, pad_mask = padded.to(device), pad_mask.to(device)
        n, m, _ = padded.shape
        weights = pad_mask.float() / pad_mask.float().sum(-1, keepdim=True).clamp(min=1)
        X = padded.unsqueeze(1).expand(n, n, m, -1).reshape(n*n, m, -1).contiguous()
        Y = padded.unsqueeze(0).expand(n, n, m, -1).reshape(n*n, m, -1).contiguous()
        Xw = weights.unsqueeze(1).expand(n, n, m).reshape(n*n, m).contiguous()
        Yw = weights.unsqueeze(0).expand(n, n, m).reshape(n*n, m).contiguous()
        dist = SamplesLoss('sinkhorn', p=1, blur=0.05, backend='auto')(Xw, X, Yw, Y)
        return torch.exp(-dist.clamp(min=0)).reshape(n, n)

    def rec_loss(self, xx, internals_pos, batch_pos, num_nodes=40):
        """structural anchor loss supervising supernode embeddings via kernel matching.

        rec_kernel='geomloss': K_true = exp(-Wasserstein(spotlight_WL_features))
          computed via GeomLoss SamplesLoss. level 0: RBF on raw features.
          level t>0: internal-edge 4-hop WL on spotlight.  dummy warmup keeps contractions
          stable so K_true doesn't shift as GIN trains.

        rec_kernel='cosine': K_true = cosine gram of scatter_sum WL hop features.

        K_predict = cosine gram of internals[t]['x_merged'] at each level —
        trains GIN and transforms; score_net receives no gradient from rec_loss.
        """
        device = next(self.parameters()).device

        with torch.no_grad():
            x_base = batch_pos.x.to(device).float()
            ei = batch_pos.edge_index.to(device)
            src, dst = ei[0], ei[1]
            n_feat = x_base.size(1)

            wl_hops_list = [x_base]
            cur = x_base
            for _ in range(self.wl_hops):
                nbr = torch.zeros_like(cur).scatter_add_(
                    0, dst.unsqueeze(1).expand(-1, n_feat), cur[src]
                )
                wl_hops_list.append(nbr)
                cur = nbr

        def _cosine_gram(hop_feats):
            """sum of per-hop cosine gram matrices."""
            def _gram(h):
                h_n = F.normalize(h, dim=-1)
                return h_n @ h_n.T
            K = _gram(hop_feats[0])
            for h in hop_feats[1:]:
                K = K + _gram(h)
            return K / len(hop_feats)

        n_nodes_orig = internals_pos[0]['x_merged'].size(0)
        if n_nodes_orig < 2:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        loss = torch.zeros(1, device=device).squeeze()
        n_levels = 0

        for t, internal in enumerate(internals_pos):
            # use supernode embeddings directly — matches main branch rec_loss
            # which uses internals[level]['x_merged'] as K_predict
            z_t = internal['x_merged']
            cum_assign = internal['cum_assign'].to(device)
            n_super = z_t.size(0)

            if n_super < 2:
                continue

            # subsample supernodes — keeps Sinkhorn cost O(k^2)
            perm = None
            if n_super > num_nodes:
                perm = torch.randperm(n_super, device=device)[:num_nodes]
                z_t = z_t[perm]

            if self.rec_kernel in ('sinkhorn', 'geomloss'):
                if t == 0:
                    x_sub = x_base if perm is None else x_base[perm]
                    K_true = torch.exp(-torch.cdist(x_sub, x_sub)).to(device)
                else:
                    # internal-edge 4-hop WL — stable because dummy scores fix contractions
                    with torch.no_grad():
                        internal_mask = cum_assign[src] == cum_assign[dst]
                        src_int = src[internal_mask]
                        dst_int = dst[internal_mask]
                        sub_hops = [x_base]
                        cur = x_base
                        for _ in range(4):
                            nbr = torch.zeros(n_nodes_orig, n_feat, device=device).scatter_add_(
                                0, dst_int.unsqueeze(1).expand(-1, n_feat), cur[src_int]
                            )
                            sub_hops.append(nbr)
                            cur = nbr
                    K_true = self._wwl_gram(sub_hops, cum_assign, n_super, perm).to(device)
            else:
                if t == 0:
                    K_true = _cosine_gram(wl_hops_list)
                    if perm is not None:
                        K_true = K_true[perm][:, perm]
                else:
                    ci_feat = cum_assign.unsqueeze(1)
                    super_hops = [
                        torch.zeros(n_super, n_feat, device=device).scatter_add_(
                            0, ci_feat.expand(-1, n_feat), h)
                        for h in wl_hops_list
                    ]
                    K_true = _cosine_gram(super_hops)
                    if perm is not None:
                        K_true = K_true[perm][:, perm]

            K_predict = F.normalize(z_t, dim=-1) @ F.normalize(z_t, dim=-1).T
            loss = loss + F.mse_loss(K_predict, K_true)
            n_levels += 1

        return loss / max(n_levels, 1)

    def freq_loss(self, internals_pos, internals_neg, pp, beta=1, lam=1.0, k=30):
        """kNN density contrast loss — trains score_net (GIN frozen during this phase).

        euclidean kNN on x_edge (per-edge representations),
        scores from edge_scores_raw. gradient: freq_loss → s → score_net → transform.
        GIN weights are frozen externally in train.py to preserve rec_loss embeddings.
        """
        device = next(self.parameters()).device

        n_levels = min(len(internals_pos), len(internals_neg), len(pp))
        if n_levels <= 1:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for t in range(1, n_levels):
            x_pos = internals_pos[t]['x_edge'].detach()
            x_neg = internals_neg[t]['x_edge'].detach()
            s = internals_pos[t]['edge_scores_raw']

            if x_pos.size(0) < 2 or x_neg.size(0) < 1:
                continue

            k_eff = min(k, x_pos.size(0) - 1, x_neg.size(0))
            if k_eff < 1:
                continue

            rho_pos = _knn_radius(x_pos, x_pos, k=k_eff)
            rho_neg = _knn_radius(x_pos, x_neg, k=k_eff)
            delta_f = rho_pos - rho_neg

            tot_loss = tot_loss + (-s.clamp(min=1e-6) * torch.exp(-beta * delta_f.clamp(min=-10, max=10))).mean()
            tot_loss = tot_loss + lam * s.clamp(min=1e-6).pow(2.0).mean()
            n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active
