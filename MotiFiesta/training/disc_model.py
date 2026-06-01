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
    O(rounds) gpu syncs instead of O(n_edges); rounds ~2-4 in practice.
    returns cluster [n_nodes], n_clusters, taken [n_nodes], selected [k] (tensor).
    """
    device = scores.device
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
    # concatenate u and v endpoint lists once; reused every round
    both_nodes = torch.cat([u_ord, v_ord])  # [2*n_edges]

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
    return cluster, next_id, taken, selected


def _knn_radius(X, X_ref, k):
    """kth nearest neighbor distance for each point in X against X_ref.

    for normalized vectors, euclidean knn = cosine knn (||x-y||^2 = 2-2cos).
    matmul is significantly faster than cdist on mps.
    returns negative cosine similarity (so smaller = closer, same sign as distance).
    """
    with torch.no_grad():
        neg_sim = -(X @ X_ref.T)
        knn, _ = neg_sim.topk(k, dim=1, largest=False)
    return knn[:, k - 1]


class MotiFiestaDisc(torch.nn.Module):
    """flat continuous embedding space for motif discovery.

    GIN runs once on the original graph, mapping nodes onto a unit sphere.
    per-level transforms adapt the embedding for scoring at each depth.
    edge scores drive greedy contraction; merged supernodes stay in the same
    flat space via normalize(z_u + z_v).

    freq_loss (density contrast pos vs neg) is the sole training signal:
    tight pos cluster → small kNN radius → negative delta_f → pushes s high.
    """

    def __init__(self, n_features=61, dim=32, depth=4, gin_layers=2,
                 steps=None, wl_hops=1, **kwargs):
        super().__init__()
        if steps is not None:
            depth = steps
        self.dim = dim
        self.depth = depth
        self.wl_hops = wl_hops
        self.n_features = n_features

        # input_proj receives original features + wl_hops neighbor histograms
        self.input_proj = torch.nn.Linear(n_features * (1 + wl_hops), dim)

        # shared GIN — same weights at every pooling level
        self.gin = torch.nn.ModuleList(
            [GINConv(_mlp(dim, dim)) for _ in range(gin_layers)]
        )

        # per-level transforms: adapt GIN space for scoring at each depth
        # gradient path: freq_loss → s → edge_scorer → transforms[t]
        self.transforms = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim) for _ in range(depth)]
        )

        # scores edges between current-level nodes/supernodes
        self.edge_scorer = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, 1),
        )

        # alias for load_model backward compat
        self.layers = self.transforms

    @property
    def hidden_dim(self):
        return self.dim

    def _encode(self, x, edge_index):
        h = x
        for layer in self.gin:
            h = F.elu(layer(h, edge_index))
        return h

    def _wl_augment(self, x, edge_index):
        """augment node features with k-hop neighbor degree histograms.

        hop i appends scatter_add of hop-(i-1) features over edges.
        gives GIN a topology-derived prior from epoch 0: K10 clique nodes,
        star hubs, and background ER nodes are immediately distinguishable.
        """
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

    def forward(self, x, edge_index, _batch, dummy=False, detach_gin=False, **kwargs):
        n = x.size(0)

        x_aug = self._wl_augment(x, edge_index)
        h = F.elu(self.input_proj(x_aug))
        z = F.normalize(self._encode(h, edge_index).clamp(-100, 100), dim=-1)
        if detach_gin:
            z = z.detach()

        feats = z
        cur_ei = edge_index
        cum_assign = torch.arange(n, device=x.device)
        # node_batch tracks which graph each current-level node/supernode belongs to
        node_batch = _batch

        # level 0: raw GIN embeddings, constant scores — no edge embeddings yet
        xx = [z]
        pp = [torch.ones(n, device=x.device)]
        ee = [edge_index]
        internals = [{
            'x_merged': z,
            'x_edge': torch.zeros(0, self.dim, device=x.device),
            'edge_scores': torch.ones(n, device=x.device),
            'edge_scores_raw': torch.zeros(0, device=x.device),
            'edge_batch': torch.zeros(0, dtype=torch.long, device=x.device),
            'cum_assign': cum_assign.clone(),
        }]

        for t in range(self.depth):
            n_nodes = feats.size(0)

            cur_ei, _ = remove_self_loops(cur_ei)
            cur_ei, _ = coalesce(cur_ei, None, n_nodes, n_nodes)

            if cur_ei.size(1) == 0:
                break

            feats_t = self.transforms[t](feats)
            logit = self.edge_scorer(
                torch.cat([feats_t[cur_ei[0]], feats_t[cur_ei[1]]], dim=-1)
            ).squeeze(-1)
            edge_score = torch.sigmoid(logit)

            # record which graph each edge belongs to before contraction remaps indices
            edge_batch_t = node_batch[cur_ei[0]]

            new_cluster, n_new, taken, selected = _greedy_contract(
                cur_ei, edge_score.detach(), n_nodes
            )

            supernode_emb = torch.zeros(n_new, self.dim, device=x.device)
            supernode_score = torch.ones(n_new, device=x.device)

            if selected.numel() > 0:
                sc = edge_score[selected]
                u_idx = cur_ei[0, selected]
                v_idx = cur_ei[1, selected]
                merged = sc.unsqueeze(-1) * F.normalize(feats[u_idx] + feats[v_idx], dim=-1)
                contracted_ids = new_cluster[u_idx]
                # functional index_put preserves gradient through merged → GIN for rec_loss
                supernode_emb = supernode_emb.index_put((contracted_ids,), merged)
                supernode_score = supernode_score.index_put((contracted_ids,), sc)

            if (~taken).any():
                sing_old = (~taken).nonzero(as_tuple=False).squeeze(-1)
                sing_new = new_cluster[sing_old]
                supernode_emb = supernode_emb.index_put((sing_new,), feats[sing_old])
                # score singletons by mean incident edge score; gradient flows
                # through edge_score back to edge_scorer for singletons too
                if cur_ei.size(1) > 0:
                    node_score_sum = torch.zeros(n_nodes, device=x.device).scatter_add(
                        0, cur_ei[0], edge_score
                    ).scatter_add_(0, cur_ei[1], edge_score)
                    node_score_cnt = torch.zeros(n_nodes, device=x.device).scatter_add(
                        0, cur_ei[0], torch.ones(cur_ei.size(1), device=x.device)
                    ).scatter_add_(0, cur_ei[1], torch.ones(cur_ei.size(1), device=x.device))
                    has_edges = node_score_cnt[sing_old] > 0
                    if has_edges.any():
                        sing_with_edges = sing_new[has_edges]
                        mean_sc = node_score_sum[sing_old[has_edges]] / node_score_cnt[sing_old[has_edges]]
                        supernode_score = supernode_score.index_put((sing_with_edges,), mean_sc)

            # per-edge embeddings and raw scores for freq_loss
            x_edge = feats_t[cur_ei[0]] + feats_t[cur_ei[1]]  # shape [n_edges, dim]

            feats = F.normalize(supernode_emb, dim=-1)
            cum_assign = new_cluster[cum_assign]
            cur_ei = new_cluster[cur_ei]
            # propagate node_batch to new supernodes (edges only connect same-graph nodes)
            node_batch = node_batch.new_zeros(n_new).scatter_(0, new_cluster, node_batch)

            # project new supernodes through transforms[t] for freq_loss density contrast
            feats_next = self.transforms[t](feats)

            xx.append(feats)
            pp.append(supernode_score)
            ee.append(cur_ei)
            internals.append({
                'x_merged': supernode_emb,  # pre-norm: retains score magnitude for rec_loss
                'feats_t': feats_next,       # transform-projected supernode embeddings
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

    def rec_loss(self, xx, internals_pos, batch_pos, num_nodes=100):
        """structural anchor loss via wl-augmented gram matrix matching.

        level 0: clean gin output z vs wl features on original graph.
        level t>0: transform_t(scatter_mean(z → supernodes)) vs aggregated wl features.
        scores are never used here — no contamination during warmup.
        trains gin AND all transforms.
        """
        device = next(self.parameters()).device

        with torch.no_grad():
            x_wl = F.normalize(
                self._wl_augment(batch_pos.x.to(device), batch_pos.edge_index.to(device)),
                dim=-1,
            )

        z = internals_pos[0]['x_merged']  # raw gin output [n_nodes, dim]
        n_total = z.size(0)
        wl_dim = x_wl.size(1)
        if n_total < 2:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        loss = torch.zeros(1, device=device).squeeze()

        for t, internal in enumerate(internals_pos):
            cum_assign = internal['cum_assign'].to(device)
            n_super = int(cum_assign.max().item()) + 1

            if t == 0:
                # level 0: normalized gin output directly
                z_t = F.normalize(z, dim=-1)
                wl_t = x_wl
            else:
                # aggregate z to supernode level without score contamination
                z_sum = torch.zeros(n_super, self.dim, device=device).scatter_add_(
                    0, cum_assign.unsqueeze(1).expand(-1, self.dim), z
                )
                cnt = torch.zeros(n_super, 1, device=device).scatter_add_(
                    0, cum_assign.unsqueeze(1), torch.ones(n_total, 1, device=device)
                )
                z_agg = F.normalize(z_sum / cnt.clamp(min=1), dim=-1)
                # apply transform_t so transforms get gradients from rec_loss
                z_t = F.normalize(self.transforms[t - 1](z_agg), dim=-1)
                # aggregate wl features to supernode level (teacher, detached)
                wl_sum = torch.zeros(n_super, wl_dim, device=device).scatter_add_(
                    0, cum_assign.unsqueeze(1).expand(-1, wl_dim), x_wl
                )
                wl_t = F.normalize(wl_sum / cnt.clamp(min=1), dim=-1)

            n_t = z_t.size(0)
            if n_t > num_nodes:
                perm = torch.randperm(n_t, device=device)[:num_nodes]
                z_t = z_t[perm]
                wl_t = wl_t[perm]

            K_true = wl_t @ wl_t.T
            K_predict = z_t @ z_t.T
            loss = loss + F.mse_loss(K_predict, K_true)

        return loss / len(internals_pos)

    def freq_loss(self, internals_pos, internals_neg, pp, beta=1, lam=0.1, k=30, **kwargs):
        """per-level density contrast on supernode embeddings, faithful to original model.

        uses x_merged (supernode embeddings) and pp (supernode scores) at each level,
        matching the original motifiesta design. pool sizes are naturally equal since
        pos and neg graphs have the same node count — rewire only changes edge count.
        gradient flows through pp (supernode scores) → edge_scorer → transforms.
        """
        device = next(self.parameters()).device

        n_levels = min(len(internals_pos), len(internals_neg), len(pp))
        if n_levels <= 1:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        tot_loss = torch.zeros(1, device=device).squeeze()
        n_active = 0

        for t in range(1, n_levels):
            # feats_t: transforms[t-1] applied to post-contraction supernodes at this level
            x_pos = F.normalize(internals_pos[t]['feats_t'], dim=-1).detach()
            x_neg = F.normalize(internals_neg[t]['feats_t'], dim=-1).detach()
            s = pp[t]

            if x_pos.size(0) < 2 or x_neg.size(0) < 1:
                continue

            k_eff = min(k, x_pos.size(0) - 1, x_neg.size(0))
            if k_eff < 1:
                continue

            rho_pos = _knn_radius(x_pos, x_pos, k=k_eff)
            rho_neg = _knn_radius(x_pos, x_neg, k=k_eff)
            delta_f = rho_pos - rho_neg

            tot_loss = tot_loss + (-s.clamp(min=1e-6) * torch.exp(-beta * delta_f.clamp(max=20))).mean()
            tot_loss = tot_loss + lam * s.clamp(min=1e-6).pow(2.0).mean()
            n_active += 1

        if n_active == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()
        return tot_loss / n_active
