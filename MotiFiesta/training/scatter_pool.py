import torch
from torch_scatter import scatter_mean
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops


class ScatterPool(torch.nn.Module):
    """simplified edge contraction layer for scatter-pool architecture.

    takes Z (scatter_mean of global GIN features) as input for scoring.
    scoring is purely from WL-equivalent Z vectors
    """

    def __init__(self, dim, matching_mode='luby'):
        super().__init__()
        self.dim = dim
        self.matching_mode = matching_mode
        # sum of endpoint Z vectors → score
        self.score_net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, 1),
        )

    def forward(self, Z, edge_index, batch):
        if edge_index.size(1) == 0 or Z.size(0) < 2:
            n = Z.size(0)
            cluster = torch.arange(n, device=Z.device, dtype=torch.long)
            empty = torch.zeros(0, device=Z.device)
            return cluster, empty, empty, edge_index, batch

        raw = Z[edge_index[0]] * Z[edge_index[1]]
        logit = self.score_net(raw).squeeze(-1)
        e = torch.sigmoid(logit)

        if self.matching_mode == 'luby':
            cluster, new_ei, new_batch = self._luby(edge_index, e, Z.size(0), batch)
        else:
            cluster, new_ei, new_batch = self._greedy(edge_index, e, Z.size(0), batch)

        return cluster, e, logit, new_ei, new_batch

    def _luby(self, edge_index, edge_score, n_nodes, batch, max_rounds=20):
        device = edge_score.device

        src_full = edge_index[0]
        dst_full = edge_index[1]
        num_edges = edge_index.size(1)

        edge_key_fwd = src_full * n_nodes + dst_full
        edge_key_rev = dst_full * n_nodes + src_full
        is_sym = torch.isin(edge_key_rev, edge_key_fwd).all().item()
        canon_mask = (src_full <= dst_full) if is_sym else torch.ones(num_edges, dtype=torch.bool, device=device)
        canon_idx = canon_mask.nonzero(as_tuple=False).squeeze(-1)
        n_canon = canon_idx.size(0)

        if n_canon == 0:
            cluster = torch.arange(n_nodes, device=device, dtype=torch.long)
            new_ei, _ = coalesce(cluster[edge_index], None, n_nodes, n_nodes)
            new_ei, _ = remove_self_loops(new_ei)
            return cluster, new_ei, batch.clone()

        src = src_full[canon_idx]
        dst = dst_full[canon_idx]
        canon_score = edge_score[canon_idx]

        edges_alive = torch.rand(n_canon, device=device) < canon_score

        if device.type == 'mps':
            priority = canon_score + 1e-4 * torch.rand(n_canon, dtype=torch.float32, device=device)
        else:
            priority = canon_score.double() + 1e-9 * torch.rand(n_canon, dtype=torch.float64, device=device)

        cluster = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
        nodes_taken = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        next_id = 0

        for _ in range(max_rounds):
            if not edges_alive.any():
                break

            masked = torch.where(edges_alive, priority, torch.full_like(priority, float('-inf')))

            if device.type == 'mps':
                _nm = torch.full((n_nodes,), float('-inf'))
                _nm.scatter_reduce_(0, src.cpu(), masked.cpu(), reduce='amax', include_self=True)
                _nm.scatter_reduce_(0, dst.cpu(), masked.cpu(), reduce='amax', include_self=True)
                node_max = _nm.to(device)
            else:
                node_max = torch.full((n_nodes,), float('-inf'), dtype=torch.float64, device=device)
                node_max.scatter_reduce_(0, src, masked, reduce='amax', include_self=True)
                node_max.scatter_reduce_(0, dst, masked, reduce='amax', include_self=True)

            selected = (
                edges_alive
                & (masked == node_max[src])
                & (masked == node_max[dst])
                & (~nodes_taken[src])
                & (~nodes_taken[dst])
            )
            if not selected.any():
                break

            sel = selected.nonzero(as_tuple=False).squeeze(-1)
            n_new = sel.size(0)
            new_ids = torch.arange(next_id, next_id + n_new, device=device)
            next_id += n_new

            cluster[src[sel]] = new_ids
            cluster[dst[sel]] = new_ids
            nodes_taken[src[sel]] = True
            nodes_taken[dst[sel]] = True
            edges_alive = edges_alive & (~nodes_taken[src]) & (~nodes_taken[dst])

        untaken = (~nodes_taken).nonzero(as_tuple=False).squeeze(-1)
        if untaken.size(0) > 0:
            ids = torch.arange(next_id, next_id + untaken.size(0), device=device)
            cluster[untaken] = ids
            next_id += untaken.size(0)

        N = next_id
        new_ei, _ = coalesce(cluster[edge_index], None, N, N)
        new_ei, _ = remove_self_loops(new_ei)
        new_batch = torch.empty(N, dtype=torch.long, device=device)
        new_batch.scatter_(0, cluster, batch)

        return cluster, new_ei, new_batch

    def _greedy(self, edge_index, edge_score, n_nodes, batch):
        device = edge_score.device
        cluster = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
        taken = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        order = torch.argsort(edge_score, descending=True)
        next_id = 0

        ei_cpu = edge_index.cpu()
        scores_cpu = edge_score.detach().cpu()

        for idx in order.tolist():
            u = ei_cpu[0, idx].item()
            v = ei_cpu[1, idx].item()
            if taken[u] or taken[v]:
                continue
            if torch.rand(1).item() > scores_cpu[idx]:
                continue
            cluster[u] = next_id
            cluster[v] = next_id
            taken[u] = True
            taken[v] = True
            next_id += 1

        untaken = (~taken).nonzero(as_tuple=False).squeeze(-1)
        if untaken.size(0) > 0:
            ids = torch.arange(next_id, next_id + untaken.size(0), device=device)
            cluster[untaken] = ids
            next_id += untaken.size(0)

        N = next_id
        new_ei, _ = coalesce(cluster[edge_index], None, N, N)
        new_ei, _ = remove_self_loops(new_ei)
        new_batch = torch.empty(N, dtype=torch.long, device=device)
        new_batch.scatter_(0, cluster, batch)

        return cluster, new_ei, new_batch
