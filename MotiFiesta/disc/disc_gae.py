"""subgraph gae encoder: gcn with edge prediction decoder.
the encoder learns subgraph embeddings where similar internal connectivity maps to
similar embedding. trained on pos subgraphs only (analogous to motifiesta rec_loss).
the edge prediction decoder forces the encoder to capture specific connectivity
patterns, not just degree statistics — enabling detection of structurally arbitrary motifs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, to_undirected, remove_self_loops


class SubgraphGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(n_layers)
        )

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return F.normalize(self.convs[-1](x, edge_index), dim=-1)

    def embed(self, x, edge_index):
        """mean-pool node embeddings into a single subgraph vector."""
        return self.encode(x, edge_index).mean(dim=0)

    def loss(self, x, edge_index, n_nodes):
        """bce edge prediction: pos edges vs randomly sampled neg edges."""
        z = self.encode(x, edge_index)
        neg_ei = negative_sampling(
            edge_index, num_nodes=n_nodes,
            num_neg_samples=max(1, edge_index.size(1)),
        )
        pos = (z[edge_index[0]] * z[edge_index[1]]).sum(-1)
        neg = (z[neg_ei[0]]    * z[neg_ei[1]]).sum(-1)
        scores = torch.cat([pos, neg])
        labels = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))]).to(scores.device)
        return F.binary_cross_entropy_with_logits(scores, labels)


    @staticmethod
    def freq_loss(z_pos, z_neg, k=10):
        """differentiable knn density contrast within a batch.
        minimizes d_pos/d_neg: pushes pos embeddings to cluster tightly
        relative to the neg cloud, analogous to motifiesta freq_loss."""
        k_pos = min(k, z_pos.size(0) - 1)
        k_neg = min(k, z_neg.size(0) - 1)
        if k_pos < 1 or k_neg < 1:
            return z_pos.sum() * 0.0
        D_pos = torch.cdist(z_pos, z_pos)
        D_neg = torch.cdist(z_pos, z_neg)
        D_pos = D_pos + torch.eye(z_pos.size(0), device=z_pos.device) * 1e9
        d_pos = D_pos.topk(k_pos, largest=False).values[:, -1]
        d_neg = D_neg.topk(k_neg, largest=False).values[:, -1]
        return (d_pos / (d_pos + d_neg + 1e-8)).mean()


def build_subgraph_tensors(nodes, adj, node_feats):
    """convert sampled node list + adjacency into (x, edge_index) tensors."""
    nm = {v: i for i, v in enumerate(nodes)}
    nm_set = set(nodes)
    x = torch.tensor(node_feats[nodes], dtype=torch.float32)
    edges = [
        [nm[u], nm[v]]
        for u in nodes
        for v in adj[u]
        if v in nm_set
    ]
    if not edges:
        return x, torch.zeros(2, 0, dtype=torch.long)
    ei = torch.tensor(edges, dtype=torch.long).T
    ei = to_undirected(ei)
    ei, _ = remove_self_loops(ei)
    return x, ei


def load_gae(path):
    """load a saved SubgraphGAE checkpoint, return (model, out_dim)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = SubgraphGAE(
        ckpt['in_dim'], ckpt['hidden_dim'], ckpt['out_dim'], ckpt['n_layers']
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['out_dim']
