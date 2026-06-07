from collections import Counter
from itertools import permutations

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from lshashpy3 import LSHash


from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.training.loading import get_loader

def _otsu_threshold(values):
    import numpy as np
    vals = values.cpu().detach().numpy()
    n = len(vals)
    if n == 0:
        return 0.0
    bins = np.linspace(vals.min(), vals.max(), 256)
    best_thresh, best_var = float(bins[0]), -1.0
    for t in bins:
        fg = vals[vals >= t]
        bg = vals[vals < t]
        if len(fg) == 0 or len(bg) == 0:
            continue
        w_fg = len(fg) / n
        w_bg = len(bg) / n
        var = w_fg * w_bg * (fg.mean() - bg.mean()) ** 2
        if var > best_var:
            best_var, best_thresh = var, float(t)
    return best_thresh


class Decoder:
    def __init__(self, model_id, dataset_id, dataset_name=None):
        self.model_id = model_id
        self.dataset_id = dataset_id

        from MotiFiesta.utils.learning_utils import get_device
        self.device = get_device()
        self.model = load_model(model_id)['model'].to(self.device)
        print(self.model)
        self.dataset = get_loader(root=dataset_id, name=dataset_name or dataset_id)
        pass

    def decode(self):
        """ Assigns a motif ID vector to each node in the graph.
        Returns a Dataset object with an additional feature `motif_pred`
        """
        raise NotImplementedError
    def eval(self):
        """ Computes jaccard score for a model."""
        raise NotImplementedError

class HashDecoder(Decoder):
    def __init__(self, model_id, dataset_id, hash_dim=4, dummy=False, level=2,
                 dataset_name=None, eval_dataset_id=None, eval_kwargs=None):
        self.level = level
        self.dummy = dummy
        self.hash_dim = hash_dim
        self.eval_dataset_id = eval_dataset_id
        self.eval_kwargs = eval_kwargs or {}

        super().__init__(model_id=model_id,
                         dataset_id=dataset_id,
                         dataset_name=dataset_name,
                         )

    @staticmethod
    def total_sigma(level, node, tree, sigmas, ee):
        """ Recursively compute total sigma score
        for a subgraph.
        """
        children = list(tree[level][node])
        n_children = len(children)

        if n_children == 0:
            return 0
        elif n_children == 1:
            return HashDecoder.total_sigma(level-1, children[0], tree, sigmas, ee)
        else:
            eind = (ee[level-1][0] == children[0]) &\
                   (ee[level-1][1] == children[1])
            eind = eind.nonzero()[0][0].item()
            score = sigmas[level-1][eind]
            # get score for this node
            return score +\
                   HashDecoder.total_sigma(level-1, children[0], tree, sigmas, ee) +\
                   HashDecoder.total_sigma(level-1, children[1], tree, sigmas, ee)


    def decode(self, n_graphs=-1):
        # one hash table for each coarsening level
        hash_table = LSHash(self.hash_dim, self.model.hidden_dim)

        hash_to_int = Counter()
        hash_set = set()

        self.model.eval()

        nb = 0
        all_hashes = []
        all_scores = []
        all_spotlights = []

        skipped_idx = set()

        spot_count = 0
        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if idx > n_graphs and n_graphs > -1:
                break
            g = g_pair['pos']
            g_hashes = [''] * len(g.x)

            batch = torch.zeros(len(g.x), dtype=torch.long, device=self.device)
            motif_scores = torch.zeros(len(g.x), dtype=torch.float32, device=self.device)
            spotlight_ids = torch.zeros(len(g.x), dtype=torch.long, device=self.device)

            with torch.no_grad():
                embs,probas,ee,_,merge_info,internals  = self.model(
                    g.x.float().to(self.device),
                    g.edge_index.to(self.device),
                    batch,
                    dummy=self.dummy)

            if self.level > len(embs)-1:
                skipped_idx.add(idx)
                all_scores.append(None)
                all_hashes.append(None)
                all_spotlights.append(None)
                continue
            g = to_networkx(g_pair['pos'])

            use_scatter_path = (
                merge_info.get('spotlights') is None
                and merge_info.get('merge_history') is not None
            )

            if use_scatter_path:
                merge_history = merge_info['merge_history'][self.level]
                Z = embs[self.level]
                n_sup = Z.size(0)
                cum_all = merge_info['merge_history']

                # total_sigma equivalent: bottom-up accumulation across levels 1..L.
                # at each level, inherited score from children + contraction edge score.
                # uses unique (cum[t-1], cum[t]) pairs so each level-(t-1) supernode
                # contributes exactly once — matching original total_sigma tree recursion.
                scores_t = probas[1].cpu() if len(probas) > 1 else torch.zeros(
                    int(cum_all[1].max().item()) + 1 if len(cum_all) > 1 else 1)
                for t in range(2, self.level + 1):
                    if t >= len(probas) or t >= len(cum_all):
                        break
                    pairs = torch.stack([cum_all[t - 1].cpu(), cum_all[t].cpu()], dim=1)
                    unique_pairs = torch.unique(pairs, dim=0)
                    src = unique_pairs[:, 0]
                    dst = unique_pairs[:, 1]
                    n_sup_t = int(cum_all[t].max().item()) + 1
                    inherited = torch.zeros(n_sup_t).scatter_add(0, dst, scores_t[src])
                    scores_t = inherited + probas[t].cpu()
                sup_scores = scores_t.to(Z.device)

                for i, x in enumerate(embs[self.level]):
                    h = hash_table.index(x.detach().cpu().numpy())[0]
                    spotlight = (merge_history == i).nonzero(as_tuple=False).flatten().tolist()
                    score = sup_scores[i].item()
                    hash_set.add(h)
                    for node in spotlight:
                        motif_scores[node] = score
                        g_hashes[node] = h
                        spotlight_ids[node] = spot_count
                    spot_count += 1
            else:
                for i, x in enumerate(embs[self.level]):
                    h = hash_table.index(x.detach().cpu().numpy())[0]
                    spotlight = list(merge_info['spotlights'][self.level][i])
                    score = self.total_sigma(self.level, i, merge_info['tree'], probas, ee)
                    hash_set.add(h)
                    for node in spotlight:
                        motif_scores[node] = score
                        g_hashes[node] = h
                        spotlight_ids[node] = spot_count
                    spot_count += 1
            all_scores.append(motif_scores.cpu())
            all_hashes.append(g_hashes)
            all_spotlights.append(spotlight_ids.cpu())

        hash_idx = {h:i+1 for i, h in enumerate(sorted(hash_set))}

        decoded_graphs = []
        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if idx > n_graphs and n_graphs > -1:
                break
            if idx in skipped_idx:
                continue
            motif_inds = torch.tensor([hash_idx[h] for h in all_hashes[idx]])
            g_pair['pos'].motif_pred = motif_inds
            g_pair['pos'].cum_scores = all_scores[idx]
            g_pair['pos'].spotlight_ids = all_spotlights[idx]
            decoded_graphs.append(g_pair['pos'])

        return decoded_graphs

    def decode_all_levels(self, n_graphs=-1):
        """run model forward once per graph, return {level: decoded_graphs} for all levels.

        avoids redundant forward passes when evaluating across multiple levels.
        """
        depth = self.model.depth
        tables = {lvl: LSHash(self.hash_dim, self.model.hidden_dim) for lvl in range(1, depth + 1)}
        hash_sets = {lvl: set() for lvl in range(1, depth + 1)}
        all_hashes_by_level = {lvl: [] for lvl in range(1, depth + 1)}
        all_scores_by_level = {lvl: [] for lvl in range(1, depth + 1)}
        skipped = set()

        self.model.eval()
        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if n_graphs > -1 and idx >= n_graphs:
                break
            g = g_pair['pos']
            n_nodes = len(g.x)
            batch = torch.zeros(n_nodes, dtype=torch.long, device=self.device)
            with torch.no_grad():
                embs, probas, _, _, merge_info, _ = self.model(
                    g.x.float().to(self.device), g.edge_index.to(self.device), batch
                )
            if merge_info.get('merge_history') is None:
                skipped.add(idx)
                for lvl in range(1, depth + 1):
                    all_hashes_by_level[lvl].append(None)
                    all_scores_by_level[lvl].append(None)
                continue
            merge_history_all = merge_info['merge_history']
            for lvl in range(1, min(depth + 1, len(embs))):
                Z = embs[lvl]
                mh = merge_history_all[lvl].cpu()
                n_sup = Z.size(0)
                sup_scores = probas[lvl].cpu() if lvl < len(probas) else torch.ones(n_sup)
                hashes = [tables[lvl].index(Z[i].detach().cpu().numpy())[0] for i in range(n_sup)]
                hash_sets[lvl].update(hashes)
                motif_scores = sup_scores[mh]
                node_hashes = [hashes[i] for i in mh.tolist()]
                all_hashes_by_level[lvl].append(node_hashes)
                all_scores_by_level[lvl].append(motif_scores.cpu())

        result = {}
        for lvl in range(1, depth + 1):
            hash_idx = {h: i + 1 for i, h in enumerate(sorted(hash_sets[lvl]))}
            decoded = []
            for idx, g_pair in enumerate(self.dataset['dataset_whole']):
                if n_graphs > -1 and idx >= n_graphs:
                    break
                if idx in skipped or all_hashes_by_level[lvl][idx] is None:
                    continue
                import copy
                g = copy.copy(g_pair['pos'])  # shallow copy — don't mutate shared object
                g.motif_pred = torch.tensor(
                    [hash_idx[h] for h in all_hashes_by_level[lvl][idx]]
                )
                g.cum_scores = all_scores_by_level[lvl][idx]
                decoded.append(g)
            result[lvl] = decoded
        return result

    def decode_multilevel(self, n_graphs=-1):
        """decode using all pooling levels simultaneously.

        hashes supernode embeddings from every level into one shared table.
        each node's sigma score accumulates across all levels it appears in.
        hash assignment comes from whichever level gave the node its highest score.
        works only with the scatter path (DiscModel).
        """
        hash_table = LSHash(self.hash_dim, self.model.hidden_dim)
        hash_set = set()
        self.model.eval()

        all_hashes = []
        all_scores = []
        all_spotlights = []
        skipped_idx = set()
        spot_count_base = 0

        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if idx > n_graphs and n_graphs > -1:
                break
            g = g_pair['pos']
            n_nodes = len(g.x)
            batch = torch.zeros(n_nodes, dtype=torch.long, device=self.device)

            with torch.no_grad():
                embs, probas, ee, _, merge_info, internals = self.model(
                    g.x.float().to(self.device),
                    g.edge_index.to(self.device),
                    batch, dummy=self.dummy
                )

            use_scatter = (
                merge_info.get('spotlights') is None
                and merge_info.get('merge_history') is not None
            )
            if not use_scatter or len(embs) < 2:
                skipped_idx.add(idx)
                all_scores.append(None)
                all_hashes.append(None)
                all_spotlights.append(None)
                continue

            merge_history_all = merge_info['merge_history']
            motif_scores = torch.zeros(n_nodes, dtype=torch.float32, device=self.device)
            best_score = torch.zeros(n_nodes, dtype=torch.float32, device=self.device)
            g_hashes = [''] * n_nodes
            spotlight_ids = torch.zeros(n_nodes, dtype=torch.long, device=self.device)

            # precompute total_sigma at every level using bottom-up accumulation
            total_sigma_by_level = {}
            scores_t = probas[1].cpu() if len(probas) > 1 else torch.zeros(
                int(merge_history_all[1].max().item()) + 1 if len(merge_history_all) > 1 else 1)
            total_sigma_by_level[1] = scores_t
            for t in range(2, len(embs)):
                if t >= len(probas) or t >= len(merge_history_all):
                    break
                pairs = torch.stack([merge_history_all[t - 1].cpu(), merge_history_all[t].cpu()], dim=1)
                unique_pairs = torch.unique(pairs, dim=0)
                src = unique_pairs[:, 0]
                dst = unique_pairs[:, 1]
                n_sup_t = int(merge_history_all[t].max().item()) + 1
                inherited = torch.zeros(n_sup_t).scatter_add(0, dst, scores_t[src])
                scores_t = inherited + probas[t].cpu()
                total_sigma_by_level[t] = scores_t

            for t in range(1, len(embs)):
                Z = embs[t]
                mh = merge_history_all[t]
                n_sup = Z.size(0)
                sup_scores = total_sigma_by_level.get(t, probas[t] if t < len(probas)
                                                      else torch.zeros(n_sup)).to(self.device)

                # hash all supernodes at this level
                sup_hashes = [hash_table.index(Z[i].detach().cpu().numpy())[0] for i in range(n_sup)]
                hash_set.update(sup_hashes)

                # vectorized score accumulation
                node_scores_t = sup_scores[mh]
                motif_scores += node_scores_t

                # vectorized update of tensor fields where this level beats best so far
                update_mask = node_scores_t > best_score
                best_score = torch.where(update_mask, node_scores_t, best_score)
                spotlight_ids = torch.where(
                    update_mask,
                    spot_count_base + mh,
                    spotlight_ids,
                )

                # hash list update — only iterate nodes that improved
                cum_list = mh.tolist()
                for n in update_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                    g_hashes[n] = sup_hashes[cum_list[n]]

                spot_count_base += n_sup

            all_scores.append(motif_scores.cpu())
            all_hashes.append(g_hashes)
            all_spotlights.append(spotlight_ids.cpu())

        hash_idx = {h: i + 1 for i, h in enumerate(sorted(hash_set))}

        decoded_graphs = []
        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if idx > n_graphs and n_graphs > -1:
                break
            if idx in skipped_idx:
                continue
            motif_inds = torch.tensor([hash_idx[h] for h in all_hashes[idx]])
            g_pair['pos'].motif_pred = motif_inds
            g_pair['pos'].cum_scores = all_scores[idx]
            g_pair['pos'].spotlight_ids = all_spotlights[idx]
            decoded_graphs.append(g_pair['pos'])

        return decoded_graphs

    @staticmethod
    def collect_output(decoded_graphs):
        sigma_all, motifs_pred_all, true_motif_ids = [], [], []
        for g in decoded_graphs:
            sigma_all.append(g.cum_scores)
            motifs_pred_all.append(g.motif_pred)
            true_motif_ids.append(g.motif_id)

        sigma_all = torch.cat(sigma_all)
        motifs_pred_all = torch.cat(motifs_pred_all)
        true_motif_ids = torch.cat(true_motif_ids)


        # reindex motifs from 0 to 1
        motifs_input = torch.unique(motifs_pred_all)
        motif_indices = {m.item():i for i,m in enumerate(motifs_input)}
        for i in range(len(motifs_pred_all)):
            item = motifs_pred_all[i]
            motifs_pred_all[i] = motif_indices[item.item()]

        return motifs_pred_all, true_motif_ids, sigma_all

    def edge_score_stats(self, n_graphs=10):
        """print mean/std of score_net outputs per level to diagnose degenerate predictors."""
        self.model.eval()
        from collections import defaultdict
        scores_by_level = defaultdict(list)

        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if idx >= n_graphs:
                break
            g = g_pair['pos']
            batch = torch.zeros(len(g.x), dtype=torch.long, device=self.device)
            with torch.no_grad():
                _, _, _, _, _, internals = self.model(
                    g.x.float().to(self.device), g.edge_index.to(self.device),
                    batch, dummy=self.dummy)
            for t, d in enumerate(internals):
                if 'edge_scores' in d and d['edge_scores'] is not None:
                    scores_by_level[t].append(d['edge_scores'].detach())

        print("\nedge score stats (sigmoid score / pre-sigmoid logit per pooling level):")
        for level in sorted(scores_by_level):
            s = torch.cat(scores_by_level[level])
            print(f"  level {level}: mean={s.mean():.4f}  std={s.std():.4f}  "
                  f"min={s.min():.4f}  max={s.max():.4f}  n={len(s)}")

    def decode_subgraph_scale(self, n_graphs=-1):
        return self.decode(n_graphs=n_graphs)

    def motif_sigma(self, decoded_graphs):
        _, true_motif_ids, sigma_all = HashDecoder.collect_output(decoded_graphs)
        sig_mot = torch.tensor([0., 0.]).scatter_add(0, true_motif_ids, sigma_all)
        vals, counts = torch.unique(true_motif_ids, return_counts=True)
        return sig_mot / counts

    def eval(self, decoded_graphs, n_motifs=1, top_k=1, rank_by='count'):
        """ Keep top k motifs and match them to the true motif annotation usin
        permutations and jaccard.

        rank_by: 'count' (default) ranks by cluster size — most frequent hash
                 bucket is selected. 'sigma' ranks by average cum_score.
        """

        # collect all the graphs into one big tensor
        motifs_pred_all, true_motif_ids, sigma_all = HashDecoder.collect_output(decoded_graphs)

        # rank motifs by count or average sigma per bucket.
        # main branch always uses average sigma: scatter_add / counts.
        motif_ids, counts = torch.unique(motifs_pred_all, return_counts=True)
        if rank_by == 'count':
            rank_scores = counts.float()
        else:
            # 'sigma': average sigma per bucket — matches main branch sigma_avg = scatter_add / counts
            rank_scores = torch.zeros_like(motif_ids, dtype=torch.float32).scatter_add(
                0, motifs_pred_all, sigma_all) / counts

        # rank motifs
        motifs_sorted = torch.argsort(rank_scores, descending=True)
        ranks = torch.zeros_like(motifs_sorted)
        for ind, val in enumerate(motifs_sorted):
            ranks[val] = ind

        # print(sigma_avg)
        # print(ranks)
        # kill motifs below top_k (by setting motif id to 0), elimiate last index which is a dummy
        motifs_pred_all = torch.where(ranks[motifs_pred_all] < top_k, motifs_pred_all+1, 0)
        motif_ids = torch.unique(motifs_pred_all)

        # tensor for motif IDs: tensor[i] = 1 if assigned to motif i, all zeros if no motif assigned
        # skip the zero motif (no motif)
        # print(motifs_pred_all)
        pred = F.one_hot(motifs_pred_all)
        non_empty_mask = pred.abs().sum(dim=0).bool()

        pred = pred[:,non_empty_mask]
        true = F.one_hot(true_motif_ids)[:,1:]
        # print(pred)
        # print(true)

        # try all permutations of predicted motif IDs
        best_jaccard = 0
        for p in permutations(range(pred.shape[1])):
            # apply permutation
            p = torch.tensor(p)
            pred_perm = pred[:,p]

            # only keep as many motifs as true ones.
            pred_slice = pred_perm[:,:n_motifs]

            num = torch.min(pred_slice, true).sum(dim=0)
            den = torch.max(pred_slice, true).sum(dim=0)

            jaccard = (num / den).sum().item()

            if jaccard > best_jaccard:
                best_jaccard = jaccard
        return best_jaccard

    def eval_sigma_threshold(self, n_graphs=-1, top_frac=None):
        """score-only eval: rank nodes by cum_score, threshold, compute M-Jaccard. no LSH."""
        decoded_graphs = self.decode(n_graphs=n_graphs)
        if not decoded_graphs:
            return 0.0
        sigma_all = torch.cat([g.cum_scores for g in decoded_graphs])
        true_all = torch.cat([g.motif_id for g in decoded_graphs])
        if top_frac is not None:
            k = max(1, int(top_frac * len(sigma_all)))
            threshold = torch.topk(sigma_all, k).values[-1].item()
        else:
            threshold = _otsu_threshold(sigma_all)
        pred = sigma_all >= threshold
        true_motif = true_all > 0
        intersection = (pred & true_motif).sum().float()
        union = (pred | true_motif).sum().float()
        return (intersection / union).item() if union > 0 else 0.0


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    # from torch_geometric.data import Data
    # scores = torch.tensor([.5, .9, .9, 1])
    # true = torch.tensor([0, 0, 0, 1])
    # pred = torch.tensor([1, 1, 1, 3])
    # g = Data(cum_scores=scores, motif_pred=pred, motif_id=true)
    # _eval([g], top_k=2)

    decoder = HashDecoder('barbell-borg-15',
                          'synth-distort-barbell-d0.00',
                          dummy=False,
                          level=2)
    graphs = decoder.decode(n_graphs=5)
    score = decoder.eval(graphs, top_k=5)
    mot_sigma = decoder.motif_sigma(graphs)
    print(mot_sigma)
    print(score)
    pass