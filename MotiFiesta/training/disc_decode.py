from collections import Counter
from itertools import permutations

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

import torch
import torch.nn.functional as F
from lshashpy3 import LSHash

from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.training.loading import get_loader

class Decoder:
    def __init__(self, model_id, dataset_id):
        self.model_id = model_id
        self.dataset_id = dataset_id

        self.model = load_model(model_id)['model']
        print(self.model)
        self.dataset = get_loader(dataset_id)
        pass

    def decode(self):
        """ Assigns a motif ID vector to each node in the graph.
        Returns a Dataset object with an additional feature `motif_pred`
        """
        raise NotImplementedError
    def eval(self):
        """ Computes jaccard score for a model."""
        raise NotImplementedError

class DiscHashDecoder(Decoder):
    def __init__(self, model_id, dataset_id, hash_dim=4, dummy=False, level=2):
        self.level = level
        self.dummy = dummy
        self.hash_dim = hash_dim

        super().__init__(model_id=model_id,
                         dataset_id=dataset_id,
                         )

    @staticmethod
    def total_sigma(level, node, tree, sigmas, ee):
        """ Recursively compute total sigma score for a subgraph. """
        if level == 0:
            return 0

        children = list(tree[level][node])

        if len(children) < 2:
            return DiscHashDecoder.total_sigma(level - 1, children[0], tree, sigmas, ee)

        c0, c1 = children[0], children[1]
        
        # use the dictionary lookup provided in the 'ee' argument
        edge_lookups = ee
        e_idx = edge_lookups[level-1].get(tuple(sorted((c0, c1))))

        if e_idx is None:
            return DiscHashDecoder.total_sigma(level-1, c0, tree, sigmas, ee) +\
                   DiscHashDecoder.total_sigma(level-1, c1, tree, sigmas, ee)

        current_score = sigmas[level-1][e_idx]

        return current_score +\
               DiscHashDecoder.total_sigma(level-1, c0, tree, sigmas, ee) +\
               DiscHashDecoder.total_sigma(level-1, c1, tree, sigmas, ee)
    
    def plot_motif_summary(self, decoded_graphs):
        g_pyg = decoded_graphs[0]
        G_full = to_networkx(g_pyg, to_undirected=True)
        unique_motifs = torch.unique(g_pyg.motif_pred)
        
        # collect data first so we can sort it
        summary_data = []
        for m_id in unique_motifs:
            if m_id == 0: continue 
            mask = (g_pyg.motif_pred == m_id)
            all_node_indices = mask.nonzero(as_tuple=True)[0].tolist()
            avg_score = g_pyg.cum_scores[mask].mean().item()

            subgraph_all = G_full.subgraph(all_node_indices)
            num_instances = nx.number_connected_components(subgraph_all)
            avg_nodes = len(all_node_indices) / num_instances if num_instances > 0 else 0
            
            summary_data.append({
                'id': m_id.item(),
                'total': len(all_node_indices),
                'avg_nodes': avg_nodes,
                'score': avg_score,
                'nodes': all_node_indices
            })

        # sort by score (Significance/Kleos) in descending order
        summary_data.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n{'ID':<5} | {'Total Nodes':<12} | {'Avg Nodes':<12} | {'Score'}")
        print("-" * 55)

        for data in summary_data:
            print(f"{data['id']:<5} | {data['total']:<12} | {data['avg_nodes']:<12.2f} | {data['score']:.4f}")
            
            # draw the representative example for the sorted motifs
            rep_node = data['nodes'][0]
            example_nodes = [n for n in G_full.neighbors(rep_node) if g_pyg.motif_pred[n] == data['id']]
            example_nodes.append(rep_node)
            self._draw_single_example(G_full, example_nodes, data['id'])

    def _draw_single_example(self, G_full, nodes, motif_id):
        subgraph = G_full.subgraph(nodes)
        plt.figure(figsize=(4, 4))
        
        pos = nx.spring_layout(subgraph, seed=42) 
        
        nx.draw(subgraph, pos, 
                with_labels=True, 
                node_color='#A0CBE2', 
                edge_color='#BBBBBB',
                node_size=600,
                font_size=10)
        
        plt.title(f"Representative Plot: Motif {motif_id}\n({len(nodes)} nodes)")
        plt.show()
    
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
            g = g_pair['pos'] if isinstance(g_pair, dict) else g_pair
            g_hashes = [''] * len(g.x)

            batch = torch.zeros(len(g.x), dtype=torch.long)
            motif_scores = torch.zeros_like(batch, dtype=torch.float32)
            spotlight_ids = torch.zeros_like(batch, dtype=torch.long)

            with torch.no_grad():
                embs,probas,ee,_,merge_info,_  = self.model(g.x,
                                                          g.edge_index,
                                                          batch,
                                                          dummy=self.dummy)

            if self.level > len(embs)-1:
                skipped_idx.add(idx)
                all_scores.append(None)
                all_hashes.append(None)
                all_spotlights.append(None)
                continue

            # convert edge lists to lookup dictionaries before recursion
            ee_lookups = []
            for layer_ee in ee:
                lookup = {tuple(sorted((e[0], e[1]))): i for i, e in enumerate(layer_ee.t().tolist())}
                ee_lookups.append(lookup)
            g = to_networkx(g)
            for i,x in enumerate(embs[self.level]):
                h = hash_table.index(x.detach().numpy())[0]
                # def total_sigma(self, level, node, tree, sigmas, ee):
                spotlight = list(merge_info['spotlights'][self.level][i])
                score = self.total_sigma(self.level, i, merge_info['tree'], probas, ee_lookups)
                hash_set.add(h)
                for node in spotlight:
                    motif_scores[node] = score
                    g_hashes[node] = h
                    spotlight_ids[node] = spot_count
                spot_count += 1
            all_scores.append(motif_scores)
            all_hashes.append(g_hashes)
            all_spotlights.append(spotlight_ids)

        hash_idx = {h:i+1 for i, h in enumerate(sorted(hash_set))}

        decoded_graphs = []
        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if idx > n_graphs and n_graphs > -1:
                break
            if idx in skipped_idx:
                continue
            motif_inds = torch.tensor([hash_idx[h] for h in all_hashes[idx]])
            g = g_pair['pos'] if isinstance(g_pair, dict) else g_pair
            g.motif_pred = motif_inds
            g.cum_scores = all_scores[idx]
            g.spotlight_ids = all_spotlights[idx]
            decoded_graphs.append(g)

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

    def motif_sigma(self, decoded_graphs):
        _, true_motif_ids, sigma_all = DiscHashDecoder.collect_output(decoded_graphs)
        sig_mot = torch.tensor([0., 0.]).scatter_add(0, true_motif_ids, sigma_all)
        vals, counts = torch.unique(true_motif_ids, return_counts=True)
        return sig_mot / counts

    def eval(self, decoded_graphs, n_motifs=1, top_k=1):
        """ Keep top k motifs and match them to the true motif annotation usin
        permutations and jaccard.

        >>> from torch_geometric.data import Data
        >>> scores = torch.tensor([.5, .9, .9, 1])
        >>> true = torch.tensor([0, 0, 0, 1])
        >>> pred = torch.tensor([1, 1, 1, 3])
        >>> g = Data(cum_scores=scores, motif_pred=pred, motif_id=true)
        >>> eval([g])
        1.0
        """

        # collect all the graphs into one big tensor
        motifs_pred_all, true_motif_ids, sigma_all = DiscHashDecoder.collect_output(decoded_graphs)

        # compute average sigma by motif ID
        motif_ids,counts = torch.unique(motifs_pred_all, return_counts=True)
        sigma_avg = torch.zeros_like(motif_ids, dtype=torch.float32).scatter_add(0, motifs_pred_all, sigma_all) / counts

        # rank motifs
        motifs_sorted = torch.argsort(sigma_avg, descending=True)
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

if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    # from torch_geometric.data import Data
    # scores = torch.tensor([.5, .9, .9, 1])
    # true = torch.tensor([0, 0, 0, 1])
    # pred = torch.tensor([1, 1, 1, 3])
    # g = Data(cum_scores=scores, motif_pred=pred, motif_id=true)
    # _eval([g], top_k=2)

    decoder = DiscHashDecoder('barbell-borg-15',
                              'synth-distort-barbell-d0.00',
                              dummy=False,
                              level=2)
    graphs = decoder.decode(n_graphs=5)
    score = decoder.eval(graphs, top_k=5)
    mot_sigma = decoder.motif_sigma(graphs)
    print(mot_sigma)
    print(score)
    pass