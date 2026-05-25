import os
from collections import defaultdict

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from lshashpy3 import LSHash

from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.training.loading import get_loader


class Decoder:
    def __init__(self, model_id, dataset_id, dataset_root=None, **dataset_kwargs):
        self.model_id = model_id
        self.dataset_id = dataset_id

        self.model = load_model(model_id)['model']
        print(self.model)

        root = dataset_root if dataset_root is not None else dataset_id
        self.dataset = get_loader(root=root, name=dataset_id, **dataset_kwargs)
        pass

    def decode(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class DiscHashDecoder(Decoder):
    """
    decoder for single-graph discovery mode.

    runs a trained motifiesta model on each graph in the dataset, pulls out
    supernode spotlights at a chosen contraction level, canonicalises each
    instance via graph6, groups by canonical form, ranks by aggregate score,
    and exports the results in formats compatible with nemo suite and nemomap.

    when ground truth is present on the input graph (is_motif and motif_id node
    attributes, as produced by synthetic.py), also computes evaluation metrics:
        pattern-level precision/recall against embedded g6 labels,
        lsh-bucket m-jaccard against motif type labels (paper methodology),
        instance-level recall against embedded motif instances.
    """

    def __init__(self,
                 model_id,
                 dataset_id,
                 dataset_root=None,
                 hash_dim=8,
                 dummy=False,
                 level=2,
                 batch_size=64,
                 **dataset_kwargs):
        self.level = level
        self.dummy = dummy
        self.hash_dim = hash_dim

        super().__init__(model_id=model_id,
                         dataset_id=dataset_id,
                         dataset_root=dataset_root,
                         batch_size=batch_size,
                         **dataset_kwargs,
                         )

    # ------------------------------------------------------------------
    # core decoding
    # ------------------------------------------------------------------

    @staticmethod
    def total_sigma(level, node, cluster_chain, sigmas, ee):
        """recursively sum edge scores along the contraction path.

        children of `node` at `level` are the level-(level-1) supernodes
        whose entry in cluster_chain[level-1] equals `node`.
        """
        if level == 0:
            return 0

        cluster = cluster_chain[level - 1]
        children = (cluster == node).nonzero(as_tuple=False).squeeze(-1).tolist()

        if len(children) == 0:
            # orphan supernode with no predecessor — contribute nothing
            return 0

        if len(children) < 2:
            return DiscHashDecoder.total_sigma(level - 1, children[0], cluster_chain, sigmas, ee)

        c0, c1 = children[0], children[1]
        e_idx = ee[level-1].get(tuple(sorted((c0, c1))))

        if e_idx is None:
            return DiscHashDecoder.total_sigma(level-1, c0, cluster_chain, sigmas, ee) +\
                   DiscHashDecoder.total_sigma(level-1, c1, cluster_chain, sigmas, ee)

        current_score = sigmas[level-1][e_idx]

        return current_score +\
               DiscHashDecoder.total_sigma(level-1, c0, cluster_chain, sigmas, ee) +\
               DiscHashDecoder.total_sigma(level-1, c1, cluster_chain, sigmas, ee)

    def _induced_subgraph(self, source_nx, spotlight):
        """
        pull the induced subgraph of the source graph over the spotlight nodes.
        returns a relabeled nx graph with consecutive 0..n-1 vertex ids and the
        sorted list of original node ids (mapping new id -> original).
        """
        nodes = sorted(spotlight)
        sub = source_nx.subgraph(nodes).copy()
        remap = {old: new for new, old in enumerate(nodes)}
        sub = nx.relabel_nodes(sub, remap)
        return sub, nodes

    def _graph6(self, g):
        """
        canonical graph6 label for an undirected nx graph.

        networkx's to_graph6_bytes is sensitive to node ordering, so two
        isomorphic graphs with different vertex labels can produce different g6
        strings. we route through igraph's canonical_permutation to get a
        consistent ordering, then emit g6 from the canonicalised form. this
        ensures that structurally identical subgraphs always map to the same
        label.
        """
        from igraph import Graph as IGraph

        g_undirected = g.to_undirected() if g.is_directed() else g

        # convert to igraph to allow for canonical_permutation (backed by bliss)
        ig = IGraph(n=g_undirected.number_of_nodes(),
                    edges=list(g_undirected.edges()),
                    directed=False)
        ig.simplify()

        perm = ig.canonical_permutation()
        ig_canon = ig.permute_vertices(perm)

        # rebuild an nx graph from the canonicalised igraph and emit g6
        n = ig_canon.vcount()
        g_canon = nx.Graph()
        g_canon.add_nodes_from(range(n))
        g_canon.add_edges_from(ig_canon.get_edgelist())

        return nx.to_graph6_bytes(g_canon, header=False).decode().strip()

    def decode(self, n_batches=-1):
        """
        run the trained model on neighborhood batches of the source graph and
        collect motif instances at the configured contraction level.

        uses loader_whole (same neighborloader and batch size as training) so
        each batch sees the same neighborhood-scale context the model was
        trained on. nodes appear in multiple overlapping batches; the
        highest-sigma assignment wins per node. spotlights are rebuilt by
        grouping nodes by their winning hash after all batches are processed.

        when sil_loss achieves full sampling invariance, different batches
        containing the same motif instance will produce identical level-t
        embeddings and hash to the same bucket. the quality of this decode
        is therefore a direct readout of how well sil_loss is working.

        returns a list with one result dict (single source graph):
          - 'source_nx': nx graph of the full input
          - 'spotlights': list of sets of original node ids (one per hash bucket)
          - 'scores': list of mean sigma scores per bucket
          - 'hashes': list of lsh bucket ids
          - 'pyg': the full pyg data object (for ground-truth access)
        """
        hash_table = LSHash(self.hash_dim, self.model.hidden_dim)
        self.model.eval()

        # full graph for source_nx and ground-truth pyg
        g_full = None
        for g_pair in self.dataset['dataset_whole']:
            g_full = g_pair['pos'] if isinstance(g_pair, dict) else g_pair
            break
        source_nx = to_networkx(g_full, to_undirected=True)

        # per-node best assignment: global_node_id → (score, hash)
        node_best_score = {}
        node_best_hash = {}

        for batch_idx, batch in enumerate(self.dataset['loader_whole']):
            if n_batches > -1 and batch_idx >= n_batches:
                break

            pos = batch['pos']

            with torch.no_grad():
                embs, probas, ee, _, merge_info, _ = self.model(
                    pos.x, pos.edge_index, pos.batch,
                    n_id=pos.n_id, dummy=self.dummy)

            if self.level > len(embs) - 1:
                continue

            ee_lookups = []
            for layer_ee in ee:
                lookup = {tuple(sorted((e[0], e[1]))): i
                          for i, e in enumerate(layer_ee.t().tolist())}
                ee_lookups.append(lookup)

            spot_assign = merge_info['spotlight_assignment']
            cluster_chain = merge_info['cluster_chain']
            n_id = merge_info['n_id']  # global node ids for this batch

            spot_at_level = spot_assign[self.level]

            level_emb = embs[self.level]
            level_emb_centered = level_emb - level_emb.mean(dim=0, keepdim=True)

            for i in range(level_emb.size(0)):
                h = hash_table.index(level_emb_centered[i].detach().numpy())[0]

                mask = spot_at_level == i
                local_members = mask.nonzero(as_tuple=False).squeeze(-1)
                spot = set(n_id[local_members].tolist())

                score = self.total_sigma(self.level, i, cluster_chain,
                                         probas, ee_lookups)
                score_val = float(score) if torch.is_tensor(score) else score

                # keep highest-sigma assignment per node across overlapping batches
                for node in spot:
                    if node not in node_best_score or score_val > node_best_score[node]:
                        node_best_score[node] = score_val
                        node_best_hash[node] = h

        # rebuild spotlights grouped by winning hash
        hash_to_nodes = defaultdict(set)
        hash_to_scores = defaultdict(list)
        for node, h in node_best_hash.items():
            hash_to_nodes[h].add(node)
            hash_to_scores[h].append(node_best_score[node])

        spotlights = list(hash_to_nodes.values())
        hashes = list(hash_to_nodes.keys())
        scores = [sum(s) / len(s) for s in hash_to_scores.values()]

        return [{
            'source_nx': source_nx,
            'spotlights': spotlights,
            'scores': scores,
            'hashes': hashes,
            'pyg': g_full,
        }]

    # ------------------------------------------------------------------
    # pattern extraction and ranking (g6-based, used for nemomap export)
    # ------------------------------------------------------------------

    def extract_patterns(self,
                        results,
                        min_size=3,
                        max_size=8,
                        require_connected=True,
                        min_instances=2):
        """
        group instances across the decoded results by canonical pattern (g6).
        returns dict keyed by g6 label, each value:
          { 'label', 'nodes', 'edges', 'instances': [...],
            'total_score', 'mean_score', 'count' }

        used to drive nemomap export and to compute g6 pattern set-overlap as
        a sanity check. not used for the primary m-jaccard metric (which is
        lsh-bucket-based per the paper methodology).
        """
        patterns = defaultdict(lambda: {'instances': [], 'nodes': 0, 'edges': 0})

        for src_idx, res in enumerate(results):
            source_nx = res['source_nx']
            for spot, score in zip(res['spotlights'], res['scores']):
                if not (min_size <= len(spot) <= max_size):
                    continue

                sub, nodes_sorted = self._induced_subgraph(source_nx, spot)

                if require_connected and not nx.is_connected(sub):
                    continue

                if sub.number_of_edges() == 0:
                    continue

                label = self._graph6(sub)

                patterns[label]['label'] = label
                patterns[label]['nodes'] = sub.number_of_nodes()
                patterns[label]['edges'] = sub.number_of_edges()
                patterns[label]['instances'].append({
                    'source_idx': src_idx,
                    'nodes': nodes_sorted,
                    'score': score,
                })

        kept = {}
        for label, data in patterns.items():
            if len(data['instances']) < min_instances:
                continue
            scores = [inst['score'] for inst in data['instances']]
            data['count'] = len(scores)
            data['total_score'] = sum(scores)
            data['mean_score'] = data['total_score'] / data['count']
            kept[label] = data

        return kept

    def rank_patterns(self, patterns, top_n=10, by='total_score'):
        """ sort patterns by the chosen criterion and return the top n. """
        ranked = sorted(patterns.values(), key=lambda p: p[by], reverse=True)
        return ranked[:top_n]

    def diagnose(self, results):
        """
        print pre-filter statistics about the decoded spotlights: size
        distribution, connectedness, and a sample of raw g6 labels. useful for
        tuning min_size / max_size / require_connected before export.
        """
        from collections import Counter

        all_sizes = []
        connected_count = 0
        label_counter = Counter()

        for res in results:
            source_nx = res['source_nx']
            for spot in res['spotlights']:
                all_sizes.append(len(spot))
                if len(spot) < 2:
                    continue
                sub, _ = self._induced_subgraph(source_nx, spot)
                if sub.number_of_edges() == 0:
                    continue
                if nx.is_connected(sub):
                    connected_count += 1
                label_counter[self._graph6(sub)] += 1

        print(f"total spotlights: {len(all_sizes)}")
        print(f"size distribution: {sorted(Counter(all_sizes).items())}")
        print(f"connected (non-trivial) subgraphs: {connected_count}")
        print(f"distinct g6 labels: {len(label_counter)}")
        print(f"top 10 labels pre-filter: {label_counter.most_common(10)}")
        return {
            'total_spotlights': len(all_sizes),
            'size_distribution': dict(Counter(all_sizes)),
            'connected_count': connected_count,
            'label_counter': label_counter,
        }

    # ------------------------------------------------------------------
    # evaluation against embedded ground truth
    # ------------------------------------------------------------------

    def _has_ground_truth(self, results):
        """ detect whether the input data has embedded motif annotations. """
        if not results:
            return False
        pyg = results[0]['pyg']
        return hasattr(pyg, 'motif_id') and pyg.motif_id is not None

    def _embedded_g6s(self, results):
        """canonical g6 labels of the embedded motif instances.

        groups nodes by instance_id (each unique value is one planted motif
        instance) and emits the g6 of each instance's induced subgraph.
        falls back to motif_id + connected_components for datasets that don't
        have instance_id (older processed files).
        """
        g6s = set()
        for res in results:
            pyg = res['pyg']
            source_nx = res['source_nx']

            if hasattr(pyg, 'instance_id') and pyg.instance_id is not None:
                instance_ids = pyg.instance_id.tolist()
                groups = defaultdict(set)
                for node_idx, iid in enumerate(instance_ids):
                    if iid > 0:
                        groups[iid].add(node_idx)
                for iid, nodes in groups.items():
                    if len(nodes) < 2:
                        continue
                    sub, _ = self._induced_subgraph(source_nx, nodes)
                    if sub.number_of_edges() == 0:
                        continue
                    g6s.add(self._graph6(sub))
            elif hasattr(pyg, 'motif_id'):
                # legacy fallback: connected components on motif_id-tagged nodes
                motif_ids = pyg.motif_id.tolist()
                groups = defaultdict(set)
                for node_idx, mid in enumerate(motif_ids):
                    if mid > 0:
                        groups[mid].add(node_idx)
                for mid, nodes in groups.items():
                    sub_all = source_nx.subgraph(nodes)
                    for component in nx.connected_components(sub_all):
                        if len(component) < 2:
                            continue
                        sub, _ = self._induced_subgraph(source_nx, component)
                        if sub.number_of_edges() == 0:
                            continue
                        g6s.add(self._graph6(sub))
        return g6s

    @staticmethod
    def collect_output(results, min_size=1, max_size=999):
        """flatten per-graph spotlight assignments into concatenated tensors.

        mirrors collect_output in decode.py. assigns each node to an integer
        bucket id based on its spotlight's lsh hash (1-indexed; 0 = unassigned),
        reindexes bucket ids to be contiguous from 0, and returns flat tensors
        ready for ranking and metric computation.

        returns:
            motifs_pred_all : long (N,)  — reindexed bucket id per node
            true_motif_ids  : long (N,)  — binary motif label (K=1, >0 → 1)
            sigma_all       : float (N,) — per-node sigma score
            graph_sizes     : list[int]  — num_nodes per graph
        """
        motifs_pred_all = []
        true_motif_ids = []
        sigma_all = []
        graph_sizes = []
        hash_to_int = {}

        for res in results:
            pyg = res['pyg']
            n_nodes = pyg.x.size(0)
            graph_sizes.append(n_nodes)

            node_pred = torch.zeros(n_nodes, dtype=torch.long)
            node_score = torch.zeros(n_nodes, dtype=torch.float32)

            for spot, score, h in zip(res['spotlights'], res['scores'], res['hashes']):
                if not (min_size <= len(spot) <= max_size):
                    continue
                if h not in hash_to_int:
                    hash_to_int[h] = len(hash_to_int) + 1  # 1-indexed, 0 = no assignment
                h_int = hash_to_int[h]
                for node in spot:
                    if node < n_nodes:
                        node_pred[node] = h_int
                        node_score[node] = score

            motifs_pred_all.append(node_pred)
            sigma_all.append(node_score)

            # K=1: collapse all instance ids to one motif class
            if hasattr(pyg, 'motif_id') and pyg.motif_id is not None:
                true_motif_ids.append((pyg.motif_id > 0).long())
            else:
                true_motif_ids.append(torch.zeros(n_nodes, dtype=torch.long))

        motifs_pred_all = torch.cat(motifs_pred_all)
        true_motif_ids = torch.cat(true_motif_ids)
        sigma_all = torch.cat(sigma_all)

        # reindex bucket ids to contiguous 0..M
        motifs_input = torch.unique(motifs_pred_all)
        motif_indices = {m.item(): i for i, m in enumerate(motifs_input)}
        for i in range(len(motifs_pred_all)):
            motifs_pred_all[i] = motif_indices[motifs_pred_all[i].item()]

        return motifs_pred_all, true_motif_ids, sigma_all, graph_sizes

    def eval(self, results, top_k=1, min_size=1, max_size=999):
        """m-jaccard (paper eq. 1 / alg. 2) and instance recall.

        calls collect_output to get flat tensors, ranks buckets by mean sigma,
        keeps top_k, then computes:
          jaccard        : best node-level jaccard over predicted columns (eq. 2)
          instance_recall: fraction of planted instances with majority of nodes
                           in any top-k bucket (unaffected by bucket FP)

        returns dict with 'jaccard' and 'instance_recall'.
        """
        if not results:
            return {'jaccard': 0.0, 'instance_recall': 0.0}

        motifs_pred_all, true_motif_ids, sigma_all, graph_sizes = \
            DiscHashDecoder.collect_output(results, min_size=min_size, max_size=max_size)

        # rank buckets by mean sigma (algorithm 2, line 7)
        motif_ids, counts = torch.unique(motifs_pred_all, return_counts=True)
        sigma_avg = torch.zeros_like(motif_ids, dtype=torch.float32).scatter_add(
            0, motifs_pred_all, sigma_all) / counts

        motifs_sorted = torch.argsort(sigma_avg, descending=True)
        ranks = torch.zeros_like(motifs_sorted)
        for ind, val in enumerate(motifs_sorted):
            ranks[val] = ind

        # kill buckets below top_k
        motifs_pred_all = torch.where(ranks[motifs_pred_all] < top_k,
                                      motifs_pred_all + 1, torch.zeros_like(motifs_pred_all))

        # instance recall: per-instance majority vote against top-k buckets.
        # motif_id is binary (0/1), so instances are inferred from connected
        # components of the motif subgraph rather than motif_id values.
        total_instances = 0
        found_instances = 0
        node_offset = 0
        for res, n_nodes in zip(results, graph_sizes):
            pyg = res['pyg']
            pred_slice = motifs_pred_all[node_offset:node_offset + n_nodes]
            node_offset += n_nodes
            if not (hasattr(pyg, 'motif_id') and pyg.motif_id is not None):
                continue
            motif_nodes = [n for n, mid in enumerate(pyg.motif_id.tolist()) if mid > 0]
            if not motif_nodes:
                continue
            motif_sub = res['source_nx'].subgraph(motif_nodes)
            for comp in nx.connected_components(motif_sub):
                nodes = list(comp)
                total_instances += 1
                n_found = sum(1 for n in nodes if pred_slice[n].item() > 0)
                if n_found > len(nodes) / 2:
                    found_instances += 1
        instance_recall = found_instances / total_instances if total_instances > 0 else 0.0

        # build pred and true one-hot matrices
        pred = F.one_hot(motifs_pred_all)
        non_empty_mask = pred.abs().sum(dim=0).bool()
        pred = pred[:, non_empty_mask]

        # column 0 is the "not predicted" pool (nodes killed by top_k or outside size
        # filters). it must be excluded from the jaccard loop — it's not a prediction.
        # after non_empty_mask the 0-bucket remains as the first column whenever any
        # node was unassigned; drop it so only actual bucket predictions are scored.
        if pred.shape[1] > 0 and (motifs_pred_all == 0).any():
            pred = pred[:, 1:]

        true = F.one_hot(true_motif_ids)[:, 1:]  # drop background column, K=1

        if true.shape[1] == 0 or pred.shape[1] == 0:
            return {'jaccard': 0.0, 'instance_recall': instance_recall}

        # permutation test (eq. 2): try each predicted column against the single true column
        best_jaccard = 0.0
        for col in range(pred.shape[1]):
            pred_col = pred[:, col:col+1].float()
            num = torch.min(pred_col, true.float()).sum(dim=0)  # eq. 1 numerator
            den = torch.max(pred_col, true.float()).sum(dim=0)  # eq. 1 denominator
            jaccard = (num / den).sum().item()
            if jaccard > best_jaccard:
                best_jaccard = jaccard

        return {'jaccard': best_jaccard, 'instance_recall': instance_recall}

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------

    def export_nemocollection(self, patterns_ranked, out_path):
        """
        nemo-style instance manifest: one line per instance,
            <g6_label>[node_id, node_id, ...]
        """
        with open(out_path, 'w') as f:
            for pat in patterns_ranked:
                label = pat['label']
                for inst in pat['instances']:
                    nodes_str = ', '.join(str(n) for n in inst['nodes'])
                    f.write(f"{label}[{nodes_str}]\n")
        return out_path

    def export_nemomap_inputs(self, patterns_ranked, results, out_dir):
        """
        write the input edge list plus a single combined queries file containing
        every top pattern separated by comment lines. each section is an edge
        list that can be pasted into nemomap's query graph field.
        files are zero-indexed plain text (one 'u v' per line).
        """
        os.makedirs(out_dir, exist_ok=True)

        source_nx = results[0]['source_nx']
        input_path = os.path.join(out_dir, 'input_graph.txt')
        with open(input_path, 'w') as f:
            for u, v in source_nx.edges():
                f.write(f"{u} {v}\n")

        queries_path = os.path.join(out_dir, 'queries.txt')
        with open(queries_path, 'w') as f:
            for rank, pat in enumerate(patterns_ranked, start=1):
                rep = pat['instances'][0]
                sub, _ = self._induced_subgraph(source_nx, set(rep['nodes']))
                f.write(f"# query {rank:02d} {pat['label']} "
                        f"({pat['nodes']}n, {pat['edges']}e, "
                        f"count={pat['count']})\n")
                for u, v in sub.edges():
                    f.write(f"{u} {v}\n")
                f.write("\n")

        return input_path, queries_path

    def print_stats(self, patterns_ranked, eval_metrics=None):
        """
        print a summary table of the top patterns to stdout. when
        eval_metrics is provided, an is_embedded column is appended.
        """
        embedded_g6s = set(eval_metrics.get('embedded_g6s', [])) if eval_metrics else set()

        header = f"{'rank':<5} {'g6_label':<10} {'nodes':<6} {'edges':<6} " \
                 f"{'count':<6} {'total_score':<12} {'mean_score':<10}"
        if eval_metrics:
            header += " embedded"
        print(header)
        print("-" * len(header))

        for rank, pat in enumerate(patterns_ranked, start=1):
            row = f"{rank:<5} {pat['label']:<10} {pat['nodes']:<6} {pat['edges']:<6} " \
                  f"{pat['count']:<6} {pat['total_score']:<12.3f} {pat['mean_score']:<10.3f}"
            if eval_metrics:
                row += "  yes" if pat['label'] in embedded_g6s else "  no"
            print(row)

    def export_all(self,
                   results,
                   out_dir,
                   top_n=10,
                   min_size=3,
                   max_size=8,
                   require_connected=True,
                   min_instances=2,
                   rank_by='total_score'):
        """
        one-shot entry point. produces inside `out_dir`:
            motifiesta_collection.txt  - nemo-style instance manifest
            input_graph.txt            - edge list of the source graph
            queries.txt                - combined query edge lists, one per motif
            eval.txt                   - eval metrics summary (only if eval ran)
        stats are printed to stdout instead of written to disk.
        """
        os.makedirs(out_dir, exist_ok=True)

        patterns = self.extract_patterns(results,
                                         min_size=min_size,
                                         max_size=max_size,
                                         require_connected=require_connected,
                                         min_instances=min_instances)
        ranked = self.rank_patterns(patterns, top_n=top_n, by=rank_by)

        eval_metrics = {}
        if self._has_ground_truth(results):
            embedded_g6s = self._embedded_g6s(results)
            predicted_g6s = {p['label'] for p in ranked}
            hit_labels = predicted_g6s & embedded_g6s if predicted_g6s and embedded_g6s else set()
            ev = self.eval(results, top_k=len(ranked), min_size=min_size, max_size=max_size)
            eval_metrics = {
                'embedded_g6s': sorted(embedded_g6s),
                'hit_labels': sorted(hit_labels),
                'jaccard': ev['jaccard'],
                'instance_recall': ev['instance_recall'],
            }

        collection_path = self.export_nemocollection(
            ranked, os.path.join(out_dir, 'motifiesta_collection.txt'))
        input_path, queries_path = self.export_nemomap_inputs(
            ranked, results, out_dir)

        self.print_stats(ranked, eval_metrics=eval_metrics)

        if eval_metrics:
            predicted_g6s_sorted = sorted({p['label'] for p in ranked})
            print(f"\nm-jaccard:        {eval_metrics['jaccard']:.4f}")
            print(f"instance recall:  {eval_metrics['instance_recall']:.4f}")
            print(f"embedded g6s:  {', '.join(eval_metrics['embedded_g6s'])}")
            print(f"predicted g6s: {', '.join(predicted_g6s_sorted)}")
            print(f"hit labels:    {', '.join(eval_metrics['hit_labels']) or 'none'}")

        return {
            'patterns': ranked,
            'eval': eval_metrics,
            'collection': collection_path,
            'input_graph': input_path,
            'queries': queries_path,
        }


if __name__ == "__main__":
    decoder = DiscHashDecoder(
        model_id='test',
        dataset_id='mips_torch',
        dataset_root='data/mips_torch',
        level=2,
    )
    results = decoder.decode()
    out = decoder.export_all(results, out_dir='decoded/test', top_n=10)

    print(f"found {len(out['patterns'])} motif patterns")
    print(f"collection: {out['collection']}")
    print(f"input graph: {out['input_graph']}")
    print(f"queries: {len(out['queries'])}")
    print(f"stats: {out['stats']}")

    print("\ntop motifs:")
    for rank, pat in enumerate(out['patterns'], start=1):
        embedded_flag = ""
        if out['eval']:
            embedded_flag = " [embedded]" if pat['label'] in out['eval']['embedded_g6s'] else ""
        print(f"  {rank}. {pat['label']} ({pat['nodes']}n, {pat['edges']}e) "
              f"count={pat['count']} total_score={pat['total_score']:.3f}"
              f"{embedded_flag}")
