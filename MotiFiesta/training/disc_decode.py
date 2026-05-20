import os
from collections import defaultdict
from itertools import permutations

import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from lshashpy3 import LSHash

from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.training.loading import get_loader


class Decoder:
    def __init__(self, model_id, dataset_id, dataset_root=None):
        self.model_id = model_id
        self.dataset_id = dataset_id

        self.model = load_model(model_id)['model']
        print(self.model)

        root = dataset_root if dataset_root is not None else dataset_id
        self.dataset = get_loader(root=root, name=dataset_id, max_degree=18)
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
                 hash_dim=4,
                 dummy=False,
                 level=2):
        self.level = level
        self.dummy = dummy
        self.hash_dim = hash_dim

        super().__init__(model_id=model_id,
                         dataset_id=dataset_id,
                         dataset_root=dataset_root,
                         )

    # ------------------------------------------------------------------
    # core decoding
    # ------------------------------------------------------------------

    @staticmethod
    def total_sigma(level, node, tree, sigmas, ee):
        """recursively sum edge scores along the contraction path.

        children of `node` at `level` are the level-(level-1) supernodes
        in tree[level][node].
        """
        if level == 0:
            return 0

        children = list(tree[level][node])

        if len(children) == 0:
            # orphan supernode with no predecessor — contribute nothing
            return 0

        if len(children) < 2:
            return DiscHashDecoder.total_sigma(level - 1, children[0], tree, sigmas, ee)

        c0, c1 = children[0], children[1]
        edge_lookups = ee
        e_idx = edge_lookups[level-1].get(tuple(sorted((c0, c1))))

        if e_idx is None:
            return DiscHashDecoder.total_sigma(level-1, c0, tree, sigmas, ee) +\
                   DiscHashDecoder.total_sigma(level-1, c1, tree, sigmas, ee)

        current_score = sigmas[level-1][e_idx]

        return current_score +\
               DiscHashDecoder.total_sigma(level-1, c0, tree, sigmas, ee) +\
               DiscHashDecoder.total_sigma(level-1, c1, tree, sigmas, ee)

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

    def decode(self, n_graphs=-1):
        """
        run the trained model on every graph in the dataset and collect motif
        instances at the configured contraction level.

        returns a list of per-graph decoded results. each result is a dict:
          - 'source_nx': nx graph of the full input
          - 'spotlights': list of sets of original node ids
          - 'scores': list of cumulative sigma scores
          - 'hashes': list of lsh bucket ids
          - 'pyg': the original pyg data object (for ground-truth access)
        """
        hash_table = LSHash(self.hash_dim, self.model.hidden_dim)
        self.model.eval()

        results = []

        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if n_graphs > -1 and idx >= n_graphs:
                break

            g = g_pair['pos'] if isinstance(g_pair, dict) else g_pair

            batch = torch.zeros(len(g.x), dtype=torch.long)

            with torch.no_grad():
                embs, probas, ee, _, merge_info, _ = self.model(
                    g.x, g.edge_index, batch, dummy=self.dummy)

            if self.level > len(embs) - 1:
                continue

            # cache edge->index lookups once per graph
            ee_lookups = []
            for layer_ee in ee:
                lookup = {tuple(sorted((e[0], e[1]))): i
                          for i, e in enumerate(layer_ee.t().tolist())}
                ee_lookups.append(lookup)

            source_nx = to_networkx(g, to_undirected=True)

            # pull the dict-based merge_info fields
            spotlights_all = merge_info['spotlights']  # dict[level][supernode_id] = set of global node ids
            tree = merge_info['tree']                  # dict[level][supernode_id] = set of child supernode ids

            spotlights = []
            scores = []
            hashes = []

            spot_at_level = spotlights_all[self.level]

            # center embeddings before hashing. random-hyperplane lsh
            # measures angular similarity from the origin, so if all
            # embeddings sit in one corner of the space (e.g. mean=-3.5)
            # they collapse to the same bucket regardless of hash_dim or
            # internal spread. zero-mean the level-T embeddings so the
            # hyperplanes cut through the data, not past it.
            level_emb = embs[self.level]
            level_emb_centered = level_emb - level_emb.mean(dim=0, keepdim=True)

            for i, x in enumerate(level_emb):
                h = hash_table.index(level_emb_centered[i].detach().numpy())[0]

                # spotlight is already a set of global node ids
                spot = set(spot_at_level.get(i, set()))

                score = self.total_sigma(self.level, i, tree,
                                         probas, ee_lookups)

                spotlights.append(spot)
                scores.append(float(score) if torch.is_tensor(score) else score)
                hashes.append(h)

            results.append({
                'source_nx': source_nx,
                'spotlights': spotlights,
                'scores': scores,
                'hashes': hashes,
                'pyg': g,
            })

        return results

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
    def _jaccard(a, b):
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def evaluate(self, results, patterns_ranked):
        """
        type-level evaluation against embedded ground truth.

        primary metric is lsh-bucket m-jaccard, matching the paper's decoding
        methodology (algorithm 2). also reports g6 pattern set-overlap as a
        sanity check on the nemomap export side: did the top-ranked g6
        patterns include the canonical topology of the embedded motifs.

        returns an empty dict if no ground truth is present.
        """
        if not self._has_ground_truth(results):
            return {}

        embedded_g6s = self._embedded_g6s(results)
        predicted_g6s = {p['label'] for p in patterns_ranked}

        hit_labels = predicted_g6s & embedded_g6s if predicted_g6s and embedded_g6s else set()

        lsh_jaccard = self._lsh_permutation_jaccard(
            results, top_n=len(patterns_ranked))

        return {
            'embedded_g6s': sorted(embedded_g6s),
            'hit_labels': sorted(hit_labels),
            'jaccard': lsh_jaccard,
        }

    def _lsh_permutation_jaccard(self, results, top_n=10,
                                  min_size=3, max_size=8):
        """
        lsh-bucket m-jaccard, matching the paper's decoding methodology
        (algorithm 2). spotlights are grouped by their lsh hash bucket,
        buckets are ranked by mean sigma score (paper algorithm 2 line 7),
        and the top_n buckets are treated as the predicted motif types for
        the permutation alignment against motif_id ground truth.

        size filter (min_size, max_size) matches extract_patterns defaults
        so the lsh pool and the g6 pool see the same spotlights.
        """
        bucket_sum_scores = defaultdict(float)
        bucket_counts = defaultdict(int)
        bucket_nodes_per_src = defaultdict(lambda: defaultdict(set))

        for src_idx, res in enumerate(results):
            for spot, score, h in zip(res['spotlights'],
                                      res['scores'],
                                      res['hashes']):
                if not (min_size <= len(spot) <= max_size):
                    continue
                bucket_sum_scores[h] += score
                bucket_counts[h] += 1
                bucket_nodes_per_src[h][src_idx].update(spot)

        if not bucket_sum_scores:
            return 0.0

        # rank by mean score per bucket (paper algorithm 2, line 7)
        bucket_mean_scores = {b: bucket_sum_scores[b] / bucket_counts[b]
                              for b in bucket_sum_scores}
        top_buckets = sorted(bucket_mean_scores.keys(),
                             key=lambda b: bucket_mean_scores[b],
                             reverse=True)[:top_n]

        best_total = 0.0
        graphs_with_truth = 0

        for src_idx, res in enumerate(results):
            pyg = res['pyg']
            if not hasattr(pyg, 'motif_id') or pyg.motif_id is None:
                continue

            motif_ids = pyg.motif_id.tolist()
            n_nodes = len(motif_ids)

            # bucket rank (index in top_buckets) is the cluster id
            pred_assign = [-1] * n_nodes
            for b_idx, bucket in enumerate(top_buckets):
                for node in bucket_nodes_per_src[bucket].get(src_idx, set()):
                    if pred_assign[node] == -1:
                        pred_assign[node] = b_idx

            true_labels = sorted({m for m in motif_ids if m > 0})
            pred_labels = sorted({p for p in pred_assign if p >= 0})
            if not true_labels or not pred_labels:
                continue

            graphs_with_truth += 1

            best_jacc = 0.0
            K = len(true_labels)
            k = min(len(pred_labels), K)
            for perm in permutations(pred_labels, k):
                mapping = dict(zip(perm, true_labels[:k]))
                total = 0.0
                for p_lab, t_lab in mapping.items():
                    pred_nodes = {i for i, p in enumerate(pred_assign)
                                  if p == p_lab}
                    true_nodes = {i for i, m in enumerate(motif_ids)
                                  if m == t_lab}
                    total += self._jaccard(pred_nodes, true_nodes)
                avg = total / K  # paper eq. 1: average over K true motif types
                if avg > best_jacc:
                    best_jacc = avg
            best_total += best_jacc

        return best_total / graphs_with_truth if graphs_with_truth else 0.0

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

        eval_metrics = self.evaluate(results, ranked)

        collection_path = self.export_nemocollection(
            ranked, os.path.join(out_dir, 'motifiesta_collection.txt'))
        input_path, queries_path = self.export_nemomap_inputs(
            ranked, results, out_dir)

        self.print_stats(ranked, eval_metrics=eval_metrics)

        if eval_metrics:
            predicted_g6s = sorted({p['label'] for p in ranked})
            print(f"\nm-jaccard:     {eval_metrics['jaccard']:.4f}")
            print(f"embedded g6s:  {', '.join(eval_metrics['embedded_g6s'])}")
            print(f"predicted g6s: {', '.join(predicted_g6s)}")
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
