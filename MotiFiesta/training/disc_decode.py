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
        self.dataset = get_loader(root=root, name=dataset_id)
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
        pattern-level precision/recall against planted g6 labels,
        node-level jaccard against planted motif nodes,
        instance-level recall against planted motif instances.
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
        """ recursively sum edge scores along the contraction path. """
        if level == 0:
            return 0

        children = list(tree[level][node])

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

            spotlights = []
            scores = []
            hashes = []

            for i, x in enumerate(embs[self.level]):
                h = hash_table.index(x.detach().numpy())[0]
                spot = set(merge_info['spotlights'][self.level][i])
                score = self.total_sigma(self.level, i, merge_info['tree'],
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
    # pattern extraction and ranking
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
    # evaluation against planted ground truth
    # ------------------------------------------------------------------

    def _has_ground_truth(self, results):
        """ detect whether the input data has planted motif annotations. """
        if not results:
            return False
        pyg = results[0]['pyg']
        return hasattr(pyg, 'motif_id') and pyg.motif_id is not None

    def _planted_g6s(self, results):
        """
        return the set of canonical g6 labels of the planted motif instances.
        derived by taking the induced subgraph of each planted instance
        (grouped by motif_id) and canonicalising.
        """
        g6s = set()
        for res in results:
            pyg = res['pyg']
            source_nx = res['source_nx']
            if not hasattr(pyg, 'motif_id'):
                continue
            motif_ids = pyg.motif_id.tolist()
            groups = defaultdict(set)
            for node_idx, mid in enumerate(motif_ids):
                if mid > 0:
                    groups[mid].add(node_idx)
            for mid, nodes in groups.items():
                sub, _ = self._induced_subgraph(source_nx, nodes)
                if sub.number_of_nodes() >= 2 and sub.number_of_edges() > 0:
                    g6s.add(self._graph6(sub))
        return g6s

    def _planted_instances(self, results):
        """
        return a list of planted instances across all graphs. each entry:
          { 'source_idx': i, 'motif_id': mid, 'nodes': [...] }
        """
        instances = []
        for src_idx, res in enumerate(results):
            pyg = res['pyg']
            if not hasattr(pyg, 'motif_id'):
                continue
            motif_ids = pyg.motif_id.tolist()
            groups = defaultdict(set)
            for node_idx, mid in enumerate(motif_ids):
                if mid > 0:
                    groups[mid].add(node_idx)
            for mid, nodes in groups.items():
                instances.append({
                    'source_idx': src_idx,
                    'motif_id': mid,
                    'nodes': sorted(nodes),
                })
        return instances

    def _planted_motif_nodes(self, results):
        """
        return one set per source graph of all nodes flagged is_motif=1.
        if is_motif is missing, fall back to motif_id > 0.
        """
        per_graph = []
        for res in results:
            pyg = res['pyg']
            if hasattr(pyg, 'is_motif') and pyg.is_motif is not None:
                flags = pyg.is_motif.tolist()
            elif hasattr(pyg, 'motif_id') and pyg.motif_id is not None:
                flags = [1 if m > 0 else 0 for m in pyg.motif_id.tolist()]
            else:
                flags = []
            per_graph.append({i for i, f in enumerate(flags) if f})
        return per_graph

    @staticmethod
    def _jaccard(a, b):
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def evaluate(self, results, patterns_ranked, instance_jaccard_threshold=0.5):
        """
        compute evaluation metrics against planted ground truth. returns a dict
        with pattern-level, node-level, and instance-level metrics. returns an
        empty dict if no ground truth is present.
        """
        if not self._has_ground_truth(results):
            return {}

        planted_g6s = self._planted_g6s(results)
        predicted_g6s = {p['label'] for p in patterns_ranked}

        # pattern-level
        if predicted_g6s and planted_g6s:
            hit_labels = predicted_g6s & planted_g6s
            precision = len(hit_labels) / len(predicted_g6s)
            recall = len(hit_labels) / len(planted_g6s)
        else:
            hit_labels = set()
            precision = 0.0
            recall = 0.0

        # node-level jaccard: predicted motif nodes vs planted motif nodes,
        # computed per graph and averaged
        planted_per_graph = self._planted_motif_nodes(results)
        predicted_per_graph = [set() for _ in results]
        for pat in patterns_ranked:
            for inst in pat['instances']:
                predicted_per_graph[inst['source_idx']].update(inst['nodes'])

        jaccs = [self._jaccard(p, t)
                 for p, t in zip(predicted_per_graph, planted_per_graph)
                 if p or t]
        node_jaccard = sum(jaccs) / len(jaccs) if jaccs else 0.0

        # instance-level: fraction of planted instances matched by any
        # predicted instance above the jaccard threshold
        planted_instances = self._planted_instances(results)
        pred_instances_per_graph = defaultdict(list)
        for pat in patterns_ranked:
            for inst in pat['instances']:
                pred_instances_per_graph[inst['source_idx']].append(set(inst['nodes']))

        matched = 0
        for p_inst in planted_instances:
            candidates = pred_instances_per_graph.get(p_inst['source_idx'], [])
            planted_set = set(p_inst['nodes'])
            best = max((self._jaccard(planted_set, c) for c in candidates),
                       default=0.0)
            if best >= instance_jaccard_threshold:
                matched += 1
        instance_recall = (matched / len(planted_instances)
                           if planted_instances else 0.0)

        return {
            'planted_g6s': sorted(planted_g6s),
            'predicted_g6s': sorted(predicted_g6s),
            'hit_labels': sorted(hit_labels),
            'pattern_precision': precision,
            'pattern_recall': recall,
            'node_jaccard': node_jaccard,
            'instance_recall': instance_recall,
            'num_planted_instances': len(planted_instances),
            'num_matched_instances': matched,
            'instance_jaccard_threshold': instance_jaccard_threshold,
        }

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
        eval_metrics is provided, an is_planted column is appended.
        """
        planted_g6s = set(eval_metrics.get('planted_g6s', [])) if eval_metrics else set()

        header = f"{'rank':<5} {'g6_label':<10} {'nodes':<6} {'edges':<6} " \
                 f"{'count':<6} {'total_score':<12} {'mean_score':<10}"
        if eval_metrics:
            header += " planted"
        print(header)
        print("-" * len(header))

        for rank, pat in enumerate(patterns_ranked, start=1):
            row = f"{rank:<5} {pat['label']:<10} {pat['nodes']:<6} {pat['edges']:<6} " \
                  f"{pat['count']:<6} {pat['total_score']:<12.3f} {pat['mean_score']:<10.3f}"
            if eval_metrics:
                row += "  yes" if pat['label'] in planted_g6s else "  no"
            print(row)

    def export_eval(self, eval_metrics, out_path):
        """ write the eval metrics dict as a small text summary. """
        if not eval_metrics:
            return None
        with open(out_path, 'w') as f:
            for key, val in eval_metrics.items():
                if isinstance(val, (list, set, tuple)):
                    val = ', '.join(map(str, val))
                f.write(f"{key}: {val}\n")
        return out_path

    def export_all(self,
                   results,
                   out_dir,
                   top_n=10,
                   min_size=3,
                   max_size=8,
                   require_connected=True,
                   min_instances=2,
                   rank_by='total_score',
                   instance_jaccard_threshold=0.5):
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

        eval_metrics = self.evaluate(results, ranked,
                                     instance_jaccard_threshold=instance_jaccard_threshold)

        collection_path = self.export_nemocollection(
            ranked, os.path.join(out_dir, 'motifiesta_collection.txt'))
        input_path, queries_path = self.export_nemomap_inputs(
            ranked, results, out_dir)
        eval_path = (self.export_eval(eval_metrics,
                                      os.path.join(out_dir, 'eval.txt'))
                     if eval_metrics else None)

        self.print_stats(ranked, eval_metrics=eval_metrics)

        return {
            'patterns': ranked,
            'eval': eval_metrics,
            'collection': collection_path,
            'input_graph': input_path,
            'queries': queries_path,
            'eval_file': eval_path,
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

    if out['eval']:
        ev = out['eval']
        print("\neval:")
        print(f"  pattern precision: {ev['pattern_precision']:.3f}")
        print(f"  pattern recall:    {ev['pattern_recall']:.3f}")
        print(f"  node jaccard:      {ev['node_jaccard']:.3f}")
        print(f"  instance recall:   {ev['instance_recall']:.3f}")
        print(f"  planted g6s:       {ev['planted_g6s']}")
        print(f"  predicted g6s:     {ev['predicted_g6s']}")
        print(f"  hit labels:        {ev['hit_labels']}")

    print("\ntop motifs:")
    for rank, pat in enumerate(out['patterns'], start=1):
        planted_flag = ""
        if out['eval']:
            planted_flag = " [planted]" if pat['label'] in out['eval']['planted_g6s'] else ""
        print(f"  {rank}. {pat['label']} ({pat['nodes']}n, {pat['edges']}e) "
              f"count={pat['count']} total_score={pat['total_score']:.3f}"
              f"{planted_flag}")
