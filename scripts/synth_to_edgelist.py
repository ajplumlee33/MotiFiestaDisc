"""
convert a sys_synth dataset to a plain edge list.

usage:
    python synth_to_edgelist.py <dataset_id> [output_path]

example:
    python synth_to_edgelist.py sys_synth-clique-d0.00 edges.txt

writes one undirected edge per line as 'u v'. zero-indexed.
"""
import os
import sys

import torch

from MotiFiesta.training.loading import get_loader


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_id = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else f'{dataset_id}_edges.txt'

    root = f'data/{dataset_id}' if not os.path.isabs(dataset_id) else dataset_id
    dataset = get_loader(root=root, name=dataset_id, max_degree=18)

    seen = set()
    edges = []

    for g_pair in dataset['dataset_whole']:
        g = g_pair['pos'] if isinstance(g_pair, dict) else g_pair

        for u, v in g.edge_index.t().tolist():
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            edges.append(key)

    with open(out_path, 'w') as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")

    print(f"wrote {len(edges)} undirected edges to {out_path}")
    print(f"nodes: {max(max(u, v) for u, v in edges) + 1 if edges else 0}")


if __name__ == "__main__":
    main()
