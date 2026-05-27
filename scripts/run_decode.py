"""
decode the trained BFS model using the original HashDecoder and evaluate
with the original node-level permutation M-Jaccard.

run from repo root:
    python scripts/run_decode.py
"""
from MotiFiesta.training.decode import HashDecoder

MODEL_ID     = 'louvain_clique_k10_wl_crossweight_filtered'
DATASET_ROOT = 'data/louvain_decomp-clique-p0.05-n500-k10-d0.00'
LEVEL        = 4
HASH_DIM     = 8
TOP_K        = 1


def main():
    decoder = HashDecoder(
        model_id=MODEL_ID,
        dataset_id=DATASET_ROOT,
        hash_dim=HASH_DIM,
        level=LEVEL,
    )

    for layer in decoder.model.layers:
        layer.matching_mode = 'luby'

    decoded_graphs = decoder.decode()

    jaccard = decoder.eval(decoded_graphs, n_motifs=1, top_k=TOP_K)
    print(f"\nM-Jaccard (top_k={TOP_K}): {jaccard:.4f}")

    mot_sigma = decoder.motif_sigma(decoded_graphs)
    print(f"motif sigma by class: {mot_sigma}")


if __name__ == "__main__":
    main()
