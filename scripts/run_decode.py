"""
decode the trained model on the synthetic clique dataset and write outputs to
decoded/{MODEL_ID}/. decode log is also written to logs/{MODEL_ID}/decode.log.

run from the repo root:
    python scripts/run_decode.py
"""
import os
from MotiFiesta.training.disc_decode import DiscHashDecoder

MODEL_ID = 'sys_synth-clique-com-luby-mlp-p0.002-n1000-k10-d0.00-sil-b256'
DATASET_ID = 'sys_synth-clique-p0.002-n1000-k10-d0.00'
DATASET_ROOT = 'data/sys_synth-clique-p0.002-n1000-k10-d0.00'
OUT_DIR = f'decoded/{MODEL_ID}'
LOG_DIR = f'logs/{MODEL_ID}'


DATASET_KWARGS = dict(
    distort_p=0.0,
    parent_e_prob=0.002,
    motif_type='clique',
    motif_size=10,
    n_motifs=20,
    parent_size=1000,
)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'decode.log')

    decoder = DiscHashDecoder(
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        dataset_root=DATASET_ROOT,
        level=4,
        hash_dim=8,
        batch_size=64,
        **DATASET_KWARGS,
    )

    for layer in decoder.model.layers:
        layer.matching_mode = 'luby'

    results = decoder.decode()

    lines = []

    lines.append("\n=== pre-filter diagnostics ===")
    diag = decoder.diagnose(results)

    # k=1 is the paper metric (top-1 sigma bucket vs all true motif nodes).
    # k=N shows the upper bound — best single bucket over all candidates.
    n = diag['total_spotlights']
    lines.append("\n=== multi-k eval (K=1 is paper metric, K=N is upper bound) ===")
    for k in [1, 4, n]:
        ev = decoder.eval(results, top_k=k, min_size=1, max_size=999)
        lines.append(f"  K={k:3d}  jaccard={ev['jaccard']:.4f}  instance_recall={ev['instance_recall']:.4f}")

    lines.append("\n=== top motifs ===")
    out = decoder.export_all(
        results,
        out_dir=OUT_DIR,
        top_n=10,
        min_size=5,
        max_size=50,
        require_connected=False,
        min_instances=1,
        rank_by='total_score',
    )

    lines.append(f"\ncollection file : {out['collection']}")
    lines.append(f"input graph     : {out['input_graph']}")
    lines.append(f"queries file    : {out['queries']}")

    output = "\n".join(lines)
    print(output)
    with open(log_path, 'w') as f:
        f.write(output + "\n")
    print(f"\ndecode log      : {log_path}")


if __name__ == "__main__":
    main()
