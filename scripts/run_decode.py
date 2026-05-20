"""
example script that decodes the current trained model on the mips ppi source
graph and writes outputs to decoded/test/.

run from the repo root:
    python scripts/run_decode.py
"""
from MotiFiesta.training.disc_decode import DiscHashDecoder


def main():
    decoder = DiscHashDecoder(
        model_id='sys_synth-clique-com2-p0.1-n1000-d0.00',
        dataset_id='sys_synth-clique-p0.1-n1000-d0.00',
        dataset_root='data/sys_synth-clique-p0.1-n1000-d0.00',
        level=3,
        hash_dim=4
    )

    for layer in decoder.model.layers:
        layer.parallel_matching = True

    results = decoder.decode()

    print("\n=== pre-filter diagnostics ===")
    decoder.diagnose(results)

    print("\n=== top motifs ===")
    out = decoder.export_all(
        results,
        out_dir='decoded/sys_synth-clique-com2-p0.1-n1000-d0.00',
        top_n=10,
        min_size=3,
        max_size=8,
        require_connected=True,
        min_instances=1,
        rank_by='total_score',
    )

    print(f"\ncollection file : {out['collection']}")
    print(f"input graph     : {out['input_graph']}")
    print(f"queries file    : {out['queries']}")


if __name__ == "__main__":
    main()
