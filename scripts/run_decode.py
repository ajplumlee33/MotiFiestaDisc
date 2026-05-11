"""
example script that decodes the current trained model on the mips ppi source
graph and writes outputs to decoded/test/.

run from the repo root:
    python scripts/run_decode.py
"""
from MotiFiesta.training.disc_decode import DiscHashDecoder


def main():
    decoder = DiscHashDecoder(
        #model_id='test',
        #dataset_id='mips_torch',
        #dataset_root='data/mips_torch',
        model_id='sys_synth-clique-com-d0.05',
        dataset_id='sys_synth-clique-d0.05',
        dataset_root='data/sys_synth-clique-d0.05',
        level=3,
    )

    results = decoder.decode()

    print("\n=== pre-filter diagnostics ===")
    decoder.diagnose(results)

    print("\n=== top motifs ===")
    out = decoder.export_all(
        results,
        out_dir='decoded/sys_synth-clique-com-d0.05',
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
    if out['eval_file']:
        print(f"eval file       : {out['eval_file']}")


if __name__ == "__main__":
    main()
