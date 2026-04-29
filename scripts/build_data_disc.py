"""Generate the MOTIF TYPE comparison grid against MotiFiesta's tests.

Mirrors the first block of build_data_motifiesta:
    for m_type in ['barbell', 'star', 'random', 'clique']:
        for d in [0, .01, .02, .05, .1, .2]:

But scaled to single-graph mode. Each cell becomes one SysSyntheticDataset.
Generation cost ~15s per cell, ~6 minutes total for the 24-cell grid.

Naming convention matches theirs: synth-{m_type}-d{d:.2f}, with a
sys_synth_ prefix so loading.get_loader's dispatch picks the right class.
"""
import os
import time
import traceback

from MotiFiesta.utils.sys_synthetic import SysSyntheticDataset

MOTIF_TYPES = ['barbell', 'star', 'random', 'clique']
DISTORT_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

# single-graph params, tuned for connectivity at this scale
# parent_size=4000, p=0.0017 → mean degree ~7, max ~14-18, connected
# generates in ~5s; 200 graftings finish in ~10s
COMMON = dict(
    motif_size=10,
    n_motifs=200,
    parent_size=4000,
    parent_e_prob=0.0017,
    random_e_prob=0.3,   # only used when motif_type='random' — gives a
                         # 10-node ER motif with mean degree ~3 (visible
                         # structure, not empty)
    seed=42,             # match MotiFiesta's seed
)

DATA_ROOT = 'data'

def build_one(motif_type, distort_p):
    """Build a single dataset cell. Returns (success, message, elapsed_seconds)."""
    name = f'sys_synth-{motif_type}-d{distort_p:.2f}'
    root = os.path.join(DATA_ROOT, name)

    t0 = time.time()
    try:
        ds = SysSyntheticDataset(
            root=root,
            motif_type=motif_type,
            distort_p=distort_p,
            **COMMON,
        )
        d = ds[0]
        elapsed = time.time() - t0

        # log instance
        n_motif_nodes = d.is_motif.sum().item()
        n_total = d.num_nodes
        pct = 100 * n_motif_nodes / n_total
        msg = (f"{n_total} nodes, {d.edge_index.size(1)//2} edges, "
               f"{n_motif_nodes} motif nodes ({pct:.0f}%)")
        return True, msg, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return False, f"{type(e).__name__}: {e}", elapsed


def main():
    cells = [(mt, d) for mt in MOTIF_TYPES for d in DISTORT_LEVELS]
    total = len(cells)
    print(f"Building {total}-cell grid: {len(MOTIF_TYPES)} motif types × "
          f"{len(DISTORT_LEVELS)} distort levels")
    print(f"Common params: {COMMON}")
    print(f"Output: {DATA_ROOT}/sys_synth-<type>-d<distort>/\n")

    grid_start = time.time()
    successes, failures = [], []

    for i, (motif_type, distort_p) in enumerate(cells, 1):
        cell_id = f'{motif_type}-d{distort_p:.2f}'
        print(f"[{i:2d}/{total}] {cell_id:>20s}  ...  ", end='', flush=True)

        ok, msg, elapsed = build_one(motif_type, distort_p)
        status = 'OK' if ok else 'FAIL'
        print(f"{status} ({elapsed:.1f}s)  {msg}")
        (successes if ok else failures).append((cell_id, msg, elapsed))

    grid_elapsed = time.time() - grid_start
    print(f"\nGrid complete in {grid_elapsed/60:.1f} min")
    print(f"  successes: {len(successes)}/{total}")
    if failures:
        print(f"  failures:  {len(failures)}")
        for cell_id, msg, _ in failures:
            print(f"    {cell_id}: {msg}")


if __name__ == '__main__':
    main()