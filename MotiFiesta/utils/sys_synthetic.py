"""Single-graph synthetic dataset with planted motif ground truth.

Mirrors SysTxtDataset's shape/interface (one large graph, igraph sidecar,
OneHotDegree features) but plants known motif instances so the decoder's
evaluate() path can score discovery accuracy.

API mirrors synthetic.generate_instances: one motif type at one size per
dataset, count controlled by n_motifs. The motif_menu dict here is copied
verbatim from generate_instances so geometry is identical between the two
pipelines.
"""
import random

import numpy as np
import networkx as nx
import torch
import torch_geometric.transforms as T
from igraph import Graph
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx

from MotiFiesta.utils.synthetic import motif_embed, generate_parent_erdos


class SysSyntheticDataset(Dataset):
    """Single-graph synthetic dataset with planted motifs of one type.

    Args:
        root: PyG dataset root. Must contain 'sys_synth' so that
              loading.get_loader dispatches correctly.
        motif_type: which motif from the menu — one of 'star', 'barbell',
              'wheel', 'random', 'clique', 'lollipop'. Same names as
              synthetic.generate_instances.
        motif_size: size knob, interpreted by the menu (matches
              generate_instances exactly: 'star' has motif_size+1 nodes,
              'barbell' uses motif_size//2 for each bell, etc).
        n_motifs: number of instances of this motif to plant.
        parent_size: number of nodes in the Erdős-Rényi parent (before
              motifs are grafted in; final graph is larger).
        parent_e_prob: edge probability for the parent ER generation and the
              motif<->parent linking step.
        random_e_prob: edge probability used internally when motif_type='random'.
              Defaults to None, meaning "fall back to parent_e_prob". For
              single-graph mode the parent is large and parent_e_prob must be
              tiny (~0.001) to keep the backbone sparse — that same value
              would produce empty random motifs. Override this to ~0.3-0.5
              when motif_type='random'. Ignored for non-random motif types.
        distort_p: per-edge perturbation probability applied to each motif
              instance at plant time. 0.0 = exact topology, >0 = noisier.
        seed: RNG seed for deterministic generation.
        max_degree: cap for T.OneHotDegree. If None, uses actual max degree.
        n_features: optional override for num_features.
    """

    def __init__(self,
                 root,
                 motif_type='clique',
                 motif_size=5,
                 n_motifs=10,
                 parent_size=1000,
                 parent_e_prob=0.005,
                 random_e_prob=None,
                 distort_p=0.0,
                 seed=0,
                 max_degree=None,
                 n_features=None,
                 transform=None,
                 pre_transform=None):
        self.motif_type = str(motif_type)
        self.motif_size = int(motif_size)
        self.n_motifs = int(n_motifs)
        self.parent_size = int(parent_size)
        self.parent_e_prob = float(parent_e_prob)
        # fall back to parent_e_prob if random_e_prob isn't given
        self.random_e_prob = (float(random_e_prob)
                              if random_e_prob is not None
                              else float(parent_e_prob))
        self.distort_p = float(distort_p)
        self.seed = int(seed)
        self.max_degree = max_degree
        self.n_features = n_features

        super().__init__(root, transform, pre_transform)

        # Load processed graph once and build igraph sidecar
        self.cached_data = torch.load(self.processed_paths[0], weights_only=False)
        edges = self.cached_data.edge_index.t().tolist()
        self.ig_graph = Graph(
            n=self.cached_data.num_nodes,
            edges=edges,
            directed=False,
        )
        self.ig_graph.simplify()

    # ------------------------------------------------------------------
    # PyG plumbing
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self):
        # No raw files — data is generated in process()
        return []

    @property
    def processed_file_names(self):
        return ['system_synth_graph.pt']

    @property
    def num_features(self):
        if self.n_features is not None:
            return self.n_features
        return self.cached_data.num_features

    def download(self):
        # Nothing to download for synthetic data
        pass

    def _build_motif_menu(self):
        """Build the motif menu as a dict of zero-arg constructors.

        Geometry matches synthetic.generate_instances verbatim, except the
        'random' entry uses self.random_e_prob (which falls back to
        parent_e_prob when not set). Lazy construction (lambdas) is needed
        because some menu entries fail at small motif_size (e.g. barbell
        requires bell size >=2, so motif_size<4 crashes). The original
        eager dict in generate_instances has the same latent issue; it just
        isn't triggered there because that function only ever uses one
        entry per call.
        """
        ms = self.motif_size
        return {
            'star':     lambda: nx.star_graph(ms),
            'barbell':  lambda: nx.barbell_graph(ms // 2, 2),
            'wheel':    lambda: nx.wheel_graph(ms),
            'random':   lambda: generate_parent_erdos(ms, p=self.random_e_prob, seed=self.seed),
            'clique':   lambda: nx.complete_graph(ms),
            'lollipop': lambda: nx.lollipop_graph(ms // 2, ms),
        }

    def process(self):
        # Determinism. motif_embed/motif_distort use `random` internally,
        # and generate_parent_erdos uses np.random via networkx.
        random.seed(self.seed)
        np.random.seed(self.seed)

        menu = self._build_motif_menu()
        if self.motif_type not in menu:
            raise ValueError(
                f"unknown motif_type {self.motif_type!r}; "
                f"valid: {sorted(menu.keys())}"
            )
        template = menu[self.motif_type]()

        if self.n_motifs <= 0:
            raise ValueError(f"n_motifs must be > 0, got {self.n_motifs}")
        if self.n_motifs >= self.parent_size:
            raise ValueError(
                f"parent_size ({self.parent_size}) must exceed n_motifs "
                f"({self.n_motifs}) to leave room for anchor nodes"
            )

        # Build the {motif_id: nx.Graph} dict for motif_embed.
        # Each instance is an independent copy so per-instance distortion
        # doesn't propagate across instances.
        motifs_to_plant = {i: template.copy() for i in range(1, self.n_motifs + 1)}

        # motif_distort (called inside motif_embed) reads node['class'],
        # so seed those. They won't flow into features — we strip before
        # converting to pyg.
        for m in motifs_to_plant.values():
            for n in m.nodes():
                m.nodes[n]['class'] = random.randint(0, 1)

        # -- Embed. Returns (planted, original, randomized) ------------------
        # We only want the planted graph. Randomized/original are discarded;
        # they're useful for the many-small-graphs use case, not this one.
        planted, _, _ = motif_embed(
            motifs_to_plant,
            parent_size=self.parent_size,
            parent_e_prob=self.parent_e_prob,
            distort_p=self.distort_p,
            embed_prob=1.0,
            n_classes=2,
        )

        # Strip node attrs we don't want PyG to pick up as separate tensors.
        # Leaves `is_motif` and `motif_id` as the only per-node attrs.
        for n in planted.nodes():
            for attr in ('class', 'deg'):
                planted.nodes[n].pop(attr, None)

        # -- Convert to PyG --------------------------------------------------
        data = from_networkx(planted)

        # OneHotDegree features to match sys_txt's pattern.
        if self.max_degree is not None:
            max_deg = int(self.max_degree)
        else:
            max_deg = int(max(dict(planted.degree()).values()))
        data = T.OneHotDegree(max_deg)(data)

        # Dtype hygiene — decoder expects Long tensors for ground truth
        data.motif_id = data.motif_id.long()
        data.is_motif = data.is_motif.long()

        # Sanity checks
        assert data.motif_id.max().item() == self.n_motifs, (
            f"motif_id max ({data.motif_id.max().item()}) != "
            f"n_motifs ({self.n_motifs})"
        )
        assert data.is_motif.sum().item() > 0, "no motif nodes tagged"

        data.num_planted_instances = torch.tensor(self.n_motifs, dtype=torch.long)

        torch.save(data, self.processed_paths[0])

    # ------------------------------------------------------------------
    # Single-graph Dataset contract
    # ------------------------------------------------------------------

    def len(self):
        return 1

    def get(self, idx):
        return self.cached_data


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Toy run. After any change to this file:
    #   rm -rf data/sys_synth_toy/processed
    ds = SysSyntheticDataset(
        root="data/sys_synth_toy",
        motif_type='clique',
        motif_size=4,
        n_motifs=5,
        parent_size=80,
        parent_e_prob=0.05,
        distort_p=0.0,
        seed=0,
    )
    d = ds[0]
    print(f"motif_type      : {ds.motif_type} (size={ds.motif_size})")
    print(f"num_nodes       : {d.num_nodes}")
    print(f"num_edges       : {d.edge_index.size(1) // 2} (undirected)")
    print(f"num_features    : {ds.num_features}")
    print(f"x.shape         : {tuple(d.x.shape)}")
    print(f"motif_id range  : [{d.motif_id.min().item()}, {d.motif_id.max().item()}]")
    print(f"is_motif sum    : {d.is_motif.sum().item()}  (total motif nodes)")
    print(f"planted count   : {d.num_planted_instances.item()}")
    print(f"ig_graph        : {ds.ig_graph.vcount()} verts, {ds.ig_graph.ecount()} edges")

    from collections import Counter
    mid_counts = Counter(d.motif_id.tolist())
    print("per-motif_id node counts:")
    for mid in sorted(mid_counts):
        tag = "background" if mid == 0 else f"instance {mid}"
        print(f"  {tag:>15s}: {mid_counts[mid]}")
