"""single-graph synthetic dataset with embedded motif ground truth.

mirrors systxtdataset's shape/interface (one large graph, igraph sidecar,
onehotdegree features) but embeds known motif instances so the decoder's
evaluate() path can score discovery accuracy.

api mirrors synthetic.generate_instances. motif_type accepts either a
single string (single-type planting, all instances share motif_id=1) or
a list of strings (multi-type planting, instances of motif_types[k] get
motif_id k+1). n_motifs is per-type: total instances planted equals
n_motifs * len(motif_type). the motif_menu dict here is copied verbatim
from generate_instances so geometry is identical between the two pipelines.
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
    """single-graph synthetic dataset with embedded motifs.

    args:
        root: pyg dataset root. must contain 'sys_synth' so that
              loading.get_loader dispatches correctly.
        motif_type: which motif(s) from the menu. either a single name
              or a list of names. valid names: 'star', 'barbell', 'wheel',
              'random', 'clique', 'lollipop'. when a list is passed,
              instances of each type get distinct motif_id values
              (1, 2, ..., K) so the decoder can run K-class M-Jaccard.
              same names as synthetic.generate_instances.
        motif_size: size knob, interpreted by the menu (matches
              generate_instances exactly: 'star' has motif_size+1 nodes,
              'barbell' uses motif_size//2 for each bell, etc).
        n_motifs: number of instances per motif type. with K types, total
              instances planted = n_motifs * K.
        parent_size: number of nodes in the Erdős-Rényi parent (before
              motifs are grafted in; final graph is larger).
        parent_e_prob: edge probability for the parent er generation and the
              motif<->parent linking step.
        random_e_prob: edge probability used internally when 'random' is
              among the motif types. defaults to none, meaning "fall back
              to parent_e_prob". For single-graph mode the parent is large
              and parent_e_prob must be tiny (~0.001) to keep the backbone
              sparse — that same value would produce empty random motifs.
              override this to ~0.3-0.5 when 'random' is in motif_type.
              ignored otherwise.
        distort_p: per-edge perturbation probability applied to each motif
              instance at embed time. 0.0 = exact topology, >0 = noisier.
        seed: RNG seed for deterministic generation.
        max_degree: cap for t.onehotdegree. if none, uses actual max degree.
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
        # normalize motif_type to a list; preserve original input for repr
        if isinstance(motif_type, str):
            self.motif_types = [motif_type]
        else:
            self.motif_types = list(motif_type)
        self.motif_type = motif_type
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

        # load processed graph once and build igraph
        self.cached_data = torch.load(self.processed_paths[0], weights_only=False)
        edges = self.cached_data.edge_index.t().tolist()
        self.ig_graph = Graph(
            n=self.cached_data.num_nodes,
            edges=edges,
            directed=False,
        )
        self.ig_graph.simplify()

    # ------------------------------------------------------------------
    # pyg overrides
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self):
        # no raw files — data is generated in process()
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
        # nothing to download for synthetic data
        pass

    def _build_motif_menu(self):
        """build the motif menu as a dict of zero-arg constructors.

        geometry matches synthetic.generate_instances verbatim, except the
        'random' entry uses self.random_e_prob (which falls back to
        parent_e_prob when not set). lazy construction (lambdas) is needed
        because some menu entries fail at small motif_size (e.g. barbell
        requires bell size >=2, so motif_size<4 crashes).
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
        # determinism. motif_embed/motif_distort use `random` internally,
        # and generate_parent_erdos uses np.random via networkx.
        random.seed(self.seed)
        np.random.seed(self.seed)

        menu = self._build_motif_menu()
        for mt in self.motif_types:
            if mt not in menu:
                raise ValueError(
                    f"unknown motif_type {mt!r}; "
                    f"valid: {sorted(menu.keys())}"
                )

        if self.n_motifs <= 0:
            raise ValueError(f"n_motifs must be > 0, got {self.n_motifs}")
        total_instances = self.n_motifs * len(self.motif_types)
        if total_instances >= self.parent_size:
            raise ValueError(
                f"parent_size ({self.parent_size}) must exceed total instances "
                f"({total_instances} = n_motifs {self.n_motifs} * "
                f"types {len(self.motif_types)}) to leave room for anchor nodes"
            )

        # build the {dict_key: nx.graph} for motif_embed. dict keys must be
        # unique, but per-instance motif_ids are remapped to type ids below
        # (instances of motif_types[k] -> motif_id k+1) so motif_id encodes
        # type, matching the original synthetic.py convention.
        templates = {mt: menu[mt]() for mt in self.motif_types}
        motifs_to_embed = {}
        key_to_type_id = {}
        next_key = 1
        for type_idx, mt in enumerate(self.motif_types, start=1):
            template = templates[mt]
            for _ in range(self.n_motifs):
                motifs_to_embed[next_key] = template.copy()
                key_to_type_id[next_key] = type_idx
                next_key += 1

        # motif_distort (called inside motif_embed) reads node['class']
        for m in motifs_to_embed.values():
            for n in m.nodes():
                m.nodes[n]['class'] = random.randint(0, 1)

        # -- embed. returns (embedded, original, randomized) ------------------
        # randomized/original are discarded;
        # they're useful for the many-small-graphs use case, not this one.
        embedded, _, _ = motif_embed(
            motifs_to_embed,
            parent_size=self.parent_size,
            parent_e_prob=self.parent_e_prob,
            distort_p=self.distort_p,
            embed_prob=1.0,
            n_classes=2,
        )

        # remap raw per-instance motif_ids to type ids in [1..K]
        for n in embedded.nodes():
            raw_mid = embedded.nodes[n].get('motif_id', 0)
            if raw_mid > 0:
                embedded.nodes[n]['motif_id'] = key_to_type_id[raw_mid]

        # strip node attrs so pyg doesn't pick up as separate tensors.
        # leaves `is_motif` and `motif_id` as the only per-node attrs.
        for n in embedded.nodes():
            for attr in ('class', 'deg'):
                embedded.nodes[n].pop(attr, None)

        # -- convert to pyg --------------------------------------------------
        data = from_networkx(embedded)

        # onehotdegree features to match sys_txt's pattern.
        if self.max_degree is not None:
            max_deg = int(self.max_degree)
        else:
            max_deg = int(max(dict(embedded.degree()).values()))
        data = T.OneHotDegree(max_deg)(data)

        # dtype hygiene — decoder expects long tensors for ground truth
        data.motif_id = data.motif_id.long()
        data.is_motif = data.is_motif.long()

        # sanity checks
        n_types = len(self.motif_types)
        assert data.motif_id.max().item() == n_types, (
            f"motif_id max should be {n_types} (one per type), got "
            f"{data.motif_id.max().item()}"
        )
        assert data.is_motif.sum().item() > 0, "no motif nodes tagged"

        data.num_embedded_instances = torch.tensor(total_instances, dtype=torch.long)

        torch.save(data, self.processed_paths[0])

    # ------------------------------------------------------------------
    # single-graph dataset contract
    # ------------------------------------------------------------------

    def len(self):
        return 1

    def get(self, idx):
        return self.cached_data


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # base run, after any change to this file:
    #   rm -rf data/sys_synth/processed data/sys_synth_toy_multi/processed
    print("=== single-type ===")
    ds = SysSyntheticDataset(
        root="data/sys_synth",
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
    print(f"embedded count   : {d.num_embedded_instances.item()}")
    print(f"ig_graph        : {ds.ig_graph.vcount()} verts, {ds.ig_graph.ecount()} edges")

    from collections import Counter
    mid_counts = Counter(d.motif_id.tolist())
    print("per-motif_id node counts:")
    for mid in sorted(mid_counts):
        tag = "background" if mid == 0 else f"type {mid}"
        print(f"  {tag:>15s}: {mid_counts[mid]}")

    print("\n=== multi-type ===")
    ds = SysSyntheticDataset(
        root="data/sys_synth_multi",
        motif_type=['clique', 'star', 'barbell'],
        motif_size=5,
        n_motifs=4,
        parent_size=120,
        parent_e_prob=0.05,
        distort_p=0.0,
        seed=0,
    )
    d = ds[0]
    print(f"motif_type      : {ds.motif_type}")
    print(f"num_nodes       : {d.num_nodes}")
    print(f"motif_id range  : [{d.motif_id.min().item()}, {d.motif_id.max().item()}]")
    print(f"is_motif sum    : {d.is_motif.sum().item()}")
    print(f"embedded count   : {d.num_embedded_instances.item()}")
    mid_counts = Counter(d.motif_id.tolist())
    print("per-motif_id node counts:")
    for mid in sorted(mid_counts):
        tag = "background" if mid == 0 else f"type {mid} ({ds.motif_types[mid-1]})"
        print(f"  {tag:>25s}: {mid_counts[mid]}")
