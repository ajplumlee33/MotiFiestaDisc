"""single-graph synthetic dataset with embedded motif ground truth.

generates one large erdos-renyi graph with planted motif instances.
used as the source graph for bfs_decompose. supports single or multi-type
motif planting; motif_id encodes type (0 = background, 1..K = motif types).
"""
import random

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx

from MotiFiesta.utils.synthetic import motif_distort


class SysSyntheticDataset(Dataset):
    """single-graph synthetic dataset with embedded motifs.

    args:
        root: pyg dataset root. must contain 'sys_synth' for get_loader dispatch.
        motif_type: motif name or list of names. valid: 'star', 'barbell',
              'wheel', 'random', 'clique', 'lollipop'.
        motif_size: size knob passed to the motif menu.
        n_motifs: instances per motif type.
        parent_size: er background node count (before motifs grafted in).
        parent_e_prob: er edge probability for background and motif-parent links.
        distort_p: per-edge distortion probability per motif instance.
        seed: rng seed.

    node features (x) are NOT set here — bfs_decompose computes within-subgraph
    onehotdegree features after decomposition so features reflect local structure.
    """

    def __init__(self,
                 root,
                 motif_type='clique',
                 motif_size=10,
                 n_motifs=20,
                 parent_size=1000,
                 parent_e_prob=0.1,
                 distort_p=0.0,
                 seed=42,
                 transform=None,
                 pre_transform=None):
        if isinstance(motif_type, str):
            self.motif_types = [motif_type]
        else:
            self.motif_types = list(motif_type)
        self.motif_type = motif_type
        self.motif_size = int(motif_size)
        self.n_motifs = int(n_motifs)
        self.parent_size = int(parent_size)
        self.parent_e_prob = float(parent_e_prob)
        self.distort_p = float(distort_p)
        self.seed = int(seed)

        super().__init__(root, transform, pre_transform)

        self.cached_data = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        motifs_str = '-'.join(self.motif_types)
        return [f'synth_{motifs_str}_k{self.motif_size}_n{self.n_motifs}'
                f'_ps{self.parent_size}_p{self.parent_e_prob:.5f}'
                f'_d{self.distort_p:.2f}_s{self.seed}.pt']

    def download(self):
        pass

    def _build_motif_menu(self):
        ms = self.motif_size
        return {
            'star':     lambda: nx.star_graph(ms),
            'barbell':  lambda: nx.barbell_graph(ms // 2, 2),
            'wheel':    lambda: nx.wheel_graph(ms),
            'random':   lambda: nx.erdos_renyi_graph(ms, self.parent_e_prob, seed=self.seed),
            'clique':   lambda: nx.complete_graph(ms),
            'lollipop': lambda: nx.lollipop_graph(ms // 2, ms),
        }

    def process(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

        menu = self._build_motif_menu()
        for mt in self.motif_types:
            if mt not in menu:
                raise ValueError(
                    f"unknown motif_type {mt!r}; valid: {sorted(menu.keys())}"
                )

        if self.n_motifs <= 0:
            raise ValueError(f"n_motifs must be > 0, got {self.n_motifs}")
        total_instances = self.n_motifs * len(self.motif_types)
        if total_instances >= self.parent_size:
            raise ValueError(
                f"parent_size ({self.parent_size}) must exceed total instances "
                f"({total_instances})"
            )

        G = nx.erdos_renyi_graph(self.parent_size, self.parent_e_prob, seed=self.seed)
        nx.set_node_attributes(G, 0, 'is_motif')
        nx.set_node_attributes(G, 0, 'motif_id')

        anchor_nodes = random.sample(sorted(G.nodes()), total_instances)

        next_id = self.parent_size
        inst = 0
        for type_idx, mt in enumerate(self.motif_types, start=1):
            template = menu[mt]()
            for n in template.nodes():
                template.nodes[n]['class'] = random.randint(0, 1)

            for _ in range(self.n_motifs):
                motif = motif_distort(template.copy(),
                                      list(template.nodes()),
                                      p=self.distort_p)

                anchor = anchor_nodes[inst]
                G.remove_node(anchor)

                motif_size = len(motif)
                motif_ids = list(range(next_id, next_id + motif_size))
                mapping = {old: new for old, new
                           in zip(sorted(motif.nodes()), motif_ids)}
                motif = nx.relabel_nodes(motif, mapping)

                for n in motif_ids:
                    G.add_node(n, is_motif=1, motif_id=type_idx)
                G.add_edges_from(motif.edges())

                link_motif = [n for n in motif_ids
                              if random.random() < self.parent_e_prob]
                link_parent = [n for n in G.nodes()
                               if G.nodes[n]['is_motif'] == 0
                               and random.random() < self.parent_e_prob]
                G.add_edges_from(zip(link_parent, link_motif))

                next_id += motif_size
                inst += 1

        G = nx.convert_node_labels_to_integers(G)
        for n in G.nodes():
            G.nodes[n].pop('class', None)

        data = from_networkx(G)

        data.motif_id = data.motif_id.long()
        data.is_motif = data.is_motif.long()

        assert data.motif_id.max().item() == len(self.motif_types), (
            f"motif_id max should be {len(self.motif_types)}, "
            f"got {data.motif_id.max().item()}"
        )
        assert data.is_motif.sum().item() > 0, "no motif nodes tagged"

        data.num_embedded_instances = torch.tensor(total_instances, dtype=torch.long)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return self.cached_data
