# Project Rules

## Paper references
- Never cite the paper (Oliver et al. 2022, "Approximate Network Motif Mining via Graph Learning") without quoting the exact page and text.
- Do NOT cite code comments as if they are paper quotes. Comments in disc_decode.py (e.g. "algorithm 2, line 7") were written by Claude in a prior session and are NOT authoritative paper quotes.
- If inferring or guessing about what the paper says, say so explicitly.

## Dataset background edge probability
- The paper (Appendix A.1, p.12) uses **p=0.1** for background graphs in the multi-graph setting on small graphs (~20 nodes, twice motif size).
- The codebase convention of **parent_e_prob=0.0017** in build_data_disc.py is NOT derived from the paper. It was set for the single-large-graph setting and makes the background very sparse, which inflates M-Jaccard by making planted motifs trivially easy to find.
- Do NOT claim p=0.0017 is "faithful to the paper." It is not. These are different experimental settings with incomparable background densities.
- For a large single graph (4000+ nodes), p=0.1 would produce an essentially complete graph. If comparing difficulty to the paper, calculate a target average degree from the paper's small-graph setting and work backwards.

## Connectivity and motif validity
- The paper (Section 2, p.2) defines connectivity as a property of the **motif** g, not the host graph G: "Every g is connected (i.e. ∃(p ∼ q)∀(p, q) ∈ N × N)."
- EdgePool architecturally guarantees connected spotlights (Section 3.1, p.4): "because all pooling events are carried out over edges, we guarantee that the subgraph being represented is always connected." The `require_connected=True` in the decoder enforces this at decode time.
- The host graph G does NOT need to be connected — the paper (Appendix A.1, p.12) generates each G with ER(p=0.1) and never enforces connectivity on G.
- The connectivity-forcing code in sys_synthetic.py (connecting all components to a single hub node) was NOT required by the paper and creates a degenerate high-degree hub that corrupts degree features. It was correctly removed.
- The paper operates on 1000 **small graphs** (~20 nodes, twice motif size), NOT one large graph. Our single-large-graph adaptation is a fundamental departure from the paper's setup.

## Project goal
- The project contribution is a **graph decomposition layer** that breaks a large input graph into small subgraphs suitable for the original MotiFiesta model, plus possible enhancements to MotiFiesta itself.
- Two decomposition strategies are implemented in `MotiFiesta/utils/bfs_decompose.py` and switchable via `DECOMP_METHOD` in `build_data_bfs.py`:
  - `'bfs'`: BFS-seeded fixed-size subgraphs. Validated at 0.55 M-Jaccard (paper: 0.58 ± 0.14, K10 clique d=0.00). Produces ~45% pure-background subgraphs on dense source graphs.
  - `'louvain'`: Louvain community detection with a `max_size` ceiling. Keeps cliques intact (0% pure-background subgraphs, 43% with complete K10). Variable-size subgraphs up to `max_size=30`. **Current default.**
- Do NOT suggest reverting to neighborhood batching (NeighborLoader/SysLoader). That approach had a hard ceiling of ~0.35 M-Jaccard and was replaced by the decomposition approach.
- Enhancements to MotiFiesta (model architecture, training objective, decoder) are in scope alongside the decomposition layer.

## Decomposition — source graph requirements
- **build_data_bfs.py MUST generate a large source graph and decompose it.** Do NOT generate small graphs directly — that defeats the project's contribution.
- The core constraint: background density within a subgraph ≈ C(subgraph_size, 2) × p_source. To match the paper's ER(20, p=0.1) subgraph statistics, p_source must be ≈ 0.1, which limits source graph size to n ≈ 500–2000 (p=0.1 on n > 4000 produces a near-complete graph).
- Valid configuration: n=500, p=0.1, n_motifs=25 K10 cliques → 33% motif coverage in source graph, ~28 subgraphs per source graph with Louvain. Run N_GRAPHS=40 source graphs (different seeds) to reach ~1120 total subgraphs.
- BFS limitation on dense graphs: the `in_queue` set grows to cover almost all nodes during core expansion, leaving the halo effectively empty. Motif cliques are still often captured whole (via degree-based seeding) but ~45% of subgraphs end up pure-background.
- Louvain advantage: maximizes intra-community edges, so K10 cliques form natural communities and are never severed. Communities exceeding `max_size` are BFS-split within the community. Feature dimension changes with `max_size` (max_size=30 → 61 features vs BFS max_size=20 → 41 features) — model must be retrained when switching.

## Training entrypoint
- **`sys_train.py` is deprecated.** Do NOT use it or suggest it.
- The correct training entrypoint is `scripts/motifiesta train` (without `--disc`), which calls `motif_train` in `MotiFiesta/training/train.py`.
- `sys_train` ignores `batch['neg']` from the dataset and generates rewired negatives via `_make_neg(pos)` instead. It also requires `source_graph.ig_graph` for the WWL rec kernel, which `LouvainDecomposedDataset` / `BFSDecomposedDataset` do not provide.
- `motif_train` uses `rec_loss_wl` (WL subtree kernel) and reads negatives from `batch['neg']`. This is the correct path for decomposed datasets.
- If WWL kernel support is needed in training, it must be added to `motif_train`, not to `sys_train`.

## DiscModel evaluation metric
- The primary metric is **M-Jaccard with sigma ranking** (`rank_by='sigma'`), which matches the original paper's evaluation (ranks hash buckets by average cumulative score, then computes permutation Jaccard against true motif labels).
- `score-only` (Otsu threshold on node scores, no LSH) is a secondary diagnostic only — do NOT report it as the main result or compare it to paper numbers.
- `rank_by='count'` is not the paper metric either.
- Paper baselines: star k10 ≈ 0.43, barbell ≈ 0.58 (Table 1).

## DiscModel training (MotiFiestaDisc)
- The new continuous-embedding model is `MotiFiestaDisc` in `MotiFiesta/training/disc_model.py`.
- Training entrypoint: `scripts/motifiesta train --model-type subgraph`.
- Dataset for DiscModel testing: use `scripts/build_synth_data.py` to pre-generate pos/neg pairs into `data/synth-{type}-k{size}/processed/`. Then pass `--dataset synth_pairs --data_root data/synth-...` to the training command.
- Do NOT use `LouvainDecomposedDataset` for DiscModel testing. The synth pairs dataset from `build_synth_data.py` is NOT a Louvain-decomposed large graph. `LouvainDecomposedDataset` is only for Louvain-decomposed subgraphs from `build_data_bfs.py`.
- The routing bug: `loading.py` currently routes `--dataset synth_pairs` to `LouvainDecomposedDataset` because both use the same file format. This is technically functional but semantically wrong — fix by adding a dedicated `PrebuiltPairsDataset` class routed to `synth_pairs`.

## Main branch model reference
- **ALWAYS say "main branch model" — NEVER "original model". The word "original" is ambiguous and causes confusion.**
- **NEVER reason from memory. ALWAYS run:**
  - `git show main:MotiFiesta/training/model.py`
  - `git show main:MotiFiesta/training/edge_pool.py`
  - `git show main:MotiFiesta/training/train.py`
- The files on the current branch have been modified. Do NOT treat them as representative of the original design.
- **THE ORIGINAL MODEL HAS NO GIN. NEVER.** It uses stacked EdgePooling layers, each with its own `transform` (Linear) and `score_net` (Linear(out_channels, 1)). There is no GINConv anywhere in the original codebase.

## Original rec_loss (main branch) — exact implementation
- `K_predict = matrix_cosine(x_merged[:num_nodes], x_merged[:num_nodes])` — cosine gram of SUPERNODE EMBEDDINGS (EdgePooling output `x_merged`) at each pooling level.
- `K_true = build_wwl_K(subgraphs[:num_nodes], node_features[:num_nodes])` — WWL kernel on **extracted NetworkX spotlight subgraphs**. Uses `get_edge_subgraphs()` to extract the actual induced subgraph for each supernode's spotlight from the merge tree. Then `wwl(graphs, node_features, num_iterations=4)` computes Wasserstein distance between WL label histograms. Returns K_true ∈ (0, 1].
- K_true ∈ (0, 1] (all positive), K_predict ∈ [-1, 1] (cosine). This range mismatch is intentional and present in the original — the model learns cosine similarities that approximate WWL values.
- The original training schedule (from train.py): `controller.keep_going('rec')` runs rec_loss until it stops improving, THEN `warmup_done=True` enables freq_loss. This is loss-based early stopping, NOT a fixed epoch count. Do NOT claim the original uses combined mode.
- lambda=0.01 is the paper's training lambda. Scores WILL saturate (s→1). The original's total_sigma with all s≈1 just counts contraction events — discrimination comes entirely from the embedding quality produced by WWL rec_loss.

## DiscModel rec_loss vs original
- DiscModel's current rec_loss uses GeomLoss Sinkhorn Wasserstein on internal-edge 4-hop WL spotlight features — the proper equivalent of the original's WWL on spotlight subgraphs.
- K_true[i,j] = exp(-Wasserstein(WL features of nodes in spotlight_i, WL features of nodes in spotlight_j)), computed via _wwl_gram using GeomLoss SamplesLoss.
- The cosine gram of scatter_mean WL features is NOT a sufficient approximation — it only works for one motif type, not across all types.

## Original freq_loss
- Operates **per-level**: `for t in range(len(pp))`, separate KDTree density at each level, averaged over steps.
- Uses `distance_density` by default: KDTree k-th nearest neighbor **Euclidean distance** (not cosine).
- `s = pp[t]` — per-level edge scores, one per supernode (NOT per original edge).
- Gradient flows ONLY through `s`. Embeddings are fully detached.
- Scoring in the original: `e = score_net(transform(x_u + x_v))` — scored on the merged embedding.

## DiscModel training schedule
- Training uses a **fixed epoch schedule** via `--stop-epochs N --epochs M`.
- `stop_epochs` = rec warmup length. `epoch < stop_epochs` → rec_loss only. `epoch >= stop_epochs` → freq_loss only. Total training = `epochs`.
- This is **hard-coded epoch comparison** in `motif_train`: `in_warmup = epoch < stop_epochs`. The Controller is NOT used for phase transitions — only for checkpoint bookkeeping.
- Do NOT use the Controller's `keep_going()` to gate phases. Do NOT reintroduce `done_training` logic that breaks out of the loop early.
- Typical config: `--stop-epochs 8 --epochs 50` (8 rec warmup + 42 freq). For a quick run: `--stop-epochs 5 --epochs 15 --max-batches 5`.

## build_synth_data.py — data generation only
- `scripts/build_synth_data.py` is a **data generation utility only**. It calls `generate_instances` from `MotiFiesta/utils/synthetic.py` and saves the output. It must NEVER alter, filter, rekey, or reinterpret the triplets returned.
- The triplet keys from `generate_instances` are: `pos = triplet['pos']` (planted graph with motif), `neg = triplet['neg']` (background-only, no motif). Do NOT change which key is used for pos or neg.
- `neg.num_nodes = neg.x.size(0)` — neg has fewer nodes than pos (background-only vs planted). Do not force neg to match pos node count.
- `MotiFiesta/utils/synthetic.py` must NOT be modified. All data generation logic lives there. Results must be reproducible against the main branch.
- The only safe edits to `build_synth_data.py` are the top-level config constants (`MOTIF_TYPE`, `MOTIF_SIZE`, `PARENT_SIZE`, `PARENT_EPROB`, `N_GRAPHS`, `USE_EIGEN_PE`, `N_EIGEN`).

## General
- Do not manage the git repo. The user commits their own changes.
- Lowercase comments in code.
- No meta-commentary in responses. Direct and concise.
