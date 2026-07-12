# MotiFiesta Discovery: Subgraph Representation Learning for Unsupervised Motif Discovery

This repository is forked from the MotiFiesta algorithm repository described in the following paper:

>Carlos Oliver, Dexiong Chen, Vincent Mallet, Pericles Philippopoulos, Karsten Borgwardt.
[Approximate Network Motif Mining Via Graph Learning][1]. Preprint 2022.

[![](http://img.shields.io/badge/cs.LG-arXiv%3A2206.01008-B31B1B.svg)][1]

## MotiFiesta

![MotiFiesta](figs/motifiesta.png)

## Discovery Experiments

### WL

![rand-esu + WL hash + LDA + simhash](figs/wl_disc.png)

### Structural

![rand-esu + structural features + decision tree](figs/structural_disc.png)

### Canonical

![rand-esu + Prżulj certificate + freq. score](figs/canonical_disc.png)

## Setup

```
$ pip install . 
```

## Build dataset

```
$ python MotiFiesta/disc/gen_synth.py barbell
```

## Run discovery

```
$ python MotiFiesta/disc/run.py --name barbell-wl --embed-mode wl --data data/synth-barbell-k10 --dataset synth_pairs
$ python MotiFiesta/disc/run.py --name barbell-canonical --embed-mode canonical --data data/synth-barbell-k10 --dataset synth_pairs
```

[1]: https://arxiv.org/abs/2206.01008