# MotiFiesta Discovery: Subgraph Representation Learning for Unsupervised Motif Discovery

This repository is forked from the MotiFiesta algorithm repository described in the following paper:

>Carlos Oliver, Dexiong Chen, Vincent Mallet, Pericles Philippopoulos, Karsten Borgwardt.
[Approximate Network Motif Mining Via Graph Learning][1]. Preprint 2022.

[![](http://img.shields.io/badge/cs.LG-arXiv%3A2206.01008-B31B1B.svg)][1]

## MotiFiesta

![MotiFiesta](figs/arch_motifiesta.png)

## Discovery Experiments

![rand-esu + WL hash + LDA + simhash](figs/arch_structural_disc.png)

![rand-esu + Prżulj certificate + freq. score](figs/arch_canonical_disc.png)

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
$ python MotiFiesta/disc/structural.py --name barbell-structural --data data/synth-barbell-k10 --dataset synth_pairs
$ python MotiFiesta/disc/canonical.py --name barbell-canonical --data data/synth-barbell-k10 --dataset synth_pairs
```

[1]: https://arxiv.org/abs/2206.01008