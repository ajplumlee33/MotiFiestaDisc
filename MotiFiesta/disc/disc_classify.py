"""
graph classification eval for MotiFiestaDisc on TUDataset benchmarks.
10-fold stratified CV with Random Forest on pooled subgraph embeddings.

run from repo root:
    python MotiFiesta/disc/disc_classify.py --name proteins_ego1 --dataset PROTEINS
"""
import sys
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from MotiFiesta.disc.disc_model import MotiFiestaDisc
from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.real_world import TUDataset


class AccuracyEvaluator:
    def eval(self, input_dict):
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        return {"acc": float((y_true == y_pred).mean())}


def load_disc(name):
    ckpt = torch.load(f'models/{name}/{name}_best.pth',
                      map_location='cpu', weights_only=False)
    with open(f'models/{name}/hparams.json') as f:
        hp = json.load(f)['model']
    wl_raw = hp.get('walk_lens', [1, 2, 3])
    walk_lens = [int(x) for x in wl_raw.split(',')] if isinstance(wl_raw, str) else (
        wl_raw if isinstance(wl_raw, list) else [wl_raw])
    disc = MotiFiestaDisc(
        n_features = hp['n_features'],
        hidden_dim = hp.get('hidden_dim', 32),
        gin_layers = hp.get('gin_layers', 2),
        walk_lens  = walk_lens,
        wl_hops    = hp.get('wl_hops', 1),
        pool       = hp.get('pool', 'mean'),
    )
    disc.load_state_dict(ckpt['model_state_dict'], strict=False)
    for p in disc.parameters():
        p.requires_grad = False
    return disc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',          required=True)
    parser.add_argument('--dataset',       default='PROTEINS')
    parser.add_argument('--batch-size',    type=int, default=32)
    parser.add_argument('--n-folds',       type=int, default=10)
    parser.add_argument('--n-estimators',  type=int, default=500)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = get_device()

    disc = load_disc(args.name)
    disc.to(device).eval()

    dataset = TUDataset(root='data', name=args.dataset)
    labels  = [dataset[i].y.item() for i in range(len(dataset))]

    print("precomputing graph embeddings...")
    full_loader = DataLoader(list(dataset), batch_size=args.batch_size, shuffle=False)
    all_embs, all_ys = [], []
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="embed"):
            batch = batch.to(device)
            levels = disc(batch.x.float(), batch.edge_index, batch.batch)
            lvl_embs = []
            for lvl in levels:
                lvl_embs.append(global_add_pool(lvl['z_sub'], lvl['sub_batch']))
            all_embs.append(torch.cat(lvl_embs, dim=-1).cpu())
            all_ys.append(batch.y.view(-1).cpu())

    embs   = torch.cat(all_embs, dim=0).numpy()
    ys     = torch.cat(all_ys,   dim=0).numpy()

    evaluator = AccuracyEvaluator()
    cv        = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(range(len(dataset)), labels)):
        clf = RandomForestClassifier()
        clf.fit(embs[train_idx], ys[train_idx])
        pred   = clf.predict(embs[test_idx]).reshape(-1, 1)
        y_np   = ys[test_idx].reshape(-1, 1)
        acc    = evaluator.eval({"y_true": y_np, "y_pred": pred})['acc']
        fold_accs.append(acc)
        print(f"fold {fold+1}/{args.n_folds}: {acc:.4f}")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n{args.dataset}  {args.n_folds}-fold acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"baseline (MotiFiesta): 73.1 ± 2.0")


if __name__ == '__main__':
    main()
