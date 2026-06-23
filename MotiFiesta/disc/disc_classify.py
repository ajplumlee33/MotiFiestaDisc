"""
graph classification eval for MotiFiestaDisc on TUDataset benchmarks.
uses the same evaluate() function as table2.py, 10-fold stratified CV.

run from repo root:
    python fig_scripts/disc_classify.py --name proteins-disc --dataset PROTEINS
"""
import sys
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from MotiFiesta.disc.disc_model import MotiFiestaDisc
from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.real_world import TUDataset


# same evaluate() as table2.py
def evaluate(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


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
    wl_raw = hp.get('walk_lens', hp.get('walk_len', 8))
    walk_lens = [int(x) for x in wl_raw.split(',')] if isinstance(wl_raw, str) else (
        wl_raw if isinstance(wl_raw, list) else [wl_raw])
    disc = MotiFiestaDisc(
        n_features  = hp['n_features'],
        hidden_dim  = hp.get('hidden_dim', 32),
        gin_layers  = hp.get('gin_layers', 2),
        rwse_steps  = hp.get('rwse_steps', 8),
        walk_lens   = walk_lens,
        n_walks     = hp.get('n_walks', 4),
        wl_hops     = hp.get('wl_hops', 1),
    )
    disc.load_state_dict(ckpt['model_state_dict'])
    for p in disc.parameters():
        p.requires_grad = False
    return disc, hp.get('hidden_dim', 32), len(walk_lens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',       required=True)
    parser.add_argument('--dataset',    default='PROTEINS')
    parser.add_argument('--batch-size',    type=int, default=32)
    parser.add_argument('--n-folds',       type=int, default=10)
    parser.add_argument('--n-estimators',  type=int, default=500)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = get_device()

    disc, _, _ = load_disc(args.name)
    disc.to(device).eval()

    dataset = TUDataset(root='data', name=args.dataset)
    labels  = [dataset[i].y.item() for i in range(len(dataset))]

    # precompute embeddings once — avoids re-running disc encoder per fold/epoch
    print("precomputing graph embeddings...")
    full_loader = DataLoader(list(dataset), batch_size=args.batch_size, shuffle=False)
    all_embs, all_ys = [], []
    disc.eval()
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="embed"):
            batch = batch.to(device)
            _, _, _, _, _, internals = disc(
                batch.x.float(), batch.edge_index, batch.batch
            )
            lvl_embs = []
            for lvl in internals:
                lvl_embs.append(global_mean_pool(lvl['z_sub'], lvl['sub_batch']))
            all_embs.append(torch.cat(lvl_embs, dim=-1).cpu())
            all_ys.append(batch.y.view(-1).cpu())
    embs = torch.cat(all_embs, dim=0)   # [N, hidden_dim * n_levels]
    ys   = torch.cat(all_ys,   dim=0)   # [N]

    evaluator = AccuracyEvaluator()
    cv        = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_accs = []

    embs_np = embs.numpy()
    ys_np   = ys.numpy()

    for fold, (train_idx, test_idx) in enumerate(cv.split(range(len(dataset)), labels)):
        clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
        clf.fit(embs_np[train_idx], ys_np[train_idx])
        pred = clf.predict(embs_np[test_idx]).reshape(-1, 1)
        y_np = ys_np[test_idx].reshape(-1, 1)
        result = evaluator.eval({"y_true": y_np, "y_pred": pred})
        acc = result['acc']
        fold_accs.append(acc)
        print(f"fold {fold+1}/{args.n_folds}: {acc:.4f}")

    print(f"\n{args.dataset}  {args.n_folds}-fold acc: "
          f"{np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")


if __name__ == '__main__':
    main()
