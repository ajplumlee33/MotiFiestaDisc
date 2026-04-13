import torch
import torch.nn.functional as f
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import math
import os
import time

from MotiFiesta.utils.stats import RunningStats
from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.graph_utils import to_graphs

class Controller:
    """
    tracks training progress and decides when to switch phases
    """
    def __init__(self, since_best_threshold=30, mode='freq'):
        self.since_best_threshold = since_best_threshold
        # rec is always active; mode determines others
        if mode == 'freq':
            self.modules = ['rec', 'mot']
        elif mode == 'zscore':
            self.modules = ['rec', 'zsc']
        else: # combined
            self.modules = ['rec', 'mot', 'zsc']
            
        self.best_losses = {key: {'best': float('inf'), 'since_best': 0}
                            for key in self.modules}

    def keep_going(self, key):
        return self.best_losses[key]['since_best'] < self.since_best_threshold

    def update(self, losses):
        for key, val in losses.items():
            if key not in self.best_losses:
                continue
            if val < self.best_losses[key]['best']:
                self.best_losses[key]['best'] = val
                self.best_losses[key]['since_best'] = 0
            elif not math.isnan(val) and val != 0:
                self.best_losses[key]['since_best'] += 1

    def state_dict(self):
        return {
            'since_best_threshold': self.since_best_threshold,
            'modules': self.modules,
            'best_losses': self.best_losses
        }

    def set_state(self, state_dict):
        self.since_best_threshold = state_dict['since_best_threshold']
        self.modules = state_dict['modules']
        self.best_losses = state_dict['best_losses']

def sys_train(model,
                train_loader,
                test_loader,
                master_data,
                mode='combined',
                model_name='default',
                estimator='kde',
                epochs=5,
                lam=1,
                beta=1,
                max_batches=-1,
                stop_epochs=30,
                volume=False,
                n_neighbors=30,
                hard_embed=False,
                epoch_start=0,
                optimizer=None,
                controller_state=None
                ):
    
    start_time = time.time()
    device = get_device()
    writer = SummaryWriter(f"logs/{model_name}")
    os.makedirs(f'models/{model_name}', exist_ok=True)

    if controller_state is None:
        controller_obj = Controller(since_best_threshold=stop_epochs, mode=mode)
    else:
        controller_obj = Controller()
        controller_obj.set_state(controller_state)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    stats = RunningStats(momentum=0.95)
    done_training = False

    for epoch in range(epoch_start, epochs):
        if done_training:
            break

        model.train()
        model.to(device)
        train_losses = {'rec': 0.0, 'zsc': 0.0, 'mot': 0.0}
        
        num_train_batches = len(train_loader)
        n_train = max_batches if max_batches > 0 else num_train_batches

        for batch_idx, batch_dict in tqdm(enumerate(train_loader), total=n_train, desc=f"epoch {epoch+1} [train]"):
            if batch_idx >= n_train:
                break

            pos = batch_dict['pos'].to(device)
            neg = batch_dict['neg'].to(device)

            g_pos_ids = pos.n_id if hasattr(pos, 'n_id') else torch.arange(pos.x.size(0), device=device)
            g_neg_ids = neg.n_id if hasattr(neg, 'n_id') else torch.arange(neg.x.size(0), device=device)

            optimizer.zero_grad()
            
            b_pos_vec = pos.batch if pos.batch is not None else torch.zeros(pos.x.size(0), dtype=torch.long, device=device)
            b_neg_vec = neg.batch if neg.batch is not None else torch.zeros(neg.x.size(0), dtype=torch.long, device=device)

            xx_pos, pp_pos, ee_pos, bb_pos, m_info_pos, int_pos = model(pos.x, pos.edge_index, b_pos_vec, n_id=g_pos_ids)
            xx_neg, pp_neg, ee_neg, bb_neg, m_info_neg, int_neg = model(neg.x, neg.edge_index, b_neg_vec, n_id=g_neg_ids)

            loss = torch.tensor(0.0, device=device)
            backward = False
            warmup_done = not controller_obj.keep_going('rec')

            # reconstruction (jointly active as anchor)
            if controller_obj.keep_going('rec') or warmup_done:
                pos.num_graphs = 1
                rec_loss = model.rec_loss(xx_pos, ee_pos, m_info_pos['spotlights'], master_data, to_graphs(pos), b_pos_vec, pos.x, int_pos)
                rec_weight = 1.0 if not warmup_done else 0.2
                loss = loss + (rec_loss * rec_weight)
                train_losses['rec'] += rec_loss.item()
                backward = True

            # motif significance/frequency logic
            if warmup_done:
                # frequency loss
                if mode in ['freq', 'combined'] and controller_obj.keep_going('mot'):
                    mot_loss = model.freq_loss(int_pos, int_neg, pp_pos, steps=model.steps, 
                                               estimator=estimator, volume=volume, k=n_neighbors, 
                                               lam=lam, beta=beta)
                    loss = loss + mot_loss
                    train_losses['mot'] += mot_loss.item()
                    backward = True

                # z-score loss
                if mode in ['zscore', 'combined'] and controller_obj.keep_going('zsc'):
                    pos_sc = torch.stack([p.mean() for p in pp_pos]).mean()
                    neg_sc = torch.stack([p.mean() for p in pp_neg]).mean()
                    stats.push(neg_sc.detach())
                    zsc_loss = model.zsc_loss(pos_sc, neg_sc, stats)
                    loss = loss + (zsc_loss * lam)
                    train_losses['zsc'] += zsc_loss.item()
                    backward = True

            if backward:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                done_training = True
                break

        # validation phase
        model.eval()
        test_losses = {'rec': 0.0, 'zsc': 0.0, 'mot': 0.0}
        n_test = max_batches if max_batches > 0 else len(test_loader)

        with torch.no_grad():
            for batch_idx, batch_dict in tqdm(enumerate(test_loader), total=n_test, desc=f"epoch {epoch+1} [test]"):
                if batch_idx >= n_test:
                    break
                pos = batch_dict['pos'].to(device)
                g_pos_ids = pos.n_id if hasattr(pos, 'n_id') else torch.arange(pos.x.size(0), device=device)
                b_pos_vec = pos.batch if pos.batch is not None else torch.zeros(pos.x.size(0), dtype=torch.long, device=device)
                xx_pos, pp_pos, ee_pos, bb_pos, m_info_pos, int_pos = model(pos.x, pos.edge_index, b_pos_vec, n_id=g_pos_ids)
                
                pos.num_graphs = 1
                rec_val = model.rec_loss(xx_pos, ee_pos, m_info_pos['spotlights'], master_data, to_graphs(pos), b_pos_vec, pos.x, int_pos)
                test_losses['rec'] += rec_val.item()

                warmup_done = not controller_obj.keep_going('rec')
                if warmup_done:
                    neg = batch_dict['neg'].to(device)
                    g_neg_ids = neg.n_id if hasattr(neg, 'n_id') else torch.arange(neg.x.size(0), device=device)
                    b_neg_vec = neg.batch if neg.batch is not None else torch.zeros(neg.x.size(0), dtype=torch.long, device=device)
                    xx_neg, pp_neg, ee_neg, bb_neg, m_info_neg, int_neg = model(neg.x, neg.edge_index, b_neg_vec, n_id=g_neg_ids)
                    
                    if mode in ['freq', 'combined']:
                        test_losses['mot'] += model.freq_loss(int_pos, int_neg, pp_pos, steps=model.steps, estimator=estimator, k=n_neighbors, lam=lam, beta=beta).item()
                    if mode in ['zscore', 'combined']:
                        test_losses['zsc'] += model.zsc_loss(torch.stack([p.mean() for p in pp_pos]).mean(), torch.stack([p.mean() for p in pp_neg]).mean(), stats).item()

        avg_train = {k: v / n_train for k, v in train_losses.items()}
        avg_test = {k: v / n_test for k, v in test_losses.items()}
        controller_obj.update(avg_test)

        print(f"\nepoch: {epoch+1} | mode: {mode}")
        print(f"train -> rec: {avg_train['rec']:.6f} | zsc: {avg_train['zsc']:.6f} | mot: {avg_train['mot']:.6f}")
        print(f"test  -> rec: {avg_test['rec']:.6f} | zsc: {avg_test['zsc']:.6f} | mot: {avg_test['mot']:.6f}\n")

        model.cpu()
        torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'controller_state_dict': controller_obj.state_dict()}, f'models/{model_name}/{model_name}.pth')
        model.to(device)
        for k in controller_obj.modules:
            writer.add_scalar(f"loss/train_{k}", avg_train[k], epoch)
            writer.add_scalar(f"loss/test_{k}", avg_test[k], epoch)

    writer.close()