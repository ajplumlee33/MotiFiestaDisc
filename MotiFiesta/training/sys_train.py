import torch
import torch.nn.functional as F
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
    def __init__(self, since_best_threshold=30):
        self.since_best_threshold = since_best_threshold
        self.modules = ['rec', 'zsc']
        self.best_losses = {key: {'best': float('inf'), 'since_best': 0}
                            for key in self.modules}

    def keep_going(self, key):
        return self.best_losses[key]['since_best'] < self.since_best_threshold

    def update(self, losses):
        for key, val in losses.items():
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
        controller = Controller(since_best_threshold=stop_epochs)
    else:
        controller = Controller()
        controller.set_state(controller_state)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    stats = RunningStats(momentum=0.95)
    done_training = False

    for epoch in range(epoch_start, epochs):
        if done_training:
            break

        model.train()
        model.to(device)
        train_losses = {'rec': 0.0, 'zsc': 0.0}
        
        num_train_batches = len(train_loader)
        n_train = max_batches if max_batches > 0 else num_train_batches

        for batch_idx, batch_dict in tqdm(enumerate(train_loader), total=n_train, desc=f"epoch {epoch+1} [train]"):
            if batch_idx >= n_train:
                break

            pos = batch_dict['pos'].to(device)
            neg = batch_dict['neg'].to(device)

            global_pos_ids = pos.n_id if hasattr(pos, 'n_id') else torch.arange(pos.x.size(0), device=device)
            global_neg_ids = neg.n_id if hasattr(neg, 'n_id') else torch.arange(neg.x.size(0), device=device)

            optimizer.zero_grad()
            
            batch_pos_vec = pos.batch if pos.batch is not None else torch.zeros(pos.x.size(0), dtype=torch.long, device=device)
            batch_neg_vec = neg.batch if neg.batch is not None else torch.zeros(neg.x.size(0), dtype=torch.long, device=device)

            xx_pos, pp_pos, ee_pos, bb_pos, m_info_pos, int_pos = model(pos.x, pos.edge_index, batch_pos_vec, n_id=global_pos_ids)
            xx_neg, pp_neg, ee_neg, bb_neg, m_info_neg, int_neg = model(neg.x, neg.edge_index, batch_neg_vec, n_id=global_neg_ids)

            loss = torch.tensor(0.0, device=device)
            backward = False
            
            # check phase transition
            warmup_done = not controller.keep_going('rec')

            # reconstruction (jointly active as anchor)
            if controller.keep_going('rec') or warmup_done:
                pos.num_graphs = 1
                graphs_pos = to_graphs(pos) 
                
                rec_loss = model.rec_loss(
                    xx=xx_pos,
                    ee=ee_pos,
                    spotlights=m_info_pos['spotlights'],
                    master_data=master_data,
                    graphs=graphs_pos,
                    batch=batch_pos_vec,
                    node_feats=pos.x,
                    internals=int_pos
                )
                
                # weight reduction during significance phase to prevent drift
                rec_weight = 1.0 if not warmup_done else 0.2
                loss = loss + (rec_loss * rec_weight)
                train_losses['rec'] += rec_loss.item()
                backward = True

            # significance (active after warmup)
            if warmup_done and controller.keep_going('zsc'):
                pos_scores = pp_pos[0]
                neg_scores = pp_neg[0]
                
                stats.push(neg_scores.detach().mean())
                zsc_loss = model.zsc_loss(pos_scores, neg_scores, stats)
                
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
        test_losses = {'rec': 0.0, 'zsc': 0.0}
        n_test = max_batches if max_batches > 0 else len(test_loader)

        with torch.no_grad():
            for batch_idx, batch_dict in tqdm(enumerate(test_loader), total=n_test, desc=f"epoch {epoch+1} [test]"):
                if batch_idx >= n_test:
                    break

                pos = batch_dict['pos'].to(device)
                
                # pos pass
                g_pos_ids = pos.n_id if hasattr(pos, 'n_id') else torch.arange(pos.x.size(0), device=device)
                b_pos_vec = pos.batch if pos.batch is not None else torch.zeros(pos.x.size(0), dtype=torch.long, device=device)
                
                xx_pos, pp_pos, ee_pos, bb_pos, m_info_pos, int_pos = model(pos.x, pos.edge_index, b_pos_vec, n_id=g_pos_ids)
                
                # rec loss
                pos.num_graphs = 1
                graphs_pos = to_graphs(pos)
                rec_val = model.rec_loss(xx_pos, ee_pos, m_info_pos['spotlights'], master_data, graphs_pos, b_pos_vec, pos.x, int_pos)
                test_losses['rec'] += rec_val.item()

                # zsc loss
                warmup_done = not controller.keep_going('rec')
                if warmup_done:
                    neg = batch_dict['neg'].to(device)
                    # use neg's own n_id to avoid key mismatches
                    g_neg_ids = neg.n_id if hasattr(neg, 'n_id') else torch.arange(neg.x.size(0), device=device)
                    b_neg_vec = neg.batch if neg.batch is not None else torch.zeros(neg.x.size(0), dtype=torch.long, device=device)
                    
                    xx_neg, pp_neg, ee_neg, bb_neg, m_info_neg, int_neg = model(neg.x, neg.edge_index, b_neg_vec, n_id=g_neg_ids)
                    
                    zsc_val = model.zsc_loss(pp_pos[0], pp_neg[0], stats)
                    test_losses['zsc'] += zsc_val.item()

        avg_train = {k: v / n_train for k, v in train_losses.items()}
        avg_test = {k: v / n_test for k, v in test_losses.items()}
        
        controller.update(avg_test)

        current_phase = "joint (significance + rec)" if not controller.keep_going('rec') else "warmup (rec only)"

        print(f"\n{'='*40}")
        print(f"epoch: {epoch+1} | phase: {current_phase}")
        print(f"train -> rec: {avg_train['rec']:.6f} | zsc: {avg_train['zsc']:.6f}")
        print(f"test  -> rec: {avg_test['rec']:.6f} | zsc: {avg_test['zsc']:.6f}")
        print(f"{'='*40}\n")

        model.cpu()
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'controller_state_dict': controller.state_dict(),
            'stats_state_dict': stats.state_dict() if hasattr(stats, 'state_dict') else None
        }
        torch.save(checkpoint, f'models/{model_name}/{model_name}.pth')
        model.to(device)
        
        writer.add_scalar("loss/train_rec", avg_train['rec'], epoch)
        writer.add_scalar("loss/test_rec", avg_test['rec'], epoch)
        writer.add_scalar("loss/train_zsc", avg_train['zsc'], epoch)
        writer.add_scalar("loss/test_zsc", avg_test['zsc'], epoch)

    model.cpu()
    torch.save(checkpoint, f'models/{model_name}/{model_name}.pth')
    writer.close()
    print(f"training finished. final model saved to models/{model_name}/{model_name}.pth")