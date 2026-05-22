import math
import os
import time

import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.training.model import SamplingInvarianceTracker
from MotiFiesta.utils.sys_txt import rewire


class Controller:
    """
    tracks training progress and decides when to switch phases.
    rec is always active; other modules depend on mode.
    """
    def __init__(self, since_best_threshold=30, mode='combined'):
        self.since_best_threshold = since_best_threshold
        self.mode = mode

        if mode == 'mot':
            self.modules = ['rec', 'mot']
        elif mode == 'sil':
            self.modules = ['rec', 'sil']
        else:  # combined
            self.modules = ['rec', 'mot', 'sil']

        self.best_losses = {
            key: {'best_loss': float('nan') if key == 'rec' else float('inf'), 'since_best': 0}
            for key in self.modules
        }

    def keep_going(self, key):
        return self.best_losses[key]['since_best'] <= self.since_best_threshold

    def update(self, losses):
        for key, val in losses.items():
            if key not in self.best_losses:
                continue
            current_best = self.best_losses[key]['best_loss']
            if val < current_best:
                self.best_losses[key]['best_loss'] = val
                self.best_losses[key]['since_best'] = 0
            elif not math.isnan(val):
                self.best_losses[key]['since_best'] += 1
            else:
                pass

    def state_dict(self):
        return {'since_best_threshold': self.since_best_threshold,
                'mode': self.mode,
                'modules': self.modules,
                'best_losses': self.best_losses
                }

    def set_state(self, state_dict):
        self.since_best_threshold = state_dict['since_best_threshold']
        self.mode = state_dict.get('mode', 'combined')
        self.modules = state_dict['modules']
        self.best_losses = state_dict['best_losses']


def _get_batch_vec(data, device):
    if hasattr(data, 'batch') and data.batch is not None:
        return data.batch.to(device)
    return torch.zeros(data.x.size(0), dtype=torch.long, device=device)


def _get_n_id(data, device):
    if hasattr(data, 'n_id') and data.n_id is not None:
        return data.n_id.to(device)
    return torch.arange(data.x.size(0), device=device)


def _forward(model, data, device):
    b_vec = _get_batch_vec(data, device)
    n_id = _get_n_id(data, device)
    return model(data.x, data.edge_index, b_vec, n_id=n_id)


def _make_neg(pos):
    """
    produce a rewired null from a positive batch via classical double edge swap.
    preserves the degree sequence. n_iter scales with undirected edge count,
    capped at 5000 to prevent O(E²) blowup on dense batches from large graphs.
    """
    device = pos.x.device
    n_edges_undirected = pos.edge_index.size(1) // 2
    n_iter = max(100, min(n_edges_undirected * 2, 5000))
    neg = rewire(pos.cpu(), n_iter=n_iter)
    return neg.to(device)


def sys_train(model,
                train_loader,
                test_loader,
                source_graph,
                mode='combined',
                sil_momentum=0.95,
                model_name='default',
                estimator='knn',
                epochs=200,
                lam=1,
                beta=1,
                max_batches=-1,
                stop_epochs=30,
                volume=False,
                n_neighbors=30,
                hard_embed=False,
                epoch_start=0,
                optimizer=None,
                controller_state=None,
                rec_kernel='wwl',
                edge_sample_rate=1.0,
                ):
    """sys_train.

    :param model: MotiFiesta model
    :param train_loader: loader yielding {'pos'} neighborhood batches
    :param test_loader: same format as train_loader
    :param source_graph: dataset providing the full graph for rec_loss subgraph lookup
    :param mode: which motif losses to activate ('mot', 'sil', 'combined')
    :param sil_momentum: ema momentum for the sampling invariance tracker
    :param model_name: ID to save model under
    :param epochs: number of epochs to train
    :param lam: loss coefficient for edge scores
    :param beta: bandwidth for freq loss exponential
    :param max_batches: if not -1, stop after given number of batches
    :param rec_kernel: 'wwl' (original) or 'wl' (vectorized wl subtree kernel)
    :param edge_sample_rate: fraction of (i,j) entries used for rec supervision
    """
    start_time = time.time()
    n_feat = source_graph.num_features if hasattr(source_graph, 'num_features') else None
    device = get_device(n_features=n_feat)

    # fix the rng trajectory so runs are reproducible
    torch.manual_seed(0)

    os.makedirs(f'models/{model_name}', exist_ok=True)
    writer = SummaryWriter(f"logs/{model_name}")
    log_file = open(f"logs/{model_name}/train.log", "a")

    if controller_state is None:
        controller = Controller(since_best_threshold=stop_epochs, mode=mode)
    else:
        controller = Controller(mode=mode)
        controller.set_state(controller_state)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    sil_tracker = SamplingInvarianceTracker()

    rec_loss, mot_loss, sil_loss = [torch.tensor(float('nan'))] * 3
    done_training = False

    for epoch in range(epoch_start, epochs):
        if done_training:
            print("DONE TRAINING")
            break

        model.train()
        model.to(device)

        num_batches = len(train_loader)
        n_train = max_batches if max_batches > 0 else num_batches

        rec_loss_tot, mot_loss_tot, sil_loss_tot = [0] * 3

        for batch_idx, batch in tqdm(enumerate(train_loader), total=n_train):
            if batch_idx >= max_batches and max_batches > 0:
                break

            pos = batch['pos'].to(device)

            optimizer.zero_grad()

            xx_pos, pp_pos, ee_pos, _, merge_info_pos, internals_pos = _forward(model, pos, device)

            loss = 0
            backward = False
            warmup_done = False

            if controller.keep_going('rec') and not hard_embed:
                if rec_kernel == 'wl':
                    rec_loss = model.rec_loss_wl(xx_pos,
                                                 ee_pos,
                                                 merge_info_pos,
                                                 source_graph,
                                                 internals_pos,
                                                 edge_sample_rate=edge_sample_rate,
                                                 )
                else:
                    rec_loss = model.rec_loss(xx_pos,
                                              ee_pos,
                                              merge_info_pos,
                                              source_graph,
                                              internals_pos,
                                              draw=False
                                              )
                rec_loss_tot += rec_loss.item()
                backward = True
                loss += rec_loss
            else:
                warmup_done = True

            if warmup_done:
                if mode in ('mot', 'combined') and controller.keep_going('mot'):
                    neg = _make_neg(pos)
                    with torch.no_grad():
                        xx_neg, pp_neg, ee_neg, _, merge_info_neg, internals_neg = _forward(model, neg, device)

                    mot_loss = model.freq_loss(internals_pos,
                                               internals_neg,
                                               pp_pos,
                                               steps=model.steps,
                                               estimator=estimator,
                                               volume=volume,
                                               k=n_neighbors,
                                               lam=lam,
                                               beta=beta
                                               )
                    loss += mot_loss
                    mot_loss_tot += mot_loss.item()
                    backward = True

                if mode in ('sil', 'combined') and controller.keep_going('mot') and controller.keep_going('sil'):
                    sil_loss = model.sil_loss(internals_pos,
                                              merge_info_pos,
                                              sil_tracker,
                                              momentum=sil_momentum)
                    loss += sil_loss * lam
                    sil_loss_tot += sil_loss.item()
                    backward = True

            if backward:
                loss.backward()
                optimizer.step()
            else:
                done_training = True

        N = max_batches if max_batches > 0 else len(train_loader)

        losses = {'rec': rec_loss_tot / N,
                  'mot': mot_loss_tot / N,
                  'sil': sil_loss_tot / N,
                  }

        ## END OF BATCHES ##
        rec_loss_tot, mot_loss_tot, sil_loss_tot = [0] * 3

        model.eval()

        n_test = max_batches if max_batches > 0 else len(test_loader)

        for batch_idx, batch in tqdm(enumerate(test_loader), total=n_test):
            if batch_idx >= max_batches and max_batches > 0:
                break

            pos = batch['pos'].to(device)

            with torch.no_grad():
                xx_pos, pp_pos, ee_pos, _, merge_info_pos, internals_pos = _forward(model, pos, device)

            warmup_done = False

            rec_loss = torch.tensor(float('nan'))
            mot_loss = torch.tensor(float('nan'))
            sil_loss = torch.tensor(float('nan'))

            if controller.keep_going('rec'):
                if rec_kernel == 'wl':
                    rec_loss = model.rec_loss_wl(xx_pos,
                                                 ee_pos,
                                                 merge_info_pos,
                                                 source_graph,
                                                 internals_pos,
                                                 edge_sample_rate=edge_sample_rate,
                                                 )
                else:
                    rec_loss = model.rec_loss(xx_pos,
                                            ee_pos,
                                            merge_info_pos,
                                            source_graph,
                                            internals_pos
                                            )

            else:
                warmup_done = True

            if warmup_done:
                if mode in ('mot', 'combined') and controller.keep_going('mot'):
                    neg = _make_neg(pos)
                    with torch.no_grad():
                        xx_neg, pp_neg, ee_neg, _, merge_info_neg, internals_neg = _forward(model, neg, device)

                    mot_loss = model.freq_loss(internals_pos,
                                               internals_neg,
                                               pp_pos,
                                               steps=model.steps,
                                               estimator=estimator,
                                               volume=volume,
                                               k=n_neighbors,
                                               lam=lam,
                                               beta=beta
                                               )

                if mode in ('sil', 'combined') and controller.keep_going('mot') and controller.keep_going('sil'):
                    sil_loss = model.sil_loss(internals_pos,
                                              merge_info_pos,
                                              sil_tracker,
                                              momentum=sil_momentum)

            rec_loss_tot += rec_loss.item()
            mot_loss_tot += mot_loss.item()
            sil_loss_tot += sil_loss.item()

        N = max_batches if max_batches > 0 else len(test_loader)

        test_losses = {'rec': rec_loss_tot / N,
                       'mot': mot_loss_tot / N,
                       'sil': sil_loss_tot / N,
                       }

        controller.update(test_losses)

        model.cpu()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'controller_state_dict': controller.state_dict()
        }, f'models/{model_name}/{model_name}.pth')

        loss_str = ' '.join([f'{k} train: {v:2f}' for k, v in losses.items()])
        test_loss_str = ' '.join([f'{k} test: {v:2f}' for k, v in test_losses.items()])
        model.to(device)
        time_elapsed = time.time() - start_time
        msg = (f"Train Epoch: {epoch+1} [{batch_idx+1}/{num_batches}]"
               f"({100. * (batch_idx+1) / num_batches:.2f}%) {loss_str}"
               f" {test_loss_str}"
               f" Time: {time_elapsed:.2f}")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        step = epoch * num_batches + batch_idx
        for k, v in losses.items():
            writer.add_scalar(f"train_{k}", v, step)
        for k, v in test_losses.items():
            writer.add_scalar(f"test_{k}", v, step)

    model.cpu()
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'controller_state_dict': controller.state_dict()
    }, f'models/{model_name}/{model_name}.pth')

    writer.close()
    log_file.close()
