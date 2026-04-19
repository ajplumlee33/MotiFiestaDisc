import math
import os
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.stats import RunningStats
from MotiFiesta.utils.sys_loader import fast_vectorized_swap


class Controller:
    """
    tracks training progress and decides when to switch phases
    """
    def __init__(self, since_best_threshold=30, mode='freq'):
        self.since_best_threshold = since_best_threshold
        self.mode = mode

        # rec is always active; mode determines others
        if mode == 'freq':
            self.modules = ['rec', 'mot']
        elif mode == 'zscore':
            self.modules = ['rec', 'zsc']
        else: # combined
            self.modules = ['rec', 'mot', 'zsc']

        self.best_losses = {key: {'best_loss': float('nan'), 'since_best': 0}
                            for key in self.modules}

    def keep_going(self, key):
        """ Returns True if model should keep training, false otherwise."""
        return self.best_losses[key]['since_best'] <= self.since_best_threshold

    def update(self, losses):
        for key, val in losses.items():
            if key not in self.best_losses:
                continue
            current_best = self.best_losses[key]['best_loss']
            if math.isnan(current_best) or val < current_best:
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
        self.mode = state_dict.get('mode', 'freq')
        self.modules = state_dict['modules']
        self.best_losses = state_dict['best_losses']


def _get_batch_vec(data, device):
    """ fall back to zeros when the loader yields a single subgraph """
    if hasattr(data, 'batch') and data.batch is not None:
        return data.batch.to(device)
    return torch.zeros(data.x.size(0), dtype=torch.long, device=device)


def _get_n_id(data, device):
    """ global node mapping from neighborloader """
    if hasattr(data, 'n_id') and data.n_id is not None:
        return data.n_id.to(device)
    return torch.arange(data.x.size(0), device=device)


def _forward(model, data, device):
    b_vec = _get_batch_vec(data, device)
    n_id = _get_n_id(data, device)
    return model(data.x, data.edge_index, b_vec, n_id=n_id)


def _pp_mean(pp):
    """ mean edge score across all contraction levels """
    return torch.stack([p.mean() for p in pp]).mean()


def _make_neg(pos):
    """ produce a single rewired null from a positive batch """
    neg = pos.clone()
    neg.edge_index = fast_vectorized_swap(pos.edge_index)
    return neg


def sys_train(model,
                train_loader,
                test_loader,
                source_graph,
                mode='combined',
                zsc_method='ema',
                zsc_k=10,
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
                controller_state=None
                ):
    """sys_train.

    :param model: MotiFiesta model
    :param train_loader: loader yielding {'pos'} neighborhood batches
    :param test_loader: same format as train_loader
    :param source_graph: dataset providing the full graph for rec_loss subgraph lookup
    :param mode: which motif losses to activate ('freq', 'zscore', 'combined')
    :param zsc_method: 'ema' (global streaming null) or 'local' (per-sample null from k rewires)
    :param zsc_k: number of null realisations per pos batch when zsc_method is 'local'
    :param model_name: ID to save model under
    :param epochs: number of epochs to train
    :param lam: loss coefficient for edge scores
    :param beta: bandwidth for freq loss exponential
    :param max_batches: if not -1, stop after given number of batches
    """
    start_time = time.time()
    device = get_device()

    os.makedirs(f'models/{model_name}', exist_ok=True)
    writer = SummaryWriter(f"logs/{model_name}")

    if controller_state is None:
        controller = Controller(since_best_threshold=stop_epochs, mode=mode)
    else:
        controller = Controller(mode=mode)
        controller.set_state(controller_state)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    # running statistics of the null distribution for the ema z-score
    stats = RunningStats(momentum=0.95)

    rec_loss, mot_loss, zsc_loss = [torch.tensor(float('nan'))] * 3
    done_training = False

    for epoch in range(epoch_start, epochs):
        if done_training:
            print("DONE TRAINING")
            break

        model.train()
        model.to(device)

        num_batches = len(train_loader)
        n_train = max_batches if max_batches > 0 else num_batches

        rec_loss_tot, mot_loss_tot, zsc_loss_tot = [0] * 3

        for batch_idx, batch in tqdm(enumerate(train_loader), total=n_train):
            if batch_idx >= max_batches and max_batches > 0:
                break

            pos = batch['pos'].to(device)

            optimizer.zero_grad()

            # do main forward pass
            xx_pos, pp_pos, ee_pos, _, merge_info_pos, internals_pos = _forward(model, pos, device)

            loss = 0

            backward = False
            warmup_done = False

            if controller.keep_going('rec') and not hard_embed:
                rec_loss = model.rec_loss(xx_pos,
                                          ee_pos,
                                          merge_info_pos['spotlights'],
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
                # primary null used by freq_loss and the ema z-score path
                neg = _make_neg(pos)
                xx_neg, pp_neg, ee_neg, _, merge_info_neg, internals_neg = _forward(model, neg, device)

                # penalises embeddings whose neighborhood density exceeds the null
                if mode in ('freq', 'combined') and controller.keep_going('mot'):
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

                # rewards embeddings that are significantly over-represented vs the null
                if mode in ('zscore', 'combined') and controller.keep_going('zsc'):

                    if zsc_method == 'ema':
                        # accumulate null scores in the streaming stats
                        neg_sc = _pp_mean(pp_neg)
                        stats.push(neg_sc.detach())

                        # skip until the running stats have enough samples for a valid sigma
                        if stats.count > 1:
                            pos_sc = _pp_mean(pp_pos)
                            zsc_loss = model.zsc_loss_ema(pos_sc, neg_sc, stats)
                            loss += zsc_loss * lam
                            zsc_loss_tot += zsc_loss.item()
                            backward = True

                    elif zsc_method == 'local':
                        # reuse the freq null as the first sample to save a forward pass
                        null_scores = [_pp_mean(pp_neg)]
                        for _ in range(zsc_k - 1):
                            neg_extra = _make_neg(pos)
                            _, pp_neg_extra, _, _, _, _ = _forward(model, neg_extra, device)
                            null_scores.append(_pp_mean(pp_neg_extra))
                        null_scores = torch.stack(null_scores)

                        pos_sc = _pp_mean(pp_pos)
                        zsc_loss = model.zsc_loss_local(pos_sc, null_scores)
                        loss += zsc_loss * lam
                        zsc_loss_tot += zsc_loss.item()
                        backward = True

            if backward:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                done_training = True

        N = max_batches if max_batches > 0 else len(train_loader)

        losses = {'rec': rec_loss_tot / N,
                  'mot': mot_loss_tot / N,
                  'zsc': zsc_loss_tot / N,
                  }

        ## END OF BATCHES ##
        rec_loss_tot, mot_loss_tot, zsc_loss_tot = [0] * 3

        model.eval()

        n_test = max_batches if max_batches > 0 else len(test_loader)

        for batch_idx, batch in tqdm(enumerate(test_loader), total=n_test):
            if batch_idx >= max_batches and max_batches > 0:
                break

            pos = batch['pos'].to(device)

            with torch.no_grad():
                # do main forward pass
                xx_pos, pp_pos, ee_pos, _, merge_info_pos, internals_pos = _forward(model, pos, device)

            warmup_done = False

            rec_loss = torch.tensor(float('nan'))

            if controller.keep_going('rec'):
                rec_loss = model.rec_loss(xx_pos,
                                        ee_pos,
                                        merge_info_pos['spotlights'],
                                        source_graph,
                                        internals_pos
                                        )
            else:
                warmup_done = True

            mot_loss = torch.tensor(float('nan'))
            zsc_loss = torch.tensor(float('nan'))

            if warmup_done:
                neg = _make_neg(pos)

                with torch.no_grad():
                    xx_neg, pp_neg, ee_neg, _, merge_info_neg, internals_neg = _forward(model, neg, device)

                if mode in ('freq', 'combined'):
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

                if mode in ('zscore', 'combined'):
                    if zsc_method == 'ema' and stats.count > 1:
                        pos_sc = _pp_mean(pp_pos)
                        neg_sc = _pp_mean(pp_neg)
                        zsc_loss = model.zsc_loss_ema(pos_sc, neg_sc, stats)

                    elif zsc_method == 'local':
                        null_scores = [_pp_mean(pp_neg)]
                        for _ in range(zsc_k - 1):
                            neg_extra = _make_neg(pos)
                            with torch.no_grad():
                                _, pp_neg_extra, _, _, _, _ = _forward(model, neg_extra, device)
                            null_scores.append(_pp_mean(pp_neg_extra))
                        null_scores = torch.stack(null_scores)

                        pos_sc = _pp_mean(pp_pos)
                        zsc_loss = model.zsc_loss_local(pos_sc, null_scores)

            rec_loss_tot += rec_loss.item()
            mot_loss_tot += mot_loss.item()
            zsc_loss_tot += zsc_loss.item()

        N = max_batches if max_batches > 0 else len(test_loader)

        test_losses = {'rec': rec_loss_tot / N,
                       'mot': mot_loss_tot / N,
                       'zsc': zsc_loss_tot / N,
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
        print(f"Train Epoch: {epoch+1} [{batch_idx +1}/{num_batches}]"\
              f"({100. * (batch_idx +1) / num_batches :.2f}%) {loss_str}"\
              f" {test_loss_str}"\
              f" Time: {time_elapsed:.2f}"
              )

        # tensorboard logging
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
