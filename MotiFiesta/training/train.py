import time
import math

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from MotiFiesta.utils.learning_utils import get_device


def _make_batch_source(batch):
    """wrap a pyg batch as a source object for rec_loss_wl."""
    class _Src:
        pass
    src = _Src()
    src.cached_data = batch
    return src


def compute_wl_cross_weights(batch_pos, batch_neg, ee_pos, merge_info_pos,
                              wl_iter=3, max_spotlight_nodes=20):
    """per-edge WL cross-similarity weights for freq_loss.

    for each edge spotlight in pos at each pooling level, extracts the induced
    subgraph from neg at the same node indices and computes WL similarity.
    weight = 1 - similarity: high for structurally distinct pairs (motif vs er),
    low for similar pairs (background louvain vs er).

    returns a list of float32 tensors (one per level), shape [n_edges_at_level].
    """
    from MotiFiesta.training.wl_kernel import (
        wl_subtree_similarity_batch,
        initial_labels_from_onehot,
    )
    from torch_geometric.utils import subgraph as pyg_subgraph

    device = get_device()
    pos_edge_index = batch_pos.edge_index.to(device)
    neg_edge_index = batch_neg.edge_index.to(device)
    n_nodes = batch_pos.num_nodes

    pos_labels = initial_labels_from_onehot(batch_pos.x.to(device))
    neg_labels = initial_labels_from_onehot(batch_neg.x.to(device))

    spot_assign = merge_info_pos['spotlight_assignment']
    n_id = merge_info_pos['n_id'].to(device)

    weights = []
    for level in range(len(ee_pos)):
        edge_idx = ee_pos[level].to(device)
        spot_t = spot_assign[level].to(device)
        n_take = edge_idx.size(1)

        pos_edges, neg_edges, counts, pos_lbls, neg_lbls, valid = [], [], [], [], [], []

        for i in range(n_take):
            u = edge_idx[0, i].item()
            v = edge_idx[1, i].item()
            mask = (spot_t == u) | (spot_t == v)
            local = mask.nonzero(as_tuple=False).squeeze(-1)
            if local.numel() == 0:
                continue
            global_idx = n_id[local].sort().values
            if global_idx.size(0) > max_spotlight_nodes:
                perm = torch.randperm(global_idx.size(0))[:max_spotlight_nodes]
                global_idx = global_idx[perm].sort().values

            n = global_idx.size(0)
            pos_ei, _ = pyg_subgraph(global_idx, pos_edge_index,
                                     relabel_nodes=True, num_nodes=n_nodes)
            neg_ei, _ = pyg_subgraph(global_idx, neg_edge_index,
                                     relabel_nodes=True, num_nodes=n_nodes)
            pos_edges.append(pos_ei)
            neg_edges.append(neg_ei)
            counts.append(n)
            pos_lbls.append(pos_labels[global_idx])
            neg_lbls.append(neg_labels[global_idx])
            valid.append(i)

        w = torch.ones(n_take, device=device)
        if len(valid) >= 1:
            # combine pos and neg spotlights into one joint kernel call
            all_edges = pos_edges + neg_edges
            all_counts = counts + counts
            all_labels = pos_lbls + neg_lbls
            K = wl_subtree_similarity_batch(all_edges, all_counts, all_labels,
                                            n_iter=wl_iter)
            n_valid = len(valid)
            # cross-diagonal: K[i, n_valid + i] = sim(pos_i, neg_i)
            cross_sim = torch.stack([K[i, n_valid + i] for i in range(n_valid)])
            for idx, edge_i in enumerate(valid):
                w[edge_i] = (1.0 - cross_sim[idx].clamp(0.0, 1.0))

        weights.append(w)
    return weights

class Controller:
    def __init__(self, since_best_threshold=1):
        self.since_best_threshold = since_best_threshold
        self.modules = ['rec', 'mot']
        self.best_losses = {key: {'best_loss': float('nan'), 'since_best': 0}
                            for key in self.modules}
        pass

    def keep_going(self, key):
        """ Returns True if model should keep training, false otherwise."""

        if self.best_losses[key]['since_best'] > self.since_best_threshold:
            return False
        else:
            return True

    def update(self, losses):
        for key, l in losses.items():
            if l < self.best_losses[key]['best_loss']:
                self.best_losses[key]['best_loss'] = l
                self.best_losses[key]['since_best'] = 0
            elif not math.isnan(l):
                self.best_losses[key]['since_best'] += 1
            else:
                pass
        pass

    def state_dict(self):
        return {'since_best_threshold': self.since_best_threshold,
                'modules': self.modules,
                'best_losses': self.best_losses
                }

    def set_state(self, state_dict):
        self.since_best_threshold = state_dict['since_best_threshold']
        self.modules = state_dict['modules']
        self.best_losses = state_dict['best_losses']

def print_gradients(model):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p, p.grad, p.requires_grad, p.shape)
    pass

def motif_train(model,
                train_loader,
                test_loader,
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
                controller_state=None,
                edge_sample_rate=1.0,
                wl_iter=3,
                ):
    """motif_train.

    :param model: MotiFiesta model
    :param loader: Graph DataLoader
    :param null_loader: optional. loader containing 'null graphs'
    :param model_name: ID to save model under
    :param epochs: number of epochs to train
    :param lambda_rec: loss coefficient for embedding representation loss
    :param lambda_mot: loss coefficient for edge scores
    :param max_batches: if not -1, stop after given number of batches
    """
    start_time = time.time()

    writer = SummaryWriter(f"logs/{model_name}")

    if controller_state is None:
        controller = Controller(since_best_threshold=stop_epochs,
                                )
    else:
        controller = Controller()
        controller.set_state(controller_state)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    mot_loss, rec_loss = [torch.tensor(float('nan'))] * 2
    done_training = False

    for epoch in range(epoch_start, epochs):
        if done_training:
            print("DONE TRAINING")
            break

        model.train()
        model.to(get_device())

        num_batches = len(train_loader)
        # only keep parameters that require grad

        rec_loss_tot, mot_loss_tot, = [0] * 2
        t_fwd = t_rec = t_bwd = 0.0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=num_batches):
            if batch_idx >= max_batches and max_batches > 0:
                break

            optimizer.zero_grad()

            batch_pos = batch['pos'].to(get_device())
            x_pos, edge_index_pos = batch_pos.x, batch_pos.edge_index

            _t = time.time()
            xx_pos, pp_pos, ee_pos, _, merge_info_pos, internals_pos = model(x_pos,
                                                                              edge_index_pos,
                                                                              batch_pos.batch
                                                                              )
            t_fwd += time.time() - _t

            loss = 0
            backward = False
            warmup_done = False

            if controller.keep_going('rec') and not hard_embed:
                source_pos = _make_batch_source(batch_pos)
                _t = time.time()
                rec_loss = model.rec_loss_wl(xx_pos,
                                             ee_pos,
                                             merge_info_pos,
                                             source_pos,
                                             internals_pos,
                                             edge_sample_rate=edge_sample_rate,
                                             wl_iter=wl_iter,
                                             )
                t_rec += time.time() - _t
                rec_loss_tot += rec_loss.item()
                backward = True
                loss += rec_loss
            else:
                warmup_done = True

            if controller.keep_going('mot') and warmup_done:
                batch_neg = batch['neg'].to(get_device())
                x_neg, edge_index_neg = batch_neg.x, batch_neg.edge_index
                xx_neg, pp_neg, ee_neg, _, merge_info_neg, internals_neg = model(x_neg,
                                                                                  edge_index_neg,
                                                                                  batch_neg.batch
                                                                                  )
                mot_loss = model.freq_loss(internals_pos,
                                           internals_neg,
                                           pp_pos,
                                           steps=model.steps,
                                           estimator=estimator,
                                           volume=volume,
                                           k=n_neighbors,
                                           lam=lam,
                                           beta=beta,
                                           )
                loss += mot_loss
                mot_loss_tot += mot_loss.item()
                backward = True

            if backward:
                _t = time.time()
                loss.backward()
                optimizer.step()
                t_bwd += time.time() - _t
            else:
                done_training = True

        N = max_batches if max_batches > 0 else len(train_loader)

        losses = {'rec': rec_loss_tot / N,
                  'mot': mot_loss_tot / N,
                  }

        ## END OF BATCHES ##
        rec_loss_tot, mot_loss_tot, = [0] * 2

        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if batch_idx >= max_batches and max_batches > 0:
                break

            model.eval()

            batch_pos = batch['pos'].to(get_device())
            x_pos, edge_index_pos = batch_pos.x, batch_pos.edge_index

            with torch.no_grad():
                xx_pos, pp_pos, ee_pos, _, merge_info_pos, internals_pos = model(x_pos,
                                                                                  edge_index_pos,
                                                                                  batch_pos.batch
                                                                                  )
            warmup_done = False

            if controller.keep_going('rec'):
                with torch.no_grad():
                    source_pos = _make_batch_source(batch_pos)
                    rec_loss = model.rec_loss_wl(xx_pos,
                                                 ee_pos,
                                                 merge_info_pos,
                                                 source_pos,
                                                 internals_pos,
                                                 edge_sample_rate=edge_sample_rate,
                                                 wl_iter=wl_iter,
                                                 )
            else:
                warmup_done = True

            mot_loss = torch.tensor(float('nan'))

            if warmup_done:
                batch_neg = batch['neg'].to(get_device())
                x_neg, edge_index_neg = batch_neg.x, batch_neg.edge_index
                with torch.no_grad():
                    xx_neg, pp_neg, ee_neg, _, merge_info_neg, internals_neg = model(x_neg,
                                                                                      edge_index_neg,
                                                                                      batch_neg.batch
                                                                                      )
                mot_loss = model.freq_loss(internals_pos,
                                           internals_neg,
                                           pp_pos,
                                           steps=model.steps,
                                           estimator=estimator,
                                           volume=volume,
                                           k=n_neighbors,
                                           lam=lam,
                                           beta=beta,
                                           )


            if not warmup_done:
                rec_loss_tot += rec_loss.item()
            mot_loss_tot += mot_loss.item()

        N = max_batches if max_batches > 0  else len(test_loader)

        test_losses = {'rec': rec_loss_tot / N,
                       'mot': mot_loss_tot / N,
                       }

        controller.update(test_losses)

        torch.save({
            'epoch': epoch,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'controller_state_dict': controller.state_dict()
        }, f'models/{model_name}/{model_name}.pth')

        loss_str = ' '.join([f'{k} train: {v:2f}' for k,v in losses.items()])
        test_loss_str = ' '.join([f'{k} test: {v:2f}' for k,v in test_losses.items()])
        print(f"  timers — fwd: {t_fwd:.1f}s  rec_loss: {t_rec:.1f}s  bwd: {t_bwd:.1f}s")
        time_elapsed = time.time() - start_time
        print(f"Train Epoch: {epoch+1} [{batch_idx +1}/{num_batches}]"\
              f"({100. * (batch_idx +1) / num_batches :.2f}%) {loss_str}"\
              f" {test_loss_str}"\
              f" Time: {time_elapsed:.2f}"
              )

        # tensorboard logging
        step = epoch * num_batches + batch_idx
        writer.add_scalar("Training loss", loss, step)
        for k,v in losses.items():
            writer.add_scalar(k, v, step)

        for k,v in test_losses.items():
            writer.add_scalar(k, v, step)

    torch.save({
        'epoch': epochs,
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'controller_state_dict': controller.state_dict()
    }, f'models/{model_name}/{model_name}.pth')
