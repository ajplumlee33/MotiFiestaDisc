import time
import math

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from MotiFiesta.utils.learning_utils import get_device




class Controller:
    def __init__(self, since_best_threshold=1):
        self.since_best_threshold = since_best_threshold
        self.modules = ['rec', 'mot']
        self.best_losses = {key: {'best_loss': float('nan'), 'since_best': 0}
                            for key in self.modules}

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
                freeze_encoder=False,
                structural_loss_coef=1.0,
                grad_clip=1.0,
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

    if freeze_encoder and hasattr(model, 'edge_scorer'):
        # freeze GIN, input_proj, transforms — only edge_scorer trains
        for name, p in model.named_parameters():
            if 'edge_scorer' not in name:
                p.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"freeze_encoder: training {sum(p.numel() for p in trainable)} params (edge_scorer only)")

    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable)

    mot_loss, rec_loss = [torch.tensor(float('nan'))] * 2
    best_combined_loss = float('inf')

    for epoch in range(epoch_start, epochs):
        in_warmup = epoch < stop_epochs

        if epoch == stop_epochs:
            # reset Adam moment estimates so stale rec_loss momentum doesn't
            # distort the first freq_loss updates
            optimizer.state.clear()
            print("  optimizer state reset for freq_loss phase")

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
                                                                              batch_pos.batch,
                                                                              dummy=in_warmup,
                                                                              )
            t_fwd += time.time() - _t

            loss = 0
            backward = False

            if in_warmup and not hard_embed and hasattr(model, 'rec_loss'):
                _t = time.time()
                rec_loss = model.rec_loss(xx_pos, internals_pos, batch_pos)
                t_rec += time.time() - _t
                rec_loss_tot += rec_loss.item()
                backward = True
                loss += rec_loss

            if not in_warmup:
                if hasattr(model, 'freq_loss'):
                    batch_neg = batch['neg'].to(get_device())
                    with torch.no_grad():
                        _, _, _, _, _, internals_neg = model(
                            batch_neg.x, batch_neg.edge_index, batch_neg.batch
                        )
                    # freeze per-level GIN weights during freq_loss — protects
                    # the rec_loss-trained embeddings from freq_loss gradient disruption.
                    # score_net and transform still learn from freq_loss.
                    # done once at transition (idempotent after first call).
                    if hasattr(model, 'pool_layers') and not getattr(model, '_gin_frozen', False):
                        for layer in model.pool_layers:
                            if hasattr(layer, 'gin'):
                                for p in layer.gin.parameters():
                                    p.requires_grad_(False)
                        model._gin_frozen = True
                    mot_loss = model.freq_loss(internals_pos, internals_neg, pp_pos,
                                               beta=beta, lam=lam, k=n_neighbors)
                else:
                    raise NotImplementedError("model has no freq_loss")
                loss += mot_loss
                mot_loss_tot += mot_loss.item()
                backward = True

            if backward:
                _t = time.time()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                t_bwd += time.time() - _t

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
                                                                                  batch_pos.batch,
                                                                                  dummy=in_warmup,
                                                                                  )
            mot_loss = torch.tensor(float('nan'))

            if in_warmup and hasattr(model, 'rec_loss'):
                with torch.no_grad():
                    rec_loss = model.rec_loss(xx_pos, internals_pos, batch_pos)
                rec_loss_tot += rec_loss.item()

            if not in_warmup:
                if hasattr(model, 'freq_loss'):
                    batch_neg = batch['neg'].to(get_device())
                    with torch.no_grad():
                        _, _, _, _, _, internals_neg = model(
                            batch_neg.x, batch_neg.edge_index, batch_neg.batch
                        )
                        mot_loss = model.freq_loss(internals_pos, internals_neg, pp_pos,
                                                   beta=beta, lam=lam, k=n_neighbors)
                else:
                    with torch.no_grad():
                        raise NotImplementedError("model has no freq_loss")
                mot_loss_tot += mot_loss.item()

        N = max_batches if max_batches > 0  else len(test_loader)

        test_losses = {'rec': rec_loss_tot / N,
                       'mot': mot_loss_tot / N,
                       }

        controller.update(test_losses)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'controller_state_dict': controller.state_dict()
        }
        torch.save(checkpoint, f'models/{model_name}/{model_name}.pth')

        combined = sum(v for v in test_losses.values() if not math.isnan(v))
        if combined < best_combined_loss:
            best_combined_loss = combined
            torch.save(checkpoint, f'models/{model_name}/{model_name}_best.pth')

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
