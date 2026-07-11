import time

from tqdm import tqdm
import torch

from MotiFiesta.utils.learning_utils import get_device


def disc_train(model,
               train_loader,
               test_loader,
               model_name='default',
               epochs=10,
               lam=1.0,
               beta=1.0,
               max_batches=-1,
               n_neighbors=30,
               grad_clip=1.0,
               epoch_start=0,
               best_loss=float('inf'),
               optimizer=None,
               warmup=0,
               freq_lr=1e-3,
               tau=0.1,
               lr_patience=0,
               max_knn_size=-1,
               sigma=0.5,
               freeze_gin=False,
               warmup_loss='ae',
               ae_freq=False,
               ae_eps=0.0,
               wwl_freq=False,
               estimator='knn',
               score_weight=False,
               scheduler_state=None,
               anneal_epochs=0,
               ae_min_weight=0.1,
               infonce_tau=0.1,
               **_,
               ):
    device = get_device()
    start_time = time.time()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    # move optimizer state tensors to device (needed when restarting from cpu checkpoint)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    freq_epochs = epochs - warmup
    scheduler = None

    for epoch in range(epoch_start, epochs):
        # at the warmup→freq_loss boundary: reset optimizer and scheduler, reset best_loss
        # for anneal mode, trigger at epoch 1 (when freq_w first becomes nonzero)
        boundary = 1 if anneal_epochs > 0 else warmup
        if epoch == boundary and (warmup > 0 or anneal_epochs > 0):
            if freeze_gin:
                # freeze everything except score_net; freq_loss only trains the scorer
                for p in model.parameters():
                    p.requires_grad_(False)
                for snet in [model.score_net_s, model.score_net_m, model.score_net_l]:
                    for p in snet.parameters():
                        p.requires_grad_(True)
                score_params = (list(model.score_net_s.parameters()) +
                                list(model.score_net_m.parameters()) +
                                list(model.score_net_l.parameters()))
                optimizer = torch.optim.Adam(score_params, lr=freq_lr)
            else:
                for pg in optimizer.param_groups:
                    pg['lr'] = freq_lr
            scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=1e-6)
                if lr_patience > 0 else None)
            if scheduler is not None and scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            best_loss = float('inf')

        is_warmup = epoch < warmup and anneal_epochs == 0
        # annealing schedule: ae weight 1.0→ae_min_weight, freq weight 0→1 over anneal_epochs
        if anneal_epochs > 0:
            progress  = min(epoch / anneal_epochs, 1.0)
            ae_w      = 1.0 - progress * (1.0 - ae_min_weight)
            freq_w    = progress
        else:
            ae_w, freq_w = None, None

        model.train()
        num_batches = len(train_loader)
        train_loss = 0.0
        t_fwd = t_loss = t_bwd = 0.0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=num_batches):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            optimizer.zero_grad()

            t = time.time()
            batch_pos = batch['pos'].to(device)
            _walk_pairs = is_warmup and warmup_loss == 'walk_pair'
            levels_pos = model(batch_pos.x, batch_pos.edge_index, batch_pos.batch,
                               walk_pairs=_walk_pairs)
            t_fwd += time.time() - t

            t = time.time()
            if anneal_epochs > 0:
                # joint loss from epoch 0: ae anneals down, freq anneals up
                batch_neg  = batch['neg'].to(device)
                levels_neg = model(batch_neg.x, batch_neg.edge_index, batch_neg.batch)
                freq = model.freq_loss(levels_pos, levels_neg, beta=beta, lam=lam, sigma=sigma, estimator=estimator, score_weight=score_weight)
                loss = ae_w * model.ae_loss(levels_pos, ae_eps=ae_eps) + freq_w * freq
                del batch_pos, batch_neg, levels_pos, levels_neg
            elif is_warmup:
                if warmup_loss == 'rec':
                    loss = model.rec_loss(levels_pos)
                elif warmup_loss == 'wwl':
                    loss = model.wwl_loss(levels_pos)
                elif warmup_loss == 'ae+wwl':
                    loss = model.ae_loss(levels_pos, ae_eps=ae_eps) + model.wwl_loss(levels_pos)
                elif warmup_loss == 'infonce':
                    loss = model.infonce_loss(levels_pos, tau=infonce_tau)
                elif warmup_loss == 'walk_pair':
                    loss = model.walk_pair_loss(levels_pos)
                else:
                    loss = model.ae_loss(levels_pos, ae_eps=ae_eps)
                del batch_pos, levels_pos
            else:
                batch_neg = batch['neg'].to(device)
                levels_neg = model(batch_neg.x, batch_neg.edge_index, batch_neg.batch)
                freq = model.freq_loss(levels_pos, levels_neg, beta=beta, lam=lam, sigma=sigma, estimator=estimator, score_weight=score_weight)
                loss = freq if (freeze_gin or not ae_freq) else 0.1 * model.ae_loss(levels_pos, ae_eps=ae_eps) + freq
                if wwl_freq:
                    loss = loss + model.wwl_loss(levels_pos)
                del batch_pos, batch_neg, levels_pos, levels_neg
            t_loss += time.time() - t
            train_loss += loss.item()

            t = time.time()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            del loss
            t_bwd += time.time() - t

        if scheduler is not None and (not is_warmup or anneal_epochs > 0):
            scheduler.step(test_loss)

        n_train = min(max_batches, num_batches) if max_batches > 0 else num_batches
        train_loss /= n_train

        model.eval()
        test_loss = 0.0
        n_test_batches = 0
        score_buf = []
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            with torch.no_grad():
                batch_pos = batch['pos'].to(device)
                _walk_pairs = is_warmup and warmup_loss == 'walk_pair'
                levels_pos = model(batch_pos.x, batch_pos.edge_index, batch_pos.batch,
                                   walk_pairs=_walk_pairs)
                if anneal_epochs > 0:
                    batch_neg  = batch['neg'].to(device)
                    levels_neg = model(batch_neg.x, batch_neg.edge_index, batch_neg.batch)
                    freq = model.freq_loss(levels_pos, levels_neg, beta=beta, lam=lam, sigma=sigma, estimator=estimator, score_weight=score_weight)
                    fl = ae_w * model.ae_loss(levels_pos, ae_eps=ae_eps) + freq_w * freq
                    test_loss += fl.item()
                    score_buf.append(levels_pos[0]['scores'].cpu())
                    del batch_pos, batch_neg, levels_pos, levels_neg
                elif is_warmup:
                    if warmup_loss == 'rec':
                        wl = model.rec_loss(levels_pos)
                    elif warmup_loss == 'wwl':
                        wl = model.wwl_loss(levels_pos)
                    elif warmup_loss == 'ae+wwl':
                        wl = model.ae_loss(levels_pos, ae_eps=ae_eps) + model.wwl_loss(levels_pos)
                    elif warmup_loss == 'infonce':
                        wl = model.infonce_loss(levels_pos, tau=infonce_tau)
                    elif warmup_loss == 'walk_pair':
                        wl = model.walk_pair_loss(levels_pos)
                    else:
                        wl = model.ae_loss(levels_pos, ae_eps=ae_eps)
                    test_loss += wl.item()
                    del batch_pos, levels_pos
                else:
                    batch_neg = batch['neg'].to(device)
                    levels_neg = model(batch_neg.x, batch_neg.edge_index, batch_neg.batch)
                    freq = model.freq_loss(levels_pos, levels_neg, beta=beta, lam=lam, sigma=sigma, estimator=estimator, score_weight=score_weight)
                    fl = freq if (freeze_gin or not ae_freq) else 0.1 * model.ae_loss(levels_pos, ae_eps=ae_eps) + freq
                    if wwl_freq:
                        fl = fl + model.wwl_loss(levels_pos)
                    test_loss += fl.item()
                    score_buf.append(levels_pos[0]['scores'].cpu())
                    del batch_pos, batch_neg, levels_pos, levels_neg
            n_test_batches += 1
        test_loss /= max(n_test_batches, 1)

        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
        checkpoint = {
            'epoch': epoch,
            'best_loss': best_loss,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(checkpoint, f'models/{model_name}/{model_name}.pth')
        if is_best:
            torch.save(checkpoint, f'models/{model_name}/{model_name}_best.pth')

        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if anneal_epochs > 0:
            phase = f'anneal({ae_w:.2f}ae+{freq_w:.2f}f)'
        else:
            phase = 'warmup' if is_warmup else 'freq'
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        score_str = ''
        if score_buf:
            s = torch.cat(score_buf)
            score_str = (f"  s_mean: {s.mean():.3f}  s_std: {s.std():.3f}"
                         f"  sat: {(s > 0.9).float().mean():.2f}")
        print(f"epoch {epoch+1}/{epochs} [{phase}]  train: {train_loss:.4f}  test: {test_loss:.4f}"
              f"  lr: {lr:.2e}  fwd: {t_fwd:.1f}s  loss: {t_loss:.1f}s  bwd: {t_bwd:.1f}s"
              f"  elapsed: {elapsed:.1f}s{score_str}")
