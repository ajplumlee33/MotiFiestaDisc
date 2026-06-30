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
        if epoch == warmup and warmup > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=freq_lr)
            scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=1e-6)
                if lr_patience > 0 else None)
            best_loss = float('inf')

        is_warmup = epoch < warmup
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
            levels_pos = model(batch_pos.x, batch_pos.edge_index, batch_pos.batch)
            t_fwd += time.time() - t

            t = time.time()
            if is_warmup:
                loss = model.rec_loss(levels_pos)
                del batch_pos, levels_pos
            else:
                batch_neg = batch['neg'].to(device)
                with torch.no_grad():
                    levels_neg = model(batch_neg.x, batch_neg.edge_index, batch_neg.batch)
                loss = model.freq_loss(levels_pos, levels_neg, beta=beta, lam=lam, k=n_neighbors)
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

        if scheduler is not None and not is_warmup:
            scheduler.step(test_loss)

        n_train = min(max_batches, num_batches) if max_batches > 0 else num_batches
        train_loss /= n_train

        model.eval()
        test_loss = 0.0
        n_test_batches = 0
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            with torch.no_grad():
                batch_pos = batch['pos'].to(device)
                levels_pos = model(batch_pos.x, batch_pos.edge_index, batch_pos.batch)
                if is_warmup:
                    test_loss += model.rec_loss(levels_pos).item()
                    del batch_pos, levels_pos
                else:
                    batch_neg = batch['neg'].to(device)
                    levels_neg = model(batch_neg.x, batch_neg.edge_index, batch_neg.batch)
                    test_loss += model.freq_loss(
                        levels_pos, levels_neg, beta=beta, lam=lam, k=n_neighbors
                    ).item()
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
        }
        torch.save(checkpoint, f'models/{model_name}/{model_name}.pth')
        if is_best:
            torch.save(checkpoint, f'models/{model_name}/{model_name}_best.pth')

        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        phase = 'warmup' if is_warmup else 'freq'
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        print(f"epoch {epoch+1}/{epochs} [{phase}]  train: {train_loss:.4f}  test: {test_loss:.4f}"
              f"  lr: {lr:.2e}  fwd: {t_fwd:.1f}s  loss: {t_loss:.1f}s  bwd: {t_bwd:.1f}s"
              f"  elapsed: {elapsed:.1f}s")
