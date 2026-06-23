import time

from tqdm import tqdm
import torch

from MotiFiesta.utils.learning_utils import get_device


def disc_train(model,
               train_loader,
               test_loader,
               model_name='default',
               epochs=10,
               warmup_epochs=0,
               lam=1,
               beta=1,
               max_batches=-1,
               n_neighbors=30,
               grad_clip=1.0,
               epoch_start=0,
               best_loss=float('inf'),
               optimizer=None,
               **_,
               ):
    device = get_device()
    start_time = time.time()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    model.to(device)

    for epoch in range(epoch_start, epochs):
        # reset optimizer at warmup→freq_loss transition so warmup momentum doesn't fight new gradients
        if epoch == warmup_epochs and warmup_epochs > 0:
            optimizer = torch.optim.Adam(model.parameters())
            best_loss = float('inf')

        model.train()
        num_batches = len(train_loader)
        train_loss = 0.0
        t_fwd = t_loss = t_bwd = 0.0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=num_batches):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            optimizer.zero_grad()
            batch_pos = batch['pos'].to(device)

            t = time.time()
            _, pp_pos, _, _, _, internals_pos = model(
                batch_pos.x, batch_pos.edge_index, batch_pos.batch
            )
            t_fwd += time.time() - t

            t = time.time()
            if epoch < warmup_epochs:
                loss = model.rec_loss(internals_pos)
            else:
                batch_neg = batch['neg'].to(device)
                with torch.no_grad():
                    _, _, _, _, _, internals_neg = model(
                        batch_neg.x, batch_neg.edge_index, batch_neg.batch
                    )
                loss = model.freq_loss(internals_pos, internals_neg, pp_pos,
                                       beta=beta, lam=lam, k=n_neighbors)
            t_loss += time.time() - t
            train_loss += loss.item()

            t = time.time()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            t_bwd += time.time() - t

            del batch_pos, internals_pos, pp_pos, loss

        n_train = min(max_batches, num_batches) if max_batches > 0 else num_batches
        train_loss /= n_train

        model.eval()
        test_loss = 0.0
        n_test_batches = 0
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch_pos = batch['pos'].to(device)
            with torch.no_grad():
                _, pp_pos, _, _, _, internals_pos = model(
                    batch_pos.x, batch_pos.edge_index, batch_pos.batch
                )
                if epoch < warmup_epochs:
                    test_loss += model.rec_loss(internals_pos).item()
                else:
                    batch_neg = batch['neg'].to(device)
                    _, _, _, _, _, internals_neg = model(
                        batch_neg.x, batch_neg.edge_index, batch_neg.batch
                    )
                    test_loss += model.freq_loss(internals_pos, internals_neg, pp_pos,
                                                 beta=beta, lam=lam, k=n_neighbors).item()
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

        elapsed = time.time() - start_time
        print(f"epoch {epoch+1}/{epochs}  train: {train_loss:.4f}  test: {test_loss:.4f}"
              f"  fwd: {t_fwd:.1f}s  loss: {t_loss:.1f}s  bwd: {t_bwd:.1f}s"
              f"  elapsed: {elapsed:.1f}s")
