"""Training loop helpers."""

import time
from math import ceil

import torch


def train_one_epoch(dataloader, model, criterion, optimizer, scheduler=None, accumulation=1,  device=torch.device("cpu"), print_per_epoch=10):
    """Train ``model`` for one epoch and return the average training loss.

    Args:
        dataloader: training dataloader returning dictionaries with keys ``"img"`` and ``"seg"``.
        model: network to train.
        criterion: loss function ``criterion(outputs, targets)``.
        optimizer: torch optimizer.
        scheduler: optional learning-rate scheduler, stepped once per epoch.
        accumulation: number of batches over which gradients are accumulated before an
            optimizer step (effective batch size = batch size x accumulation).
        device: device used for training.
        print_per_epoch: number of progress messages printed per epoch.
    """
    model.train()

    loss_sum = 0.0
    verbose_step = ceil(len(dataloader) / print_per_epoch)

    start_time = time.time()
    for step_i, batch in enumerate(dataloader):

        inputs = batch["img"].to(device, non_blocking=True)
        if type(batch["seg"]) == type([]):
            targets = [x.to(device, non_blocking=True) for x in batch["seg"]]
        else:
            targets = batch["seg"].to(device, non_blocking=True)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss /= accumulation
        loss.backward()

        if ((step_i + 1) % accumulation == 0) or (step_i + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        loss_sum += loss.item()*accumulation

        if step_i % verbose_step == verbose_step - 1:
            print(
                f"loss={loss_sum / (step_i + 1):.4f}\t"
                f"training progress={100. * (step_i + 1) / len(dataloader):.1f}%\t"
                f"elapsed time={time.time() - start_time:.3f} s")
            start_time = time.time()

    if scheduler:
        scheduler.step()

    # return average loss
    return loss_sum / len(dataloader)
