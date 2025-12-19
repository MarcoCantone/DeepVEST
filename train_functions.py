from math import ceil

import torch
import time
import numpy as np

from utilities import create_figure

import os
import json
import random
from sklearn import metrics


def train_one_epoch(dataloader, model, criterion, optimizer, scheduler=None, accumulation=1,  device=torch.device("cpu"), print_per_epoch=10):
    model.train()

    loss_sum = 0.0
    verbose_step = ceil(len(dataloader) / print_per_epoch)

    start_time = time.time()
    for step_i, batch in enumerate(dataloader):

        if step_i == 45:
            print("stop")

        # TODO: specify keys in function parameters?
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


def training(dataloader_train,
             dataloader_valid,
             model,
             criterion,
             optimizer,
             scheduler=None,
             epochs: int = 30,
             device=torch.device("cpu"),
             resume: bool = False,
             workspace: str = None,
             print_per_epoch=10):
    model = model.to(device, non_blocking=True)

    results = {"epochs": [],
               "losses": [],
               "train_accuracies": [],
               "valid_accuracies": [],
               "mcc": [],
               "roc": [],
               "auc": [],
               "tn": [],
               "fp": [],
               "fn": [],
               "tp": []}
    best_accuracy = float("-inf")

    epoch = 0

    if resume:
        checkpoint = torch.load(os.path.join(workspace, "checkpoint"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        criterion = checkpoint["loss"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["np_rng_state"])
        random.setstate(checkpoint["random_state"])

        results = torch.load(os.path.join(workspace, "results"))

        best_accuracy = np.max(results["valid_accuracies"])

        epoch = results["epochs"][-1]

        print("Loaded checkpoint...")

    while epoch < epochs:
        epoch += 1

        print(f"-------- EPOCH {epoch} --------")
        # train
        print(f"TRAINING...")
        start_time = time.time()
        avg_loss, accuracy_train = train_one_epoch(dataloader_train,
                                                   model,
                                                   criterion,
                                                   optimizer,
                                                   scheduler,
                                                   device,
                                                   print_per_epoch=print_per_epoch)
        print(f"training time={time.time() - start_time:.3f} s")

        # test
        print(f"\nTEST...")
        start_time = time.time()
        score, targets_one_hot = test(dataloader_valid, model, device, print_per_epoch=print_per_epoch)
        print(f"test time={time.time() - start_time:.3f} s")

        # compute metrics
        num_classes = score.shape[1]
        predictions = torch.argmax(score, dim=1)
        targets = torch.argmax(targets_one_hot, dim=1)
        acc = float(metrics.accuracy_score(targets.cpu(), predictions.cpu()))
        mcc = float(metrics.matthews_corrcoef(targets.cpu(), predictions.cpu()))

        results["epochs"].append(epoch)
        results["losses"].append(avg_loss)
        results["train_accuracies"].append(accuracy_train)
        results["valid_accuracies"].append(acc)
        results["mcc"].append(mcc)

        if num_classes == 2:  # only for binary classification
            # tn, fp, fn, tp = metrics.confusion_matrix(targets.cpu(), predictions.cpu()).ravel()
            # tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
            roc = metrics.roc_curve(targets.cpu(), score[:, 1].cpu())  # roc = (fpr, tpr, threshold)
            auc = float(metrics.auc(roc[0], roc[1]))
            # results["tn"].append(tn)
            # results["fp"].append(fp)
            # results["fn"].append(fn)
            # results["tp"].append(tp)
            results["roc"].append(roc)
            results["auc"].append(auc)

        if workspace:
            torch.save(results, os.path.join(workspace, "results"))
            scheduler_state_dict = scheduler.state_dict() if scheduler else None
            torch.save({
                'scheduler_state_dict': scheduler_state_dict,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'random_state': random.getstate()
            }, os.path.join(workspace, "checkpoint"))
            with open(workspace + "/metrics.json", "w") as metric_file:
                metric_file.write(json.dumps(
                    # {i: results[i] for i in ['losses', 'train_accuracies', 'valid_accuracies', 'mcc', 'auc']},
                    {i: results[i] for i in results if i != "roc"},
                    indent=4))

            create_figure(results, workspace)

            if acc > best_accuracy:
                best_accuracy = acc
                # save best model
                torch.save(model.state_dict(), os.path.join(workspace, "best_model.pth"))
                # save valid roc associated to best model
                if num_classes == 2:
                    with open(workspace + "/best_roc.json", "w") as roc_file:
                        roc_file.write(json.dumps(
                            {"fpr": list(roc[0]), "tpr": list(roc[1]), "threshold": list(roc[2].astype("float64"))},
                            indent=4))

        print(f"\nRESULT EPOCH {epoch}")
        print(f"avg_loss: {avg_loss}")
        print(f"train_accuracies: {accuracy_train}")
        print(f"valid_accuracies: {acc}")
        print(f"mcc: {mcc}\n")

    print()
    print(f"EXPERIMENT RESULT")
    print(f"avg_loss:\n{results['losses']}\n")
    print(f"train_accuracies:\n{results['train_accuracies']}\n")
    print(f"valid_accuracies:\n{results['valid_accuracies']}\n")
    print(f"mcc:\n{results['mcc']}\n")

    return results


def test(dataloader,
         model,
         device=torch.device("cpu"),
         print_per_epoch=10):
    model.eval()

    score = torch.zeros(0).to(device, non_blocking=True)
    db_labels = torch.zeros(0)

    with torch.no_grad():
        start_time = time.time()
        for batch_i, (inputs, labels) in enumerate(dataloader):

            if type(inputs) == type([]):
                inputs = [x.to(device, non_blocking=True) for x in inputs]
            else:
                inputs = inputs.to(device, non_blocking=True)

            outputs = model(inputs)

            score = torch.cat((score, outputs), dim=0)
            db_labels = torch.cat((db_labels, labels), dim=0)

            verbose_step = ceil(len(dataloader) / print_per_epoch)
            if batch_i % verbose_step == verbose_step - 1:
                print(f"test progress={100. * (batch_i + 1) / len(dataloader):.1f}%\t"
                      f"elapsed time={time.time() - start_time:.3f} s")
                start_time = time.time()

    return score, db_labels
