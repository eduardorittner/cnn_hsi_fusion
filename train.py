from torch.cuda import is_available
from dataset import TrainDataset, ValidationDataset
from utils.loss import Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_SSIM, Loss_SAM
from utils.log import (
    AverageMeter,
    time2file_name,
    initialize_logger,
    save_checkpoint,
    format_interval_sec,
)
from torch.utils.data import DataLoader
from torch import nn
from models import *
import argparse
import datetime
import os
import time
import torch
from validate import validate
from typing import Callable, Dict


# CLI Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-path", type=str, default=None)
parser.add_argument("--model", type=str, default="mst_plus_plus")
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--patch-size", type=int, default=128)
parser.add_argument("--stride", type=int, default=64)
parser.add_argument("--end-epoch", type=int, default=300)
parser.add_argument("--outf", type=str, default="./exp/", help="path for log files")
parser.add_argument("--data-root", type=str, default="./data", help="path to dataset")
parser.add_argument("--disable-cuda", action="store_true")
parser.add_argument(
    "--light",
    action="store_true",
    help="For running single batch small patches on the cpu",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="For testing. A pretrained-path must be provided",
)
opt = parser.parse_args()

if not opt.disable_cuda and not opt.light:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"[Warning]: CUDA is not available")
else:
    device = torch.device("cpu")

if opt.light:
    opt.patch_size = 32
    opt.stride = 32
    opt.batch_size = 1

# Load datasets

train_data = TrainDataset(
    data_root=opt.data_root, patch_size=opt.patch_size, stride=opt.stride
)
val_data = ValidationDataset(data_root=opt.data_root)

# Iteration numbers

iter = 0
iters_per_update = 100
iters_per_epoch = len(train_data) // opt.batch_size
total_iters = iters_per_epoch * opt.end_epoch

print(f"{iters_per_epoch} iterations per epoch")

# Loss functions

loss_mrae = Loss_MRAE()
loss_rmse = Loss_RMSE()
loss_psnr = Loss_PSNR()
loss_ssim = Loss_SSIM()
loss_sam = Loss_SAM()

# Logger

if not opt.test:
    date_time = time2file_name(str(datetime.datetime.now()))
    logdir = opt.outf + opt.model + date_time
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = os.path.join(logdir, "train.log")
    logger = initialize_logger(logfile)

# Load model

torch.set_float32_matmul_precision("medium")

if opt.test and not opt.pretrained_path:
    print("Warning: Testing an untrained model")

model = model_generator(opt.model, opt.pretrained_path)

model = model.to(device)
loss_fns = {
    "mrae": loss_mrae.to(device),
    "rmse": loss_rmse.to(device),
    "psnr": loss_psnr.to(device),
    "ssim": loss_ssim.to(device),
    "sam": loss_sam.to(device),
}

optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, total_iters, eta_min=1e-6
)

# Load pretrained model

if opt.pretrained_path:
    if os.path.isfile(opt.pretrained_path):
        print(f"Loading checkpoint: '{opt.pretrained_path}'")
        checkpoint = torch.load(opt.pretrained_path)
        start_epoch, iter = checkpoint["epoch"], checkpoint["iter"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])


def test(model: nn.Module, device: torch.device, loss_fns: Dict[str, Callable]):
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    val_losses = validate(model, val_loader, device, loss_fns)
    test_loss = ""
    for loss, value in val_losses.items():
        test_loss += "Test " + loss + f": {value:.5f} | "

    print(test_loss)


def main():
    if opt.test:
        test(model, device, loss_fns)
        return 0
    global iter
    torch.backends.cudnn.benchmark = True
    record_mrae_loss = 1000
    epochs_since_record = 0
    start_time = time.time()
    elapsed_time = 0

    while True:  # We exit this loop from inside the for loop so the check is there
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad()
            output = model(images)
            loss = loss_mrae(output, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iter += 1
            if iter % iters_per_update == 0:
                elapsed_time = time.time() - start_time
                predicted_time = (total_iters / iter) * elapsed_time
                time_left = predicted_time - elapsed_time

                print(
                    f"[iter:{iter}/{total_iters}], lr={lr:.5f}, losses average: {losses.avg:.5f}, {format_interval_sec(elapsed_time)} passed, {format_interval_sec(time_left)} to go"
                )

            if iter % iters_per_epoch == 0 or iter > total_iters:
                val_losses = validate(model, val_loader, device, loss_fns)
                mrae_loss = val_losses["mrae"]
                if iter % 5 * iters_per_epoch == 0 and (
                    torch.abs(mrae_loss - record_mrae_loss) < 0.01
                    or mrae_loss < record_mrae_loss
                ):
                    print(f"Saving to checkpoint: {logfile}")
                    save_checkpoint(
                        logdir, (iter // iters_per_epoch), iter, model, optimizer
                    )
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss
                        epochs_since_record = 0
                    else:
                        epochs_since_record += 1
                else:
                    epochs_since_record += 1

                if epochs_since_record == 10:
                    print(
                        "10 epochs since mrae validation loss improved, stopping now."
                    )
                    save_checkpoint(
                        logdir, (iter // iters_per_epoch), iter, model, optimizer
                    )
                    return 0

                test_loss = ""
                for loss, value in val_losses.items():
                    test_loss += ", Test " + loss + f": {value:.5f}"

                test_loss_log = ""
                for loss, value in val_losses.items():
                    test_loss_log += ", Test " + loss + f": {value}"

                print(f"iter: {iter}/{total_iters}, lr: {lr:.5f}")
                print(f"Train MRAE: {losses.avg}{test_loss}")
                logger.info(
                    f"iter: {iter}/{total_iters}, epoch: {total_iters//iters_per_epoch}, lr: {lr} Train MRAE: {losses.avg}{test_loss_log}"
                )

                if iter > total_iters:
                    return 0


if __name__ == "__main__":
    main()
