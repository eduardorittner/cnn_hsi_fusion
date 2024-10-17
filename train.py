from torch.cuda import is_available
from dataset import TrainDataset, ValidationDataset
from utils.loss import Loss_MRAE, Loss_RMSE, Loss_PSNR
from utils.log import AverageMeter, time2file_name, initialize_logger, save_checkpoint
from torch.utils.data import DataLoader
from models import *
import argparse
import datetime
import os
import torch
from validate import validate


# CLI Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-path", type=str, default=None)
parser.add_argument("--model", type=str, default="mst_plus_plus")
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--patch-size", type=int, default=128)
parser.add_argument("--stride", type=int, default=8)
parser.add_argument("--end-epoch", type=int, default=300)
parser.add_argument(
    "--outf", type=str, default="./exp/mst-plus-plus", help="path for log files"
)
parser.add_argument("--data-root", type=str, default="./data", help="path to dataset")
parser.add_argument("--disable-cuda", action="store_true")
parser.add_argument("--light", action="store_true")
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
iters_per_epoch = 1000
total_iters = iters_per_epoch * opt.end_epoch

# Loss functions

loss_mrae = Loss_MRAE()
loss_rmse = Loss_RMSE()
loss_psnr = Loss_PSNR()

# Logger

date_time = time2file_name(str(datetime.datetime.now()))
logdir = opt.outf + date_time
if not os.path.exists(logdir):
    os.makedirs(logdir)
logfile = os.path.join(logdir, "train.log")
logger = initialize_logger(logfile)

# Load model

model = model_generator(opt.model, opt.pretrained_path)

model = model.to(device)
loss_fns = {
    "mrae": loss_mrae.to(device),
    "rmse": loss_rmse.to(device),
    "psnr": loss_psnr.to(device),
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


def main():
    global iter
    torch.backends.cudnn.benchmark = True
    record_mrae_loss = 1000
    while iter < total_iters:
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
            if iter % 20 == 0:
                print(
                    f"[iter:{iter}/{total_iters}], lr={lr}, losses average: {losses.avg}"
                )

            if iter % iters_per_epoch == 0:
                val_losses = validate(model, val_loader, device, loss_fns)
                mrae_loss = val_losses["mrae"]
                rmse_loss = val_losses["rmse"]
                psnr_loss = val_losses["psnr"]
                if (
                    torch.abs(mrae_loss - record_mrae_loss) < 0.01
                    or mrae_loss < record_mrae_loss
                    or iter % 5000 == 0
                ):
                    print(f"Saving to checkpoint: {logfile}")
                    save_checkpoint(logdir, (iter // 1000), iter, model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss

                print(
                    f"""iter: {iter}/{total_iters}, lr: {lr}
                    Train MRAE: {losses.avg}, Test MRAE: {mrae_loss}, Test RMSE: {rmse_loss}, Test PSNR: {psnr_loss}"""
                )
                logger.info(
                    f"iter: {iter}/{total_iters}, epoch: {total_iters//1000}, lr: {lr} Train MRAE: {losses.avg}, Test MRAE: {mrae_loss}, Test RMSE: {rmse_loss}, Test PSNR: {psnr_loss}"
                )


if __name__ == "__main__":
    main()
