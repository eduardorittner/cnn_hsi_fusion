from dataset import TrainDataset, ValidationDataset
from utils.loss import Loss_MRAE, Loss_RMSE, Loss_PSNR
from utils.log import AverageMeter, time2file_name, initialize_logger, save_checkpoint
from torch.utils.data import DataLoader
from models import *
import argparse
import datetime
import os
import torch


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
opt = parser.parse_args()

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

if torch.cuda.is_available():
    model.cuda()
    loss_mrae.cuda()
    loss_rmse.cuda()
    loss_psnr.cuda()
else:
    print("[Warning]: Cuda is not available")

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


def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()

    # We compute the validation loss on a 256x256 patch on the center of the image
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(input)
            mrae = loss_mrae(
                output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128]
            )
            rmse = loss_rmse(
                output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128]
            )
            psnr = loss_psnr(
                output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128]
            )
        losses_mrae.update(mrae.data)
        losses_rmse.update(rmse.data)
        losses_psnr.update(psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg


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
            lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad()
            output = model(images)
            loss = loss_mrae(output, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter += 1
            if iter % 20 == 0:
                print(
                    f"[iter:{iter}/{total_iters}], lr={lr}, losses average: {losses.avg}"
                )

            if iter % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                if (
                    torch.abs(mrae_loss - record_mrae_loss) < 0.01
                    or mrae_loss < record_mrae_loss
                    or iter % 5000 == 0
                ):
                    print(f"Saving to checkpoint: {logfile}")
                    save_checkpoint(logfile, (iter // 1000), iter, model, optimizer)
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
