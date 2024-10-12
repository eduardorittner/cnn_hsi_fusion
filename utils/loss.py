import torch
import torch.nn as nn
import numpy as np
import os
import logging


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        "epoch": epoch,
        "iter": iteration,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, "net_%depoch.pth" % epoch))


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = (
        year + "_" + month + "_" + day + "_" + hour + "_" + minute + "_" + second
    )
    return time_filename


def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """Record many results."""
    loss_csv.write(
        "{},{},{},{},{},{}\n".format(
            epoch, iteration, epoch_time, lr, train_loss, test_loss
        )
    )
    loss_csv.flush()
    loss_csv.close


class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, output, target):
        assert output.shape == target.shape
        error = torch.abs(output - target) / target
        mrae = torch.mean(error.view(-1))
        return mrae


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, output, target):
        assert output.shape == target.shape
        error = output - target
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, target, output, data_range=255):
        N = target.size()[0]
        C = target.size()[1]
        H = target.size()[2]
        W = target.size()[3]
        target = target.clamp(0.0, 1.0).mul_(data_range).resize_(N, C * H * W)
        output = output.clamp(0.0, 1.0).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(target, output).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10.0 * torch.log((data_range**2) / err) / np.log(10.0)
        return torch.mean(psnr)
