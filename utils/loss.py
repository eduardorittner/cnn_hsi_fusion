import torch
import torch.nn as nn
import numpy as np
from torchmetrics.functional.image import (
    spectral_angle_mapper,
    structural_similarity_index_measure,
)


class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        assert output.shape == target.shape
        error = torch.abs(output - target) / target
        mrae = torch.mean(error.flatten())
        return mrae


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        assert output.shape == target.shape
        error = output - target
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.flatten()))
        return rmse


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, data_range: int = 255
    ):
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


class Loss_SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target: torch.Tensor, output: torch.Tensor):
        return structural_similarity_index_measure(
            output, target, data_range=(0.0, 1.0)
        )


class Loss_SAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target: torch.Tensor, output: torch.Tensor):
        return spectral_angle_mapper(output, target)
