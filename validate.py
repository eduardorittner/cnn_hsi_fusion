from utils.log import AverageMeter
from torch.utils.data import DataLoader
import torch
from typing import Callable, Dict
import torch.nn as nn


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_fns: Dict[str, Callable],
) -> Dict[str, float]:
    model.eval()
    losses = {}
    for loss in loss_fns.keys():
        losses[loss] = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(input)
            for loss in losses.keys():
                losses[loss].update(loss_fns[loss](output, target).data)

    for loss in losses.keys():
        losses[loss] = losses[loss].avg

    return losses
