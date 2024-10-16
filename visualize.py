import argparse
import math
from os import path
from models import model_generator
import glob
from dataset import ValidationDataset, TrainDataset
import matplotlib.pyplot as plt
import torch
import numpy as np

# CLI Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=str, default="./data")
parser.add_argument(
    "--split",
    type=str,
    default="training",
    help="""'training' or 'validation'
'training' is the default value.
""",
)
parser.add_argument(
    "--stage",
    type=str,
    default="pure-rgb",
    help="""'pure-rgb' or 'pure-hsi' for visualizing directly from stored files
'input' for the result from '__getitem__()' call on the Dataset
'predict' for the model output
'target' for the target used
'error' for the difference between output and target
""",
)
parser.add_argument("--id", type=int, default=1, help="Image id")
parser.add_argument("--model", type=str, default="mst_plus_plus", help="model name")
parser.add_argument("--model-path", type=str, help="path to model checkpoint")
opt = parser.parse_args()


def visualize_image(img: np.ndarray, bands: int):
    if img.shape[0] < bands:
        raise Exception(
            f"[ERROR]: tried to visualize {bands} bands but image has {img.shape[0]}"
        )

    plt_dims = math.ceil(math.sqrt(bands))

    fig, axs = plt.subplots(plt_dims, plt_dims)
    i, j = 0, 0
    stride = img.shape[0] // bands
    for i in range(plt_dims):
        for j in range(plt_dims):
            if (i * plt_dims + j) < bands:
                band = stride * (i * plt_dims + j)
                axs[i, j].set_title(f"band: {band}")
                axs[i, j].imshow(img[band, :, :])
            else:
                break
    plt.show()


def pure_rgb(dataset: TrainDataset | ValidationDataset, idx: int):
    dataset.debug = True
    img = dataset[idx - 1][0]  # Filename ids are 1-indexed
    img = (img - img.min()) / (img.max() - img.min())
    visualize_image(img, 3)


def pure_hsi(dataset: TrainDataset | ValidationDataset, idx: int):
    dataset.debug = True
    img = dataset[idx - 1][1]  # Filename ids are 1-indexed
    img = (img - img.min()) / (img.max() - img.min())
    visualize_image(img, 4)


def input(dataset: TrainDataset | ValidationDataset, idx: int):
    if isinstance(dataset, TrainDataset):
        idx = (idx - 1) * dataset.patch_per_img
    else:
        idx -= 1
    img = dataset[(idx)][0]
    visualize_image(img, 4)


def predict(
    dataset: TrainDataset | ValidationDataset,
    idx: int,
    model_name: str,
    model_path: None | str,
):
    if isinstance(dataset, TrainDataset):
        idx = (idx - 1) * dataset.patch_per_img
    else:
        idx -= 1
    img = dataset[(idx)][0]
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)

    model = model_generator(model_name, model_path)
    with torch.no_grad():
        img = model(img).squeeze()

    visualize_image(img, 4)


def target(dataset: TrainDataset | ValidationDataset, idx: int):
    if isinstance(dataset, TrainDataset):
        idx = (idx - 1) * dataset.patch_per_img
    else:
        idx -= 1
    img = dataset[(idx)][1]

    visualize_image(img, 4)


def error(dataset: TrainDataset | ValidationDataset, idx: int, model_name, model_path):
    if isinstance(dataset, TrainDataset):
        idx = (idx - 1) * dataset.patch_per_img
    else:
        idx -= 1
    input, target = dataset[(idx)]
    input, target = torch.from_numpy(input), torch.from_numpy(target)
    input = input.unsqueeze(0)

    model = model_generator(model_name, model_path)
    with torch.no_grad():
        output = model(input).squeeze()

    error = torch.abs(output - target) / target
    error = error.numpy(force=True)

    visualize_image(error, 4)


def main():
    match opt.split:
        case "training":
            dataset = TrainDataset(opt.data_root, 256)
        case "validation":
            dataset = ValidationDataset(opt.data_root)
        case _:
            raise Exception(f"[ERROR]: Unknown stage {opt.split}")

    match opt.stage:
        case "pure-rgb":
            pure_rgb(dataset, opt.id)
        case "pure-hsi":
            pure_hsi(dataset, opt.id)
        case "input":
            input(dataset, opt.id)
        case "predict":
            predict(dataset, opt.id, opt.model, opt.model_path)
        case "target":
            target(dataset, opt.id)
        case "error":
            error(dataset, opt.id, opt.model, opt.model_path)
        case _:
            raise Exception(f"No stage named {opt.stage}")

    return 0


if __name__ == "__main__":
    main()
