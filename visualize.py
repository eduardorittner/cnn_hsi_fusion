import argparse
from os import path
import glob
from dataset import ValidationDataset, TrainDataset
import matplotlib.pyplot as plt

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
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--model-path", type=str, help="path to model checkpoint")
opt = parser.parse_args()


def pure_rgb(dataset: TrainDataset | ValidationDataset, idx: int):
    dataset.debug = True
    fig, axs = plt.subplots(2, 2)
    img = dataset[idx - 1][0]  # Filename ids are 1-indexed
    img = (img - img.min()) / (img.max() - img.min())
    axs[0, 0].imshow(img.transpose(1, 2, 0))
    axs[0, 1].imshow(img[0, :, :], cmap="gray")
    axs[1, 0].imshow(img[1, :, :], cmap="gray")
    axs[1, 1].imshow(img[2, :, :], cmap="gray")
    plt.show()


def pure_hsi(dataset, idx):
    dataset.debug = True
    fig, axs = plt.subplots(2, 2)
    img = dataset[idx - 1][1]  # Filename ids are 1-indexed
    img = (img - img.min()) / (img.max() - img.min())
    axs[0, 0].set_title("0")
    axs[0, 0].imshow(img[0, :, :], cmap="gray")
    axs[0, 1].set_title("10")
    axs[0, 1].imshow(img[10, :, :], cmap="gray")
    axs[1, 0].set_title("20")
    axs[1, 0].imshow(img[20, :, :], cmap="gray")
    axs[1, 1].set_title("30")
    axs[1, 1].imshow(img[30, :, :], cmap="gray")
    plt.show()


def input(dataset, idx):
    return 0


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

    return 0


if __name__ == "__main__":
    main()
