import argparse
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from os import path


def read_log(file: str, loss_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    iter_string = "iter: "
    train_loss_str = "Train"

    with open(file, "r") as f:
        lines = f.readlines()
        epochs = np.zeros((len(lines)))
        losses = np.zeros((len(loss_keys) + 1, len(lines)))

        for i, line in enumerate(lines):
            iter_start = line.find(iter_string) + len(iter_string)
            iter_end = line.find("/")
            epochs[i] = int(line[iter_start:iter_end]) // 1000

            train_loss_start = line.find(train_loss_str)
            train_loss_start += line[train_loss_start:].find(":") + 2
            train_loss_end = train_loss_start + line[train_loss_start:].find(",")
            losses[0][i] = float(line[train_loss_start:train_loss_end])

            line = line[train_loss_end:]

            for j, loss in enumerate(loss_keys):
                loss_start = line.find(loss) + len(loss) + 2
                if line[loss_start:].find(",") == -1:
                    losses[j + 1][i] = float(line[loss_start:])
                else:
                    loss_end = loss_start + line[loss_start:].find(",")
                    losses[j + 1][i] = float(line[loss_start:loss_end])

    return epochs, losses


def plot(
    epochs: np.ndarray, data: np.ndarray, title: str, losses: list[str]
) -> tuple[Figure, Any]:
    assert epochs.shape[0] == data.shape[1]
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(epochs, data[0, :])
    axs[0, 0].set_title(title)

    for i, loss in enumerate(losses):
        if i > 2:
            axs[1, i % 3].plot(epochs, data[i, :])
            axs[1, i % 3].set_title(loss)
        else:
            axs[0, i].plot(epochs, data[i, :])
            axs[0, i].set_title(loss)

    plt.show()

    return fig, axs


def save_plot(file: str, fig: Figure, axs):
    fig.set_size_inches(12, 7, forward=True)
    fig.savefig(file, dpi=fig.dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=str, help="path to .log file")
    parser.add_argument("--dir", type=str, help="where to save plot")
    parser.add_argument("--name", type=str, default="train-loss", help="plot name")
    parser.add_argument(
        "--losses",
        type=str,
        default="mrae rmse psnr ssim sam",
        help="list of loss names to be plotted separated by space",
    )
    opt = parser.parse_args()

    losses = opt.losses.split(" ")
    epochs, data = read_log(opt.log, losses)

    print(data.shape)
    fig, axs = plot(epochs, data, opt.name, ["train mrae", *losses])

    if opt.dir:
        plot_file = path.join(opt.dir, opt.name)
        save_plot(plot_file, fig, axs)


if __name__ == "__main__":
    main()
