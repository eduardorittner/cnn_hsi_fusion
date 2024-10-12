import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import torch
import os
import h5py
from PIL import Image

jpg_shape = (3, 482, 512)
mat_shape = (31, 482, 512)


def arad_open(file: str) -> np.ndarray:
    if os.path.isfile(file):
        if ".mat" in file:
            img = np.astype(np.asarray(h5py.File(file)["cube"]), np.float32)
            img = img.transpose(0, 2, 1)  # [31, 482, 512]

            img = (img - img.min()) / (img.max() - img.min())

            assert (
                img.shape == mat_shape
            ), f".mat images should have shape: {mat_shape}, found: {img.shape}"

            return img
        elif ".jpg" in file:
            img = np.astype(np.asarray(Image.open(file)), np.float32)
            img = img.transpose(2, 0, 1)  # [3, 482, 512]

            img = (img - img.min()) / (img.max() - img.min())

            assert (
                img.shape == jpg_shape
            ), f".jpg images should have shape: {jpg_shape}, found: {img.shape}"

            return img
        raise Exception(f"[ERROR]: expected .mat or .jpg file, got: {file}")
    raise Exception(f"[ERROR]: {file} is not a file.")


def produce_input(
    rgb: np.ndarray, spectral: np.ndarray, res: tuple[int, int]
) -> np.ndarray:
    # We are simulating a situation with high-res MSI and low-res HSI
    # so we downsample the HSI to simulate low-res, then upsample it using
    # bilinear interpolation so we can then stack one on top of the other
    # and input that into the networks

    assert (rgb.shape[1], rgb.shape[2]) == res
    assert spectral.shape == (31, res[0], res[1])

    tensor = spectral[:, ::4, ::4]
    tensor = torch.from_numpy(spectral).unsqueeze(0).unsqueeze(0)
    upsampled = torch.empty((1, 1, 31, res[0], res[1]))
    for i in range(tensor.size()[2]):
        band = tensor[:, :, i, :, :]
        upsampled[0, 0, i, :, :] = interpolate(band, (res[0], res[1]), mode="bilinear")

    upsampled = upsampled.squeeze()

    spectral = upsampled.numpy(force=True)

    input = np.concatenate((rgb, spectral), axis=0)
    input = np.ascontiguousarray(input, np.float32)

    assert input.shape == (34, res[0], res[1])

    return input


# We train on image patches and validate on full images


class TrainDataset(Dataset):
    def __init__(self, data_root: str, patch_size: int, stride: int = 8):
        height, width = 482, 512
        self.data_root = data_root
        self.patch_size = patch_size
        self.stride = stride
        self.current_spectral: None | np.ndarray = None
        self.current_rgb: None | np.ndarray = None
        self.current_idx: int | None = None
        self.vertical_patches = ((height - patch_size) // stride) + 1
        self.horizontal_patches = ((width - patch_size) // stride) + 1
        self.patch_per_img = self.vertical_patches * self.horizontal_patches

        spectral_path = f"{data_root}/Train_spectral/"
        rgb_path = f"{data_root}/Train_RGB/"

        with open(f"{data_root}/train_list.txt", "r") as f:
            spectral_list = [line.replace("\n", ".mat") for line in f]
            rgb_list = [line.replace("mat", "jpg") for line in spectral_list]

        spectral_list.sort()
        rgb_list.sort()

        assert len(spectral_list) == len(
            rgb_list
        ), "[Error]: Expected same number of RGB and spectral images, got {len(rgb_list)} RGBs and {len(spectral_list)} spectral"

        print(
            f"Train dataset with {len(spectral_list)} images and {(len(spectral_list)*self.patch_per_img)} total patches."
        )

        self.spectral = [spectral_path + img for img in spectral_list]
        self.rgb = [rgb_path + img for img in rgb_list]

    def __getitem__(self, idx):
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
        h_idx, w_idx = (
            patch_idx // self.horizontal_patches,
            patch_idx // self.vertical_patches,
        )

        if img_idx != self.current_idx:
            self.current_rgb = arad_open(self.rgb[img_idx])
            self.current_spectral = arad_open(self.spectral[img_idx])
            self.current_idx = idx

        rgb = self.current_rgb[
            :,
            h_idx * self.stride : h_idx * self.stride + self.patch_size,
            w_idx * self.stride : w_idx * self.stride + self.patch_size,
        ]

        spectral = self.current_spectral[
            :,
            h_idx * self.stride : h_idx * self.stride + self.patch_size,
            w_idx * self.stride : w_idx * self.stride + self.patch_size,
        ]

        input = produce_input(rgb, spectral, (self.patch_size, self.patch_size))

        # TODO: Maybe add: rotation, horizontal and vertical flip randomly

        return input, np.ascontiguousarray(spectral, np.float32)

    def __len__(self):
        return self.patch_per_img * len(self.spectral)


class ValidationDataset(Dataset):
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.current_spectral: None | np.ndarray = None
        self.current_rgb: None | np.ndarray = None
        self.current_idx: int | None = None

        spectral_path = f"{data_root}/Valid_spectral/"
        rgb_path = f"{data_root}/Valid_RGB/"

        with open(f"{data_root}/valid_list.txt", "r") as f:
            spectral_list = [line.replace("\n", ".mat") for line in f]
            rgb_list = [line.replace("mat", "jpg") for line in spectral_list]

        spectral_list.sort()
        rgb_list.sort()

        assert len(spectral_list) == len(
            rgb_list
        ), "[Error]: Expected same number of RGB and spectral images, got {len(rgb_list)} RGBs and {len(spectral_list)} spectral"

        print(f"Validation dataset with {len(spectral_list)} images.")

        self.spectral = [spectral_path + img for img in spectral_list]
        self.rgb = [rgb_path + img for img in rgb_list]

    def __getitem__(self, idx):
        self.current_rgb = arad_open(self.rgb[idx])[:, 113:-113, 128:-128]
        self.current_spectral = arad_open(self.spectral[idx])[:, 113:-113, 128:-128]
        self.current_idx = idx

        input = produce_input(self.current_rgb, self.current_spectral, (256, 256))

        return input, np.ascontiguousarray(self.current_spectral, np.float32)
