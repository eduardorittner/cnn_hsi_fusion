import numpy as np
import os
import h5py
from PIL import Image

jpg_shape = (3, 482, 512)
cropped_jpg_shape = (3, 256, 256)
mat_shape = (31, 482, 512)
cropped_mat_shape = (31, 256, 256)


def arad_open(file: str, cropped=False) -> np.ndarray:
    if os.path.isfile(file):
        if ".mat" in file:
            img = np.astype(np.asarray(h5py.File(file)["cube"]), np.float32)
            img = img.transpose(0, 2, 1)  # [31, 482, 512]

            if cropped:
                assert (
                    img.shape == cropped_mat_shape
                ), f"cropped .mat images should have shape: {cropped_mat_shape}, found: {img.shape}"
            else:
                assert (
                    img.shape == mat_shape
                ), f".mat images should have shape: {mat_shape}, found: {img.shape}"

            return img
        elif ".jpg" in file:
            img = np.astype(np.asarray(Image.open(file)), np.float32)
            img = img.transpose(2, 0, 1)  # [3, 482, 512]

            if cropped:
                assert (
                    img.shape == cropped_mat_shape
                ), f"cropped .jpg images should have shape: {cropped_jpg_shape}, found: {img.shape}"
            else:
                assert (
                    img.shape == jpg_shape
                ), f".jpg images should have shape: {jpg_shape}, found: {img.shape}"

            return img
        raise Exception(f"[ERROR]: expected .mat or .jpg file, got: {file}")
    raise Exception(f"[ERROR]: {file} is not a file.")


def arad_save_hsi(file: str, img: np.ndarray):
    assert img.shape[0] == 31, f"HSI to be saved must have 31 bands, found: {img.shape}"

    f = h5py.File(file + ".mat", "w")

    f.create_dataset("cube", data=img)
    f.create_dataset("bands", data=[float(i) for i in range(400, 701, 10)])
    f.close()
