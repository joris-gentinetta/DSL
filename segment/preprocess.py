import time

print("Starting preprocess imports...")
start = time.time()
import gc
from pathlib import Path
from tifffile import imread

import numpy as np
import dask.array as da

from ultrack.imgproc import normalize
from ultrack.utils.array import array_apply

from utils import remove_background

print(f"Finished imports in {time.time() - start} seconds")


def preprocess(folder_path, imgs, RESCALE=False):
    data_dir = Path(folder_path)
    foreground_path = data_dir / "foreground.npy"
    normalized_path = data_dir / "normalized.npy"

    foreground = np.zeros(imgs.shape, dtype=imgs.dtype)
    # foreground = create_zarr(imgs.shape, imgs.dtype, foreground_path, chunks = chunks)
    print("Starting background removal...")
    array_apply(
        imgs,
        out_array=foreground,
        func=remove_background,
        sigma=10.0,
        axis=(0, 3),
    )
    np.save(foreground_path, foreground)

    normalized = np.zeros(imgs.shape, dtype=np.float16)
    # normalized = create_zarr(imgs.shape, np.float16, normalized_path, chunks = chunks)
    print("Normalizing Image")
    array_apply(
        foreground,
        out_array=normalized,
        func=normalize,
        gamma=0.5,
        lower_q=0.55,
        axis=(0, 3),
    )
    summ = np.sum(normalized, axis=-1, keepdims=True)
    summn = np.zeros(summ.shape, dtype=np.float16)
    array_apply(
        summ,
        out_array=summn,
        func=normalize,
        gamma=0.5,
        lower_q=0.55,
        axis=0,
    )
    normalized = np.concatenate((normalized, summn), axis=-1)

    np.save(normalized_path, normalized)
