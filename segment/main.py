import os
from os.path import join
import time

# os.environ["OMP_NUM_THREADS"] = "10"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Starting main imports...")
start = time.time()
import argparse
from tifffile import imread
from pathlib import Path
import dask.array as da
import numpy as np

from skimage.transform import rescale

print(f"Finished imports in {time.time() - start} seconds")

RESCALE = False


def segment(data_dir, input_file, n_frames, override=False):
    img_path = input_file
    data_dir = Path(data_dir)
    normalized_path = data_dir / "normalized.npy"
    cellpose_path = data_dir / "cellpose_labels.npy"
    stardist_path = data_dir / "stardist_labels.npy"

    print("Loading image...")
    start = time.time()
    imgs = imread(img_path)
    print(f"Loaded image in {time.time() - start} seconds")

    imgs = imgs[:, 1:, :, :]

    if n_frames is not None:
        imgs = imgs[:n_frames]
    imgs = np.swapaxes(imgs, 1, 3)

    if RESCALE:
        ## Downscale images
        rescale_factor = 0.5
        downscaled = np.zeros(
            (imgs.shape[0], int(rescale_factor * imgs.shape[1]), int(rescale_factor * imgs.shape[2]), imgs.shape[3]))
        for i in range(imgs.shape[0]):
            downscaled[i, ...] = rescale(imgs[i, ...], rescale_factor, channel_axis=2, anti_aliasing=True,
                                         preserve_range=True)
        chunks = (1, downscaled.shape[1], downscaled.shape[2], 1)  # chunk size used to compress input
        imgs = da.from_array(downscaled, chunks=chunks)
    else:
        chunks = (1, imgs.shape[1], imgs.shape[2], 1)
        imgs = da.from_array(imgs, chunks=chunks)

    if not normalized_path.exists() or override:
        import preprocess
        preprocess_start = time.time()
        preprocess.preprocess(data_dir, imgs, RESCALE=RESCALE)
        print(f'Preprocess: {(time.time() - preprocess_start) / 60} minutes')

    if not cellpose_path.exists() or override:
        import cellpose_segment
        cellpose_start = time.time()
        cellpose_segment.segment(data_dir, RESCALE=RESCALE)
        print(f'Cellpose: {(time.time() - cellpose_start) / 60} minutes')

    if not stardist_path.exists() or override:
        import stardist_segment
        stardist_start = time.time()
        stardist_segment.segment(data_dir, RESCALE=RESCALE)
        print(f'Stardist: {(time.time() - stardist_start) / 60} minutes')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, default="demo.tif", required=False, help='Path to the image file')
    parser.add_argument('--n_frames', type=int, default=None, required=False, help='Number of frames (optional)')
    parser.add_argument('--override', default=True, required=False, action='store_true', help='Override existing files')
    args = parser.parse_args()

    # create the folder to store the results:
    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent.parent, "output", experiment)
    os.makedirs(output_dir, exist_ok=True)

    input_file = join(Path(__file__).parent.parent, "input", args.file)

    segment(output_dir, input_file, args.n_frames, args.override)