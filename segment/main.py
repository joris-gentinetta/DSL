import os
from os.path import join
import time

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Starting main imports...")
start = time.time()
import argparse
from tifffile import imread
from pathlib import Path
import dask.array as da
import zarr

import numpy as np

from ultrack.utils import labels_to_edges
from skimage.transform import rescale
print(f"Finished imports in {time.time() - start} seconds")

RESCALE = False

def segment(data_dir, input_file, n_frames, override=False):
    img_path = input_file
    data_dir = Path(data_dir)
    normalized_path = data_dir / "normalized.npy"
    cellpose_path = data_dir / "cellpose_labels.npy"
    wscp_path = data_dir / "wscp_labels.npy"
    stardist_path = data_dir / "stardist_labels.npy"
    wssd_path = data_dir / "wssd_labels.npy"
    detection_path = data_dir / "detections.npz"

    print("Loading image...")
    start = time.time()
    imgs = imread(img_path)
    print(f"Loaded image in {time.time() - start} seconds")

    imgs = imgs[:, 1:, :, :]

    if n_frames is not None:
        imgs = imgs[:n_frames]
    imgs = np.swapaxes(imgs, 1, 3)

    original_scale = imgs.shape

    if RESCALE:
        ## Downscale images
        rescale_factor = 0.5
        downscaled = np.zeros((imgs.shape[0], int(rescale_factor * imgs.shape[1]), int(rescale_factor * imgs.shape[2]), imgs.shape[3]))
        for i in range(imgs.shape[0]):
            downscaled[i, ...] = rescale(imgs[i, ...], rescale_factor, channel_axis=2, anti_aliasing=True, preserve_range = True)
        chunks = (1, downscaled.shape[1], downscaled.shape[2], 1) # chunk size used to compress input
        imgs = da.from_array(downscaled, chunks=chunks)
    else:
        chunks = (1, imgs.shape[1], imgs.shape[2], 1)
        imgs = da.from_array(imgs, chunks=chunks)

    if not normalized_path.exists() or override:
        import preprocess
        preprocess.preprocess(data_dir, imgs, RESCALE=RESCALE)
    
    if not cellpose_path.exists() or override:
        import cellpose_segment
        cellpose_segment.segment(data_dir, RESCALE=RESCALE)

    cellpose_labels = da.from_array(np.load(cellpose_path))
    # wscp_labels = da.from_array(np.load(wscp_path))

    if not stardist_path.exists() or override:
        import stardist_segment
        stardist_segment.segment(data_dir, RESCALE=RESCALE)
    
    stardist_labels = da.from_array(np.load(stardist_path))
    # wssd_labels = da.from_array(np.load(wssd_path))

    detection, contours = labels_to_edges(
        [stardist_labels[..., c] for c in range(stardist_labels.shape[-1])] +\
        [cellpose_labels[..., c] for c in range(cellpose_labels.shape[-1])],
        # [wscp_labels[..., c] for c in range(wscp_labels.shape[-1])] +\
        # [wssd_labels[..., c] for c in range(wssd_labels.shape[-1])],
        sigma=1.0,
        detection_store_or_path=zarr.TempStore(),
        edges_store_or_path=zarr.TempStore(),
    )

    if detection.shape != original_scale:
        rescale_factor = original_scale[1] / detection.shape[1]
        detection_upscale = rescale(detection, rescale_factor, channel_axis=0, anti_aliasing=False, preserve_range=True)
        detection = detection_upscale
    
    if contours.shape != original_scale:
        rescale_factor = original_scale[1] / contours.shape[1]
        contours_upscale = rescale(contours, rescale_factor, channel_axis=0, anti_aliasing=False, preserve_range=True)
        contours = contours_upscale
    
    np.savez_compressed(detection_path, detection=detection, edges=contours)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, default="4T1 p27 trial period.HTD - Well D02 Field #3.tif" , required=False, help='Path to the image file')
    parser.add_argument('--n_frames', type=int, default=-1, required=False, help='Number of frames (optional)')
    parser.add_argument('--override', default=True, required=False, action='store_true', help='Override existing files')
    args = parser.parse_args()

    # create the folder to store the results:
    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent.parent, "output", experiment)
    os.makedirs(output_dir, exist_ok=True)

    input_file = join(Path(__file__).parent.parent, "input", args.file)

    segment(output_dir, input_file, args.n_frames, args.override)