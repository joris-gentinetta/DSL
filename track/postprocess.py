# stardist / tensorflow env variables setup
import os
from os.path import join

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import pickle
import napari
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import dask.array as da
from tqdm import tqdm
from skimage.transform import rescale

from tifffile import imread
import matplotlib.pyplot as plt
from PIL import Image
import os



def preprocess(folder_path, img_path, config_id, n_frames=None):
    data_dir = Path(folder_path)
    normalized_path = data_dir / "normalized.npy"
    cellpose_path = data_dir / "cellpose_labels.npy"
    wscp_path = data_dir / "wscp_labels.npy"
    stardist_path = data_dir / "stardist_labels.npy"
    wssd_path = data_dir / "wssd_labels.npy"
    detection_path = data_dir / config_id / "detections.npz"
    tracks_path = data_dir / config_id / "tracks.pkl"
    track_label_path = data_dir / config_id / "track_labels.npy"
    graph_path = data_dir / config_id / "graph.pkl"

    imgs = imread(img_path)

    imgs = imgs[:, 1:, :, :]

    if n_frames is not None:
        imgs = imgs[:n_frames]
    imgs = np.swapaxes(imgs, 1, 3)

    chunks = (1, imgs.shape[1], imgs.shape[2], 1)  # chunk size used to compress input

    normalized = da.from_array(np.load(normalized_path), chunks=chunks)

    tracks_df = pd.read_pickle(tracks_path)
    track_name = Path(tracks_path).name
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    labels = np.load(track_label_path)
    track_label_name = Path(track_label_path).name
    # for id in tqdm(tracks_df.index):
    #     for channel in range(4):
    #         mask = labels[tracks_df.loc[id, 't']] == tracks_df.loc[id, 'track_id']
    #         tracks_df.loc[id, f"c_{channel}"] = normalized[tracks_df.loc[id, 't'], :, :, channel][mask].mean()
    print()



    # Function to apply to each row
    def compute_means(row):
        t = row['t']
        track_id = row['track_id']
        mask = labels[t] == track_id
        means = [normalized[t, :, :, channel][mask].mean() for channel in range(4)]
        means = means + np.sum(means)
        return pd.Series(means, index=[f'c_{channel}' for channel in range(4)] + ['c_sum'])

    # Apply the function to each row
    tracks_df[[f'c_{channel}' for channel in range(4)]] = tracks_df.apply(compute_means, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display images and optional overlays.')
    parser.add_argument('--file', type=str, default="demo.tif",
                        help='Path to the image file')
    parser.add_argument('--config_id', type=str, default="1", required=False, help='Name of config file')

    parser.add_argument('--n_frames', type=int, default=100, help='Number of frames (optional)')
    args = parser.parse_args()

    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent.parent, "output", experiment)
    # print(f'rsync -avz --progress jorisg@192.168.1.203:/home/jorisg/projects/DSL/output/ "{join(Path(__file__).parent, "output")}/"')
    # os.system(
    #     f'rsync -avz --progress jorisg@192.168.1.203:/home/jorisg/projects/DSL/output/ {join(Path(__file__).parent, "output")}/')

    input_file = join(Path(__file__).parent.parent, "input", args.file)

    preprocess(output_dir, input_file, args.config_id, args.n_frames)



