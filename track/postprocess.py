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
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor



def postprocess(folder_path, img_path, config_id, n_frames=None):
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

    print()
    # tracks_df = tracks_df.iloc[:1000]



    # Function to apply to each row
    def compute_means(row):
        t = int(row['t'])
        mask = labels[t] == row['track_id']
        means = [float(normalized[t, :, :, channel][mask].mean()) for channel in range(3)]
        sum = np.sum(means)
        means = [mean/sum for mean in means]
        return pd.Series(means, index=[f'c_{channel}' for channel in range(3)] )

    # Apply the function to each row
    time1 = time.time()
    tracks_df.loc[:, [f'c_{channel}' for channel in range(3)]] = tracks_df.apply(compute_means, axis=1)
    print(f"Single thread took {time.time() - time1} seconds")
    print()
    return tracks_df



    # # Corrected function
    # def compute_means(row):
    #     t = int(row['t'])
    #     mask = labels[t] == row['track_id']
    #     means = [float(normalized[t, :, :, channel][mask].mean()) for channel in range(4)]
    #     sum = np.sum(means)
    #     means = [mean / sum for mean in means]
    #     return pd.Series(means, index=[f'c_{channel}' for channel in range(4)])
    #
    # # Function to parallelize the computation
    # def parallelize_computation(df, func):
    #     with ThreadPoolExecutor() as executor:
    #         results = list(executor.map(func, [row for _, row in df.iterrows()]))
    #     return pd.DataFrame(results)

    # # Apply the function in parallel
    # time1 = time.time()
    # tracks_df.loc[:, [f'c_{channel}' for channel in range(4)]] = parallelize_computation(tracks_df, compute_means)
    # print(f"Parallel computation took {time.time() - time1} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display images and optional overlays.')
    parser.add_argument('--file', type=str, default="demo.tif",
                        help='Path to the image file')
    parser.add_argument('--config_id', type=str, default="1", required=False, help='Name of config file')

    parser.add_argument('--n_frames', type=int, default=100, help='Number of frames (optional)')
    parser.add_argument('--beta', type=float, default=0.3, help='Max color composition difference')
    args = parser.parse_args()

    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent.parent, "output", experiment)
    # print(f'rsync -avz --progress jorisg@192.168.1.203:/home/jorisg/projects/DSL/output/ "{join(Path(__file__).parent, "output")}/"')
    # os.system(
    #     f'rsync -avz --progress jorisg@192.168.1.203:/home/jorisg/projects/DSL/output/ {join(Path(__file__).parent, "output")}/')

    input_file = join(Path(__file__).parent.parent, "input", args.file)


    if not os.path.exists(join(output_dir, args.config_id, 'tracks_pp.parquet')):
        tracks_df = postprocess(output_dir, input_file, args.config_id, args.n_frames)
        tracks_df.to_parquet(join(output_dir, args.config_id, 'tracks_pp.parquet'))
    else:
        tracks_df = pd.read_parquet(join(output_dir, args.config_id, 'tracks_pp.parquet'))

    for id in tqdm(tracks_df.index):
        parent_id = tracks_df.loc[id, 'parent_id']
        if not parent_id == -1:
            cut_track = False
            if tracks_df.loc[parent_id, 'c_0'] > tracks_df.loc[id, 'c_0'] + args.beta or \
                    tracks_df.loc[parent_id, 'c_0'] < tracks_df.loc[id, 'c_0'] - args.beta:
                cut_track = True
            elif tracks_df.loc[parent_id, 'c_1'] > tracks_df.loc[id, 'c_1'] + args.beta or \
                    tracks_df.loc[parent_id, 'c_1'] < tracks_df.loc[id, 'c_1'] - args.beta:
                cut_track = True
            elif tracks_df.loc[parent_id, 'c_2'] > tracks_df.loc[id, 'c_2'] + args.beta or \
                    tracks_df.loc[parent_id, 'c_2'] < tracks_df.loc[id, 'c_2'] - args.beta:
                cut_track = True
            if cut_track:
                track_id = tracks_df.loc[id, 'track_id']
                parent_track_id = tracks_df.loc[parent_id, 'track_id']

                rows_to_update = tracks_df.index >= id
                rows_to_update &= tracks_df['track_id'] == track_id
                tracks_df.loc[rows_to_update, 'parent_track_id'] = -1

                tracks_df.loc[id, 'parent_id'] = -1

    tracks_df.to_pickle(join(output_dir, args.config_id, 'tracks_ppc.pkl'))


    # todo merge tracks where one branch of a split was removed