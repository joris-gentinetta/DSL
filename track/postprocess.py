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
from ultrack.tracks.graph import inv_tracks_df_forest


def compute_color_composition(folder_path, img_path, config_id, n_frames=None):
    data_dir = Path(folder_path)
    normalized_path = data_dir / "normalized.npy"
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
    labels = np.load(track_label_path)

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

    # # multithreading (doesn't work)
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


def filter_color_composition(tracks_df, labels, beta=0.3):
    new_track_id = tracks_df['track_id'].max() + 1
    for id in tqdm(tracks_df.index):
        parent_id = tracks_df.loc[id, 'parent_id']
        t = tracks_df.loc[id, 't']
        if not parent_id == -1:
            cut_track = False
            if tracks_df.loc[parent_id, 'c_0'] > tracks_df.loc[id, 'c_0'] + beta or \
                    tracks_df.loc[parent_id, 'c_0'] < tracks_df.loc[id, 'c_0'] - beta:
                cut_track = True
            elif tracks_df.loc[parent_id, 'c_1'] > tracks_df.loc[id, 'c_1'] + beta or \
                    tracks_df.loc[parent_id, 'c_1'] < tracks_df.loc[id, 'c_1'] - beta:
                cut_track = True
            elif tracks_df.loc[parent_id, 'c_2'] > tracks_df.loc[id, 'c_2'] + beta or \
                    tracks_df.loc[parent_id, 'c_2'] < tracks_df.loc[id, 'c_2'] - beta:
                cut_track = True
            if cut_track:
                track_id = tracks_df.loc[id, 'track_id']
                parent_id = tracks_df.loc[id, 'parent_id']
                if tracks_df.loc[parent_id, 'track_id'] != track_id:
                    # we're cutting a whole branch of a split
                    # find the track_id's of all rows with t = t and parent_track_id = tracks_df.loc[parent_id, 'track_id']:
                    rows_to_update = tracks_df['t'] == t
                    rows_to_update &= tracks_df['parent_track_id'] == tracks_df.loc[parent_id, 'track_id']
                    branch_track_id = tracks_df.loc[rows_to_update, 'track_id'].unique()
                    #remove track_id from branch_track_id:
                    branch_track_id = branch_track_id[branch_track_id != track_id][0]
                    rows_to_update = tracks_df['track_id'] == branch_track_id
                    tracks_df.loc[rows_to_update, 'parent_track_id'] = tracks_df.loc[parent_id, 'parent_track_id']
                    tracks_df.loc[rows_to_update, 'track_id'] = tracks_df.loc[parent_id, 'track_id']

                    rows_to_update = tracks_df['parent_track_id'] == branch_track_id
                    tracks_df.loc[rows_to_update, 'parent_track_id'] = tracks_df.loc[parent_id, 'track_id']


                tracks_df.loc[id, 'parent_id'] = -1


                #set the track_ids of all rows in the track after the cut to a new track_id and set their parent_track_id to -1
                rows_to_update = tracks_df['t'] >= t
                rows_to_update &= tracks_df['track_id'] == track_id

                tracks_df.loc[rows_to_update, 'parent_track_id'] = -1
                tracks_df.loc[rows_to_update, 'track_id'] = new_track_id
                labels[t, :, :][labels[t, :, :] == track_id] = new_track_id

                #get the ids of rows_to_update and update all their sucessors:
                ids_to_update = tracks_df.loc[rows_to_update].index
                #get all rows with parent_id in ids_to_update:
                rows_to_update = tracks_df['parent_id'].isin(ids_to_update)
                #get all track_ids of these rows:
                track_ids_to_update = tracks_df.loc[rows_to_update, 'track_id'].unique()
                #remove new_track_id from track_ids_to_update:
                track_ids_to_update = track_ids_to_update[track_ids_to_update != new_track_id]
                #get all rows with track_id in track_ids_to_update:
                rows_to_update = tracks_df['track_id'].isin(track_ids_to_update)
                #update parent_track_id:
                tracks_df.loc[rows_to_update, 'parent_track_id'] = new_track_id

                new_track_id += 1


    return tracks_df, labels

def get_subtree(track_id, tracks_df):
    children = tracks_df.loc[tracks_df['parent_track_id'] == track_id, 'track_id'].unique()
    if children.size == 0:
        subtree = {track_id}
    else:
        subtree = set.union(*[get_subtree(child, tracks_df) for child in children])
        subtree.add(track_id)
    return subtree

def prune_short_tracks(tracks_df, labels, min_length=20):
    root_nodes = tracks_df.loc[tracks_df['parent_track_id']==-1, 'track_id'].unique()
    for root_node in tqdm(root_nodes):
        start_time = tracks_df.loc[tracks_df['track_id'] == root_node, 't'].min()
        tree = get_subtree(root_node, tracks_df)
        #get the max t of all rows where track_id in leaves:
        end_time = tracks_df.loc[tracks_df['track_id'].isin(tree), 't'].max()
        if end_time - start_time < min_length:
            #remove all rows where track_id in tree:
            rows_to_update = tracks_df['track_id'].isin(tree)
            tracks_df = tracks_df[~rows_to_update]
            for track_id in tree:
                labels[labels == track_id] = 0
            #remove all track_ids in tree from labels:

    return tracks_df, labels







def filter_division_time(tracks_df, min_division_time=10):
    return tracks_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display images and optional overlays.')
    parser.add_argument('--file', type=str, default="4T1 p27 trial period.HTD - Well D02 Field #3.tif",
                        help='Path to the image file')
    parser.add_argument('--config_id', type=str, default="100", required=False, help='Name of config file')

    parser.add_argument('--n_frames', type=int, default=100, help='Number of frames (optional)')
    parser.add_argument('--beta', type=float, default=0.2, help='Max color composition difference')
    # parser.add_argument('--min_division_time', type=int, default=10, help='Min division time')
    args = parser.parse_args()
    # args.file = 'demo.tif'

    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent.parent, "output", experiment)
    input_file = join(Path(__file__).parent.parent, "input", args.file)


    if not os.path.exists(join(output_dir, args.config_id, 'tracks_pp.parquet')):
        tracks_df = compute_color_composition(output_dir, input_file, args.config_id, args.n_frames)
        tracks_df.to_parquet(join(output_dir, args.config_id, 'tracks_pp.parquet'))
    else:
        tracks_df = pd.read_parquet(join(output_dir, args.config_id, 'tracks_pp.parquet'))

    labels = np.load(join(output_dir, args.config_id, 'track_labels.npy'))
    if not os.path.exists(join(output_dir, args.config_id, 'tracks_ppc.pkl')):
        tracks_df, labels = filter_color_composition(tracks_df, labels, args.beta)
        # tracks_df = filter_division_time(tracks_df, args.min_division_time)

        tracks_df.to_pickle(join(output_dir, args.config_id, 'tracks_ppc.pkl'))
        np.save(join(output_dir, args.config_id, 'track_labels_ppc.npy'), labels)
        graph = inv_tracks_df_forest(tracks_df)
        with open(join(output_dir, args.config_id, 'graph_ppc.pkl'), 'wb') as f:
            pickle.dump(graph, f)
    else:
        tracks_df = pd.read_pickle(join(output_dir, args.config_id, 'tracks_ppc.pkl'))
        labels = np.load(join(output_dir, args.config_id, 'track_labels_ppc.npy'))

    tracks_df, labels = prune_short_tracks(tracks_df, labels, 50)
    graph = inv_tracks_df_forest(tracks_df)
    tracks_df.to_pickle(join(output_dir, args.config_id, 'tracks_ppc_pruned.pkl'))
    np.save(join(output_dir, args.config_id, 'track_labels_ppc_pruned.npy'), labels)
    with open(join(output_dir, args.config_id, 'graph_ppc_pruned.pkl'), 'wb') as f:
        pickle.dump(graph, f)




