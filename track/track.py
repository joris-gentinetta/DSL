import os
from pathlib import Path
from os.path import join
import numpy as np
from rich.pretty import pprint
import pickle
from ultrack import track, to_tracks_layer, tracks_to_zarr
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
from ultrack.config import MainConfig
import pandas as pd
from time import time
import argparse
import dask.array as da
import zarr
import json

def tracking(output_dir, config_id,  n_frames=-1, override=False):
    data_dir = Path(output_dir)
    cellpose_path = data_dir / "cellpose_labels.npy"
    wscp_path = data_dir / "wscp_labels.npy"
    stardist_path = data_dir / "stardist_labels.npy"
    wssd_path = data_dir / "wssd_labels.npy"
    detection_path = data_dir / config_id / "detections.npz"



    cellpose_labels = da.from_array(np.load(cellpose_path))[:n_frames]
    wscp_labels = da.from_array(np.load(wscp_path))[:n_frames]

    stardist_labels = da.from_array(np.load(stardist_path))[:n_frames]
    wssd_labels = da.from_array(np.load(wssd_path))[:n_frames]

    with open(join('configs', f'{config_id}.json'), 'r') as f:
        config_data = json.load(f)
    segmentation_channels = config_data["segmentation_channels"]
    if not detection_path.exists() or override:
        detection, edges = labels_to_edges(
            [stardist_labels[..., c] for c in segmentation_channels['stardist']] +\
            [cellpose_labels[..., c] for c in segmentation_channels['cellpose']] +\
            [wscp_labels[..., c] for c in segmentation_channels['wscp']] +\
            [wssd_labels[..., c] for c in segmentation_channels['wssd']],
            sigma=1.0,
            detection_store_or_path=zarr.TempStore(),
            edges_store_or_path=zarr.TempStore(),
        )
        np.savez_compressed(detection_path, detection=detection, edges=edges)
    else:
        de = np.load(detection_path)
        detection, edges = de['detection'], de['edges']

    config = MainConfig()

    # params_df = estimate_parameters_from_labels(stardist_labels, is_timelapse=True)
    # params_df.to_csv(join(data_dir, 'params.csv'))
    # params_df["area"].plot(kind="hist", bins=100, title="Area histogram")


    for key in config_data:
        if hasattr(config, key):
            for sub_key in config_data[key]:
                if hasattr(getattr(config, key), sub_key):
                    setattr(getattr(config, key), sub_key, config_data[key][sub_key])

    pprint(config)

    track(
        detection=detection,
        edges=edges,
        config=config,
        overwrite=True,
    )

    tracks_df, graph = to_tracks_layer(config)
    labels = tracks_to_zarr(config, tracks_df)
    pd.to_pickle(tracks_df, join(data_dir, config_id, 'tracks.pkl'))
    with open(join(data_dir, config_id, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)
    np.save(join(data_dir, config_id, 'track_labels.npy'), labels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, default="demo.tif" , required=False, help='Name of file')
    parser.add_argument('--config_id', type=str, default="2" , required=False, help='Name of config file')

    parser.add_argument('--n_frames', type=int, default=None, required=False, help='Number of frames (optional)')
    parser.add_argument('--override', default=True, required=False, action='store_true', help='Override existing files')
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "40"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # create the folder to store the results:
    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent.parent, "output", experiment)

    for config_id in [100]:
        args.config_id = str(config_id)
        os.makedirs(join(output_dir, args.config_id), exist_ok=True)
        start_time = time()
        tracking(output_dir, args.config_id, args.n_frames, args.override)
        end_time = time()
        print(f"{config_id}: {(end_time - start_time) / 60} minutes")