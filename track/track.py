# stardist / tensorflow env variables setup
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

def tracking(output_dir, n_frames=-1, override=False):
    start_time = time()
    data_dir = Path(output_dir)
    normalized_path = data_dir / "normalized.npy"
    cellpose_path = data_dir / "cellpose_labels.npy"
    wscp_path = data_dir / "wscp_labels.npy"
    stardist_path = data_dir / "stardist_labels.npy"
    wssd_path = data_dir / "wssd_labels.npy"
    detection_path = data_dir / "detections.npz"

    cellpose_labels = da.from_array(np.load(cellpose_path))
    wscp_labels = da.from_array(np.load(wscp_path))

    stardist_labels = da.from_array(np.load(stardist_path))
    wssd_labels = da.from_array(np.load(wssd_path))

    if not detection_path.exists() or override:
        detection, edges = labels_to_edges(
            [stardist_labels[..., c] for c in range(stardist_labels.shape[-1])] +\
            [cellpose_labels[..., c] for c in range(cellpose_labels.shape[-1])],
            # [wscp_labels[..., c] for c in range(wscp_labels.shape[-1])] +\
            # [wssd_labels[..., c] for c in range(wssd_labels.shape[-1])],
            sigma=1.0,
            detection_store_or_path=zarr.TempStore(),
            edges_store_or_path=zarr.TempStore(),
        )
        np.savez_compressed(detection_path, detection=detection, edges=edges)
    else:
        detection, edges = np.load(detection_path)

    config = MainConfig()
    pprint(config)

    params_df = estimate_parameters_from_labels(stardist_labels, is_timelapse=True)
    params_df.to_csv(join(data_dir, 'params.csv'))
    # params_df["area"].plot(kind="hist", bins=100, title="Area histogram")

    config.segmentation_config.min_area = 50
    config.segmentation_config.max_area = 200
    config.segmentation_config.n_workers = 40

    config.linking_config.max_distance = 10
    config.linking_config.n_workers = 40

    config.tracking_config.appear_weight = -1
    config.tracking_config.disappear_weight = -1
    config.tracking_config.division_weight = -0.1
    config.tracking_config.power = 4
    config.tracking_config.bias = -0.001
    config.tracking_config.solution_gap = 0.003 #todo

    pprint(config)

    track(
        detection=detection,
        edges=edges,
        config=config,
        overwrite=True,
    )

    tracks_df, graph = to_tracks_layer(config)
    labels = tracks_to_zarr(config, tracks_df)
    pd.to_pickle(tracks_df, join(data_dir, 'tracks.pkl'))
    with open(join(data_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)
    np.save(join(data_dir, 'track_labels.npy'), labels)

    end_time = time()
    print(f"Total time: {(end_time - start_time)/60} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, default="4T1 p27 trial period.HTD - Well D02 Field #3.tif" , required=False, help='Path to the image file')
    parser.add_argument('--n_frames', type=int, default=-1, required=False, help='Number of frames (optional)')
    parser.add_argument('--override', default=True, required=False, action='store_true', help='Override existing files')
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "40"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # create the folder to store the results:
    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent.parent, "output", experiment)
    tracking(output_dir, args.n_frames, args.override)