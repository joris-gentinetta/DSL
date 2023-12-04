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

if __name__ == "__main__":
    start_time = time()

    # data_dir = "/cluster/scratch/jorisg/data"
    data_dir = join(Path(__file__).parent.parent, "data")

    os.environ["OMP_NUM_THREADS"] = "40"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    stardist_labels = np.load(join(data_dir, 'stardist_labels.npy'))
    detection, edges = labels_to_edges(stardist_labels,
                                       sigma=4.0)  # multiple labels can be used with [labels_0, labels_1, ...]
    np.save(join(data_dir, 'detection.npy'), detection)
    np.save(join(data_dir, 'edges.npy'), edges)
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
    np.save(join(data_dir, 'labels.npy'), labels)

    end_time = time()
    print(f"Total time: {(end_time - start_time)/60} minutes")

