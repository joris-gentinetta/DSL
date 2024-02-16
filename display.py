# stardist / tensorflow env variables setup
import os
from os.path import join

# os.environ["OMP_NUM_THREADS"] = "10"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


# from dask_image.imread import imread

def show_area_hist(params_file_path):
    params_df = pd.read_csv(params_file_path, index_col=0)
    params_df["area"].plot(kind="hist", bins=100, title="Area histogram")
    plt.show()


def initialize_viewer(imgs):
    viewer = napari.Viewer()
    viewer.window.resize(1800, 1000)
    layers = viewer.add_image(imgs, channel_axis=3, name="raw")

    return viewer


def add_detection(viewer, detection_file_path):
    full_detection = np.load(detection_file_path)
    detection_name = f"{Path(detection_file_path).name}_detection"
    edges_name = f"{Path(detection_file_path).name}_edges"
    viewer.add_image(full_detection['detection'], visible=False, name=detection_name)
    viewer.add_image(full_detection['edges'], blending="additive", colormap="magma", name=edges_name)

    return detection_name, edges_name


def add_tracks(viewer, tracks_file_path, track_label_file_path, graph_file_path):
    tracks_df = pd.read_pickle(tracks_file_path)
    track_name = Path(tracks_file_path).name
    with open(graph_file_path, 'rb') as f:
        graph = pickle.load(f)
    labels = np.load(track_label_file_path)
    track_label_name = Path(track_label_file_path).name
    layer = viewer.add_tracks(tracks_df[["track_id", "t", "y", "x"]].values, graph=graph, name=track_name)

    layer.visible = False
    viewer.add_labels(labels, name=track_label_name)

    return track_name, track_label_name


def display_folder(folder_path, img_path, config_id, n_frames=None):
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

    tracks_ppc_path = data_dir / config_id / "tracks_ppc.pkl"
    graph_ppc_path = data_dir / config_id / "graph_ppc.pkl"

    tracks_ppc_pruned_path = data_dir / config_id / "tracks_ppc_pruned.pkl"
    graph_ppc_pruned_path = data_dir / config_id / "graph_ppc_pruned.pkl"
    track_label_ppc_pruned_path = data_dir / config_id / "track_labels_ppc_pruned.npy"


    imgs = imread(img_path)

    imgs = imgs[:, 1:, :, :]

    if n_frames is not None:
        imgs = imgs[:n_frames]
    imgs = np.swapaxes(imgs, 1, 3)

    chunks = (1, imgs.shape[1], imgs.shape[2], 1)  # chunk size used to compress input

    viewer = initialize_viewer(imgs)

    if normalized_path.exists():
        # image size is (t, y, x, c)
        normalized = da.from_array(np.load(normalized_path), chunks=chunks)
        layers = viewer.add_image(normalized, rgb=False, channel_axis=3, name="normalized")
        for layer in layers:
            layer.visible = False

    if cellpose_path.exists():
        cellpose_labels = da.from_array(np.load(cellpose_path), chunks=chunks)
        for i in range(cellpose_labels.shape[-1]):
            layer = viewer.add_labels(cellpose_labels[:, :, :, i], name=f"cellpose labels {i}")
            layer.visible = False


    if wscp_path.exists():
        wscp_labels = da.from_array(np.load(wscp_path), chunks=chunks)
        for i in range(wscp_labels.shape[-1]):
            layer = viewer.add_labels(wscp_labels[:, :, :, i], name=f"watershed with cellpose labels {i}")
            layer.visible = False

    if stardist_path.exists():
        stardist_labels = da.from_array(np.load(stardist_path), chunks=chunks)
        for i in range(stardist_labels.shape[-1]):
            layer = viewer.add_labels(stardist_labels[:, :, :, i], name=f"stardist labels {i}")
            layer.visible = False

    if wssd_path.exists():
        wssd_labels = da.from_array(np.load(wssd_path), chunks=chunks)
        for i in range(wssd_labels.shape[-1]):
            layer = viewer.add_labels(wssd_labels[:, :, :, i], name=f"watershed with stardist labels {i}")
            layer.visible = False

    if detection_path.exists():
        detection_name, edges_name = add_detection(viewer, detection_path)
        viewer.layers[edges_name].visible = False

    if tracks_path.exists() and graph_path.exists():
        track_name, track_label_name = add_tracks(viewer, tracks_path, track_label_path, graph_path)
    if tracks_ppc_path.exists() and graph_ppc_path.exists():
        track_name, track_label_name = add_tracks(viewer, tracks_ppc_path, track_label_path, graph_ppc_path)
    if tracks_ppc_pruned_path.exists() and graph_ppc_pruned_path.exists():
        track_name, track_label_name = add_tracks(viewer, tracks_ppc_pruned_path, track_label_ppc_pruned_path, graph_ppc_pruned_path)

    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display images and optional overlays.')
    parser.add_argument('--file', type=str, default="demo.tif",
                        help='Path to the image file')
    parser.add_argument('--config_id', type=str, default="high_quality", required=False, help='Name of config file')
    parser.add_argument('--n_frames', type=int, default=None, help='Number of frames (optional)')
    args = parser.parse_args()


    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent, "output", experiment)
    input_file = join(Path(__file__).parent, "input", args.file)

    display_folder(output_dir, input_file, str(args.config_id), args.n_frames)

