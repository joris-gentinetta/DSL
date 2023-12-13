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
    # layer = viewer.add_tracks(tracks_df[["track_id", "t", "y", "x"]].values, graph=graph, name=track_name)
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
        # if normalized.shape != imgs.shape:
        #     rescale_factor = imgs.shape[1] / normalized.shape[1]
        #     upscaled = np.zeros(imgs.shape, dtype=normalized.dtype)
        #     for i in range(imgs.shape[0]):
        #         upscaled[i, ...] = rescale(normalized[i, ...].compute(), rescale_factor, channel_axis=2,
        #                                    anti_aliasing=True, preserve_range=True)
        #     normalized = da.from_array(upscaled, chunks=chunks)

        layers = viewer.add_image(normalized, rgb=False, channel_axis=3, name="normalized")
        for layer in layers:
            layer.visible = False

    if cellpose_path.exists():
        cellpose_labels = da.from_array(np.load(cellpose_path), chunks=chunks)
        # if cellpose_labels.shape != imgs.shape:
        #     rescale_factor = imgs.shape[1] / cellpose_labels.shape[1]
        #     upscaled = np.zeros(imgs.shape, dtype=cellpose_labels.dtype)
        #     for i in range(imgs.shape[0]):
        #         upscaled[i, ...] = rescale(cellpose_labels[i, ...].compute(), rescale_factor, channel_axis=2,
        #                                    anti_aliasing=False, preserve_range=True)
        #     cellpose_labels = da.from_array(upscaled, chunks=chunks)
        for i in range(cellpose_labels.shape[-1]):
            layer = viewer.add_labels(cellpose_labels[:, :, :, i], name=f"cellpose labels {i}")
            layer.visible = False


    if wscp_path.exists():
        wscp_labels = da.from_array(np.load(wscp_path), chunks=chunks)
        # if wscp_labels.shape != imgs.shape:
        #     rescale_factor = imgs.shape[1] / wscp_labels.shape[1]
        #     upscaled = np.zeros(imgs.shape, dtype=wscp_labels.dtype)
        #     for i in range(imgs.shape[0]):
        #         upscaled[i, ...] = rescale(wscp_labels[i, ...].compute(), rescale_factor, channel_axis=2,
        #                                    anti_aliasing=False, preserve_range=True)
        #     wscp_labels = da.from_array(upscaled, chunks=chunks)
        for i in range(wscp_labels.shape[-1]):
            layer = viewer.add_labels(wscp_labels[:, :, :, i], name=f"watershed with cellpose labels {i}")
            layer.visible = False

    if stardist_path.exists():
        stardist_labels = da.from_array(np.load(stardist_path), chunks=chunks)
        # if stardist_labels.shape != imgs.shape:
        #     rescale_factor = imgs.shape[1] / stardist_labels.shape[1]
        #     upscaled = np.zeros(imgs.shape, dtype=stardist_labels.dtype)
        #     for i in range(imgs.shape[0]):
        #         upscaled[i, ...] = rescale(stardist_labels[i, ...].compute(), rescale_factor, channel_axis=2,
        #                                    anti_aliasing=False, preserve_range=True)
        #     stardist_labels = da.from_array(upscaled, chunks=chunks)
        for i in range(stardist_labels.shape[-1]):
            layer = viewer.add_labels(stardist_labels[:, :, :, i], name=f"stardist labels {i}")
            layer.visible = False

    if wssd_path.exists():
        wssd_labels = da.from_array(np.load(wssd_path), chunks=chunks)
        # if wssd_labels.shape != imgs.shape:
        #     rescale_factor = imgs.shape[1] / wssd_labels.shape[1]
        #     upscaled = np.zeros(imgs.shape, dtype=wssd_labels.dtype)
        #     for i in range(imgs.shape[0]):
        #         upscaled[i, ...] = rescale(wssd_labels[i, ...].compute(), rescale_factor, channel_axis=2,
        #                                    anti_aliasing=False, preserve_range=True)
        #     wssd_labels = da.from_array(upscaled, chunks=chunks)
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
    # screenshot = viewer.screenshot()
    # viewer.close()
    # return screenshot

    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display images and optional overlays.')
    parser.add_argument('--file', type=str, default="4T1 p27 trial period.HTD - Well D02 Field #3.tif",
                        help='Path to the image file')
    parser.add_argument('--config_id', type=str, default="100", required=False, help='Name of config file')

    parser.add_argument('--n_frames', type=int, default=100, help='Number of frames (optional)')
    args = parser.parse_args()
    # args.file = 'demo.tif'

    experiment = Path(args.file).stem
    output_dir = join(Path(__file__).parent, "output", experiment)
    print(f'rsync -avz --progress jorisg@192.168.1.203:/home/jorisg/projects/DSL/output/ "{join(Path(__file__).parent, "output")}/"')
    os.system(
        f'rsync -avz --progress jorisg@192.168.1.203:/home/jorisg/projects/DSL/output/ {join(Path(__file__).parent, "output")}/')
    input_file = join(Path(__file__).parent, "input", args.file)

    display_folder(output_dir, input_file, str(args.config_id), args.n_frames)

    # # List of config IDs
    # config_ids = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56]
    # # config_ids = [1, 2, 3]
    #
    # # Determine the number of rows and columns for the grid
    #
    # n_cols = 6
    # n_rows = len(config_ids) // n_cols + (len(config_ids) % n_cols > 0)
    #
    # # Create a figure with specified dimensions
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))  # Adjust figsize as needed
    #
    # # Remove the space between plots
    # plt.subplots_adjust(wspace=0, hspace=0)
    #
    # # Iterate over the grid and plot
    # for i, config_id in tqdm(enumerate(config_ids)):
    #     row = i // n_cols
    #     col = i % n_cols
    #
    #     args.config_id = str(config_id)
    #
    #     screenshot = display_folder(output_dir, input_file, str(args.config_id), args.n_frames)
    #     axes[row, col].imshow(screenshot)
    #     axes[row, col].set_title(str(config_id), fontsize=10)
    #
    #     # Remove axes
    #     axes[row, col].axis('off')
    #
    # # Hide any unused subplots
    # for i in range(len(config_ids), n_rows * n_cols):
    #     row = i // n_cols
    #     col = i % n_cols
    #     axes[row, col].axis('off')
    #
    # # Save the figure
    # # plt.show()
    # plt.savefig('combined_screenshots.png', dpi=600)  # Adjust the filename and dpi as needed
    # #
    # # # Close the plot
    # # plt.close(fig)
