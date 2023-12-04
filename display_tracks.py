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
from tifffile import imread

#rsync to local using os:
data_dir = join(Path(__file__).parent.parent, "data")
# os.system('rsync -avz --progress jorisg@euler.ethz.ch:/cluster/scratch/jorisg/data/ data/')
os.system(f'rsync -avz --progress jorisg@192.168.1.203:/home/jorisg/ultrack/examples/data/ {data_dir}/')

# (T, Y, X, C) data, where T=time, Y, X =s patial coordinates and C=channels
data_dir = join(Path(__file__).parent.parent, "data")

img_path = Path(join(data_dir, "test.tif"))

n_frames = 200

params_df = pd.read_csv(join(data_dir, 'params.csv'), index_col=0)
params_df["area"].plot(kind="hist", bins=100, title="Area histogram")
plt.show()
imgs = imread(img_path)
imgs = imgs[:, 1:, :, :]

# chunks = (1, *imgs.shape[1:-1], 1) # chunk size used to compress data

if n_frames is not None:
    imgs = imgs[:n_frames]
imgs = np.swapaxes(imgs, 1, 3)

# dataset_path = Path("Fluo-N2DL-HeLa/01")


viewer = napari.Viewer()
viewer.window.resize(1800, 1000)
# viewer.open(sorted(dataset_path.glob("*.tif")), stack=True)
layers = viewer.add_image(imgs, channel_axis=3, name="raw")

stardist_labels = np.load(join(data_dir, 'stardist_labels.npy'))
viewer.add_labels(stardist_labels, name="stardist")

detection = np.load(join(data_dir, 'detection.npy'))
edges = np.load(join(data_dir, 'edges.npy'))
viewer.add_image(detection, visible=False)
viewer.add_image(edges, blending="additive", colormap="magma")

tracks_df = pd.read_pickle(join(data_dir, 'tracks.pkl'))
with open(join(data_dir, 'graph.pkl'), 'rb') as f:
    graph = pickle.load(f)
labels = np.load(join(data_dir, 'labels.npy'))
viewer.add_tracks(tracks_df[["track_id", "t", "y", "x"]].values, graph=graph)
viewer.add_labels(labels)

viewer.layers["stardist"].visible = False
viewer.layers["edges"].visible = False
napari.run()
