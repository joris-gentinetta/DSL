import os
import time
# os.environ["OMP_NUM_THREADS"] = "10"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
start = time.time()
from pathlib import Path
import numpy as np
from ultrack.utils.array import array_apply
from stardist.models import StarDist2D, Config2D
from utils import predict_stardist, watershed_segm

def segment(folder_path, ws=True, RESCALE=False):
    data_dir = Path(folder_path)
    normalized_path = data_dir / "normalized.npy"
    stardist_path = data_dir / "stardist_labels.npy"
    wssd_path = data_dir / "wssd_labels.npy"

    normalized = np.load(normalized_path)

    stardist_labels = np.zeros(normalized.shape, dtype=np.uint16)

    print("Initializing Stardist Model")
    stardist_config = Config2D(
        axes='YX',
        n_channel_in=3,
    )
    model = StarDist2D(stardist_config).from_pretrained("2D_versatile_fluo")
    print("Starting Stardist Segmentation")
    array_apply(
        normalized, 
        out_array=stardist_labels,
        func=predict_stardist,
        model=model,
        axis=(0, 3)
    )
    np.save(stardist_path, stardist_labels)

    if ws:
        wssd_labels = np.zeros(normalized.shape, dtype=np.uint16)
        # ws_labels = create_zarr(imgs.shape, np.uint16, ws_path, chunks = chunks)
        print("Starting watershed...")
        array_apply(
            normalized,
            stardist_labels,
            out_array=wssd_labels,
            func=watershed_segm,
            min_area=20,
            axis=(0, 3),
        )
        np.save(wssd_path, wssd_labels)

