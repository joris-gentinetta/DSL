import os
import time

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Starting cellpose imports...")
start = time.time()
from pathlib import Path

import numpy as np

from ultrack.utils.array import array_apply
from ultrack.utils.cuda import torch_default_device
from ultrack.imgproc.segmentation import Cellpose

from utils import watershed_segm
print(f"Finished imports in {time.time() - start} seconds")

def segment(folder_path, RESCALE=False):
    data_dir = Path(folder_path)
    img_path = data_dir / "raw.tif"
    normalized_path = data_dir / "normalized.npy"
    cellpose_path = data_dir / "cellpose_labels.npy"
    wscp_path = data_dir / "wscp_labels.npy"

    normalized = np.load(normalized_path)

    cellpose_labels = np.zeros(normalized.shape, dtype=np.uint16)
    # cellpose_labels = create_zarr(imgs.shape, np.uint16, cellpose_path, chunks = chunks)
    print("Initializing Cellpose model")
    model = Cellpose(model_type = 'nuclei', gpu=True, device=torch_default_device())
    print("Starting Cellpose Segmentation")
    array_apply(
        normalized, 
        out_array=cellpose_labels,
        func=model,
        axis=(0, 3),
        tile=False,
        diameter=(8 if RESCALE else 16),
        min_size=-1,
        flow_threshold=0.6, 
        batch_size=2,
        normalize=False,
    )
    np.save(cellpose_path, cellpose_labels)
    
    # wscp_labels = np.zeros(normalized.shape, dtype=np.uint16)
    # # ws_labels = create_zarr(imgs.shape, np.uint16, ws_path, chunks = chunks)
    # print("Starting watershed...")
    # array_apply(
    #     normalized,
    #     cellpose_labels,
    #     out_array=wscp_labels,
    #     func=watershed_segm,
    #     min_area=20,
    #     axis=(0, 3),
    # )
    # np.save(wscp_path, wscp_labels)