from numpy.typing import ArrayLike

import gc

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.ndimage as ndi

from ultrack.imgproc.segmentation import reconstruction_by_dilation
from ultrack.utils.cuda import import_module, to_cpu

from skimage.filters import threshold_otsu
from pyift.shortestpath import watershed_from_minima
from skimage.segmentation import relabel_sequential
from skimage import morphology as morph

try:
    import cupy as xp
except ImportError:
	import numpy as xp


# helper functions
def remove_background(image: ArrayLike, sigma=15.0) -> ArrayLike:
    """
    Removes background using morphological reconstruction by dilation.
    Reconstruction seeds are an extremely blurred version of the input.

    Parameters
    ----------
    imgs : ArrayLike
        Raw image.

    Returns
    -------
    ArrayLike
        Foreground image.
    """
    image = xp.asarray(image)
    ndi = import_module("scipy", "ndimage")
    seeds = ndi.gaussian_filter(image, sigma=sigma)
    background = reconstruction_by_dilation(seeds, image, iterations=50)
    foreground = np.maximum(image, background) - background
    return to_cpu(foreground)


def watershed_segm(
    frame: ArrayLike,
    aux_labels: ArrayLike,
    min_area: int,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Detects foreground using Otsu threshold and auxiliary labels,
    and execute watershed from minima inside that region.

    Parameters
    ----------
    frame : ArrayLike
        Images as an Y,X array.
    aux_labels : ArrayLike
        Auxiliary labels are used to detect the foreground.
    min_area : int
        Minimum size to be considered a cell.

    Returns
    -------
    ArrayLike
        Watershed segmentation labels.
    """
    disk3 = ndi.generate_binary_structure(frame.ndim, 3)

    frame = frame.astype(np.float32)
    # frame = ndi.gaussian_filter(frame, 3.0)
    det = frame > (threshold_otsu(frame) * 0.75)  # making otsu less conservative

    det = np.logical_or(det, np.asarray(aux_labels) > 0)

    det = morph.remove_small_objects(det, min_area)
    det = ndi.binary_closing(det, structure=disk3)

    edt = ndi.distance_transform_edt(det)
    labels = relabel_sequential(watershed_from_minima(-edt, det, H_minima=2.0)[1])[0]

    return labels


def plot_tracks(tracks_df: pd.DataFrame) -> None:
    """Center tracks at their initial position and plot them.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks datafarame sorted by `track_id` and `t`.

    Returns
    -------
    pd.DataFrame
        Centered dataframe.
    """
    centered_df = tracks_df.copy()
    centered_df[["y", "x"]] = centered_df.groupby(
        "track_id",
        as_index=False,
    )[["y", "x"]].transform(lambda x: x - x.iloc[0])

    # sanity check
    assert (centered_df[centered_df["t"] == 0][["y", "x"]] == 0).all().all()

    pallete = sns.color_palette(["gray"], len(centered_df["track_id"].unique()))
    sns.lineplot(
        data=centered_df,
        x="x",
        y="y",
        hue="track_id",
        palette=pallete,
        legend=False,
        alpha=0.5,
        sort=False,
        estimator=None,
    )

    return centered_df


def smooth(image: ArrayLike) -> ArrayLike:
    """
    Applies gaussian blur.

    Parameters
    ----------
    imgs : ArrayLike
        Raw image.

    Returns
    -------
    ArrayLike
        Smoothed image.
    """
    image = xp.asarray(image)
    ndi = import_module("scipy", "ndimage")
    # seeds = ndi.gaussian_filter(image, sigma=sigma)
    # background = reconstruction_by_dilation(seeds, image, iterations=100)
    # foreground = np.maximum(image, background) - background
    foreground = ndi.gaussian_filter(image, sigma=2.0)
    thresh = image.copy()
    thresh[foreground<0,7] = 0
    
    return to_cpu(thresh)

def predict_stardist(image: ArrayLike, model) -> ArrayLike:
	"""Normalizes and computes stardist prediction."""
	labels, _ = model.predict_instances(image)
    # labels, _ = model.predict_instances_big(
	# 	image, "YX", block_size=block_size, min_overlap=min_overlap, show_progress=False,
	# )
	gc.collect()
	return labels