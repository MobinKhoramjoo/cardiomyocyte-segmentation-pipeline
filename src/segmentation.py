"""Single-image segmentation utilities.

The segmentation only uses two channels:

  - DAPI  -> nuclei (used as seeds)
  - WGA   -> cell borders (used to define the cytoplasm shape)

Any additional immunofluorescence channels in the source file are ignored
here -- they live in the .lif file but are not needed for segmentation.

The pipeline is:

  1. Pull a 2D plane out of the z-stack (max projection by default).
  2. Optionally clean up each channel with an Otsu or manual threshold.
  3. Run Cellpose on the (WGA, DAPI) pair to get instance masks.
  4. Drop cells that touch the image border (they're truncated and would
     bias the area distribution).
  5. Drop very small objects below a physical-area cutoff in microns squared,
     which removes debris and Cellpose splinters.

All public functions accept plain numpy arrays so this module is easy to
test in isolation and reuse in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from cellpose import models
from skimage import exposure, filters, measure
from skimage.segmentation import clear_border


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------

@dataclass
class SegmentationConfig:
    """Knobs that control a single-image segmentation run.

    Defaults are tuned for 63x Leica confocal stacks with ~0.18 um/px and
    cardiomyocytes that are typically a couple of hundred pixels across.
    Adjust ``diameter`` and ``min_area_um2`` if your acquisition differs.
    """

    # Channel indices inside the .lif file
    dapi_channel: int = 0
    wga_channel: int = 2

    # Z-stack collapsing strategy: "max", "mid", or an int z-index.
    z_strategy: str | int = "max"

    # Per-channel pre-cleaning. -1 = automatic Otsu, 0 = no threshold,
    # any positive value is used as a hard intensity cutoff.
    wga_threshold: float = -1
    dapi_threshold: float = -1

    # Cellpose settings. ``diameter`` is in pixels (Cellpose requirement).
    # ``flow_threshold`` is forgiving so faint or irregular cells are kept.
    # ``cellprob_threshold`` is low so dim cells aren't dropped early.
    diameter: float = 200
    flow_threshold: float = 0.8
    cellprob_threshold: float = -2.0
    model_type: Optional[str] = "cyto2"
    use_gpu: bool = True

    # Minimum physical area in um^2. Anything smaller is treated as debris.
    min_area_um2: float = 160.0


# ---------------------------------------------------------------------------
# Z-stack handling
# ---------------------------------------------------------------------------

def get_z_projection(lif_img, channel_idx: int, strategy: str | int = "max") -> np.ndarray:
    """Collapse a z-stack of one channel into a single 2D image.

    ``lif_img`` is a single image returned by :class:`readlif.reader.LifFile`.
    If the image only has one z-plane we just return it. Otherwise we pull
    every slice for that channel and combine them according to ``strategy``.
    """
    z_depth = lif_img.dims.z
    if z_depth <= 1:
        return np.array(lif_img.get_frame(z=0, t=0, c=channel_idx))

    stack = [np.array(lif_img.get_frame(z=z, t=0, c=channel_idx))
             for z in range(z_depth)]
    volume = np.array(stack)

    if strategy == "max":
        return np.max(volume, axis=0)
    if strategy == "mid":
        return volume[z_depth // 2]
    if isinstance(strategy, int):
        return volume[strategy]
    return volume[0]


def apply_threshold(image: np.ndarray, thresh_val: float) -> np.ndarray:
    """Optionally pre-clean a channel before sending it to Cellpose.

    -1 picks an Otsu threshold automatically; any positive value is used
    as a hard cutoff; 0 leaves the image untouched.
    """
    if thresh_val == -1:
        val = filters.threshold_otsu(image)
        return image * (image > val)
    if thresh_val > 0:
        return image * (image > thresh_val)
    return image


# ---------------------------------------------------------------------------
# Visualisation helper (used by the storyboard plot, kept here for reuse)
# ---------------------------------------------------------------------------

def create_custom_rgb(wga: np.ndarray, dapi: np.ndarray,
                      wga_color: str = "grey", dapi_color: str = "blue") -> np.ndarray:
    """Build a display RGB image from the two segmentation channels.

    This is purely for visualisation -- it doesn't feed Cellpose. Each
    channel is independently rescaled to its 99th percentile so that bright
    flecks don't wash out the image.
    """

    def _norm(arr: np.ndarray) -> np.ndarray:
        if arr.max() == 0:
            return arr.astype(np.uint8)
        p_hi = np.percentile(arr, 99) or arr.max()
        arr = exposure.rescale_intensity(arr, in_range=(0, p_hi))
        return (arr * 255).astype(np.uint8)

    h, w = wga.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    col_map = {"red": 0, "green": 1, "blue": 2}

    def _add(channel: np.ndarray, color: str) -> None:
        norm_data = _norm(channel)
        if color in ("grey", "gray"):
            for c in range(3):
                rgb[..., c] = np.maximum(rgb[..., c], norm_data)
        elif color in col_map:
            idx = col_map[color]
            rgb[..., idx] = np.maximum(rgb[..., idx], norm_data)

    _add(wga, wga_color)
    _add(dapi, dapi_color)
    return rgb


# ---------------------------------------------------------------------------
# Cellpose wrapper
# ---------------------------------------------------------------------------

def _build_model(cfg: SegmentationConfig):
    """Instantiate a CellposeModel. Kept as a small helper so callers can
    reuse a single model across many images instead of reloading it."""
    if cfg.model_type:
        return models.CellposeModel(gpu=cfg.use_gpu, model_type=cfg.model_type)
    return models.CellposeModel(gpu=cfg.use_gpu)


def segment_image(wga: np.ndarray, dapi: np.ndarray,
                  cfg: SegmentationConfig,
                  pixel_size_um: float,
                  model=None) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """Run the full single-image pipeline.

    Parameters
    ----------
    wga, dapi : np.ndarray
        Already z-projected 2D images. Same shape, same dtype.
    cfg : SegmentationConfig
        Knobs for thresholding, Cellpose, and the size filter.
    pixel_size_um : float
        Microns per pixel. Used to convert raw pixel area to um^2.
    model : optional
        A pre-loaded Cellpose model, useful for batch processing so we
        don't reload the weights for every image.

    Returns
    -------
    masks : np.ndarray
        Final integer label image (0 = background) after edge clearing
        and the size filter.
    cells_df : pd.DataFrame
        One row per kept cell with columns ``label`` and ``area_um2``.
    stats : dict
        Counts at each filtering step plus the mean area, suitable for
        logging or accumulating into a results CSV.
    """
    # Pre-clean each channel.
    clean_wga = apply_threshold(wga, cfg.wga_threshold)
    clean_dapi = apply_threshold(dapi, cfg.dapi_threshold)

    # Cellpose expects an HxWxC stack with channel 1 = cyto, 2 = nuclei.
    img_stack = np.stack([clean_wga, clean_dapi], axis=-1)

    if model is None:
        model = _build_model(cfg)

    masks, _flows, _styles = model.eval(
        img_stack,
        diameter=cfg.diameter,
        channels=[1, 2],
        flow_threshold=cfg.flow_threshold,
        cellprob_threshold=cfg.cellprob_threshold,
    )

    # Counts before any post-filtering, useful for QC.
    initial_count = int(len(np.unique(masks)) - 1)

    # Drop cells that touch the image border -- they're truncated and would
    # skew the area distribution downward.
    masks = clear_border(masks)
    post_edge_count = int(len(np.unique(masks)) - 1)
    edge_dropped = initial_count - post_edge_count

    # Size filter in physical units (um^2), which makes the threshold
    # comparable across acquisitions with different pixel sizes.
    props = measure.regionprops_table(masks, properties=["label", "area"])
    df = pd.DataFrame(props)

    filtered_masks = np.zeros_like(masks)
    if not df.empty:
        df["area_um2"] = df["area"] * (pixel_size_um ** 2)
        df_kept = df[df["area_um2"] >= cfg.min_area_um2]
        valid_labels = df_kept["label"].values
        keep = np.isin(masks, valid_labels)
        filtered_masks[keep] = masks[keep]
        cells_df = df_kept[["label", "area_um2"]].reset_index(drop=True)
        mean_area = float(cells_df["area_um2"].mean()) if len(cells_df) else 0.0
    else:
        cells_df = pd.DataFrame(columns=["label", "area_um2"])
        mean_area = 0.0

    final_count = len(cells_df)
    size_dropped = post_edge_count - final_count

    stats = {
        "initial_count": initial_count,
        "edge_dropped": edge_dropped,
        "size_dropped": size_dropped,
        "cell_count": final_count,
        "mean_area_um2": mean_area,
    }
    return filtered_masks, cells_df, stats
