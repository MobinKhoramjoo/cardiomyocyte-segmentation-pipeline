"""Storyboard visualisation for a single image.

Useful for sanity-checking a new acquisition or tweaking the Cellpose
parameters before launching a long batch. Shows the two input channels,
the merged RGB, the raw Cellpose detection, the size-filtered detection,
and the final outlines overlaid on the image.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from cellpose import plot, utils
from readlif.reader import LifFile

from .segmentation import (
    SegmentationConfig,
    apply_threshold,
    create_custom_rgb,
    get_z_projection,
    segment_image,
)


def visualize_pipeline(lif_path: str,
                       image_index: int = 0,
                       cfg: SegmentationConfig | None = None,
                       wga_color: str = "grey",
                       dapi_color: str = "blue",
                       figsize: tuple = (24, 16)):
    """Run the full pipeline on one image and plot every step.

    Returns the matplotlib Figure so it can be saved by the caller.
    """
    cfg = cfg or SegmentationConfig()

    reader = LifFile(lif_path)
    lif_img = reader.get_image(image_index)
    pixel_size_um = 1 / lif_img.scale[0]
    print(f"Pixel size: {pixel_size_um:.4f} um/px")

    raw_wga = get_z_projection(lif_img, cfg.wga_channel, cfg.z_strategy)
    raw_dapi = get_z_projection(lif_img, cfg.dapi_channel, cfg.z_strategy)
    clean_wga = apply_threshold(raw_wga, cfg.wga_threshold)
    clean_dapi = apply_threshold(raw_dapi, cfg.dapi_threshold)
    rgb_img = create_custom_rgb(clean_wga, clean_dapi, wga_color, dapi_color)

    masks, cells_df, stats = segment_image(
        wga=raw_wga,
        dapi=raw_dapi,
        cfg=cfg,
        pixel_size_um=pixel_size_um,
    )
    final_count = stats["cell_count"]
    mean_area = stats["mean_area_um2"]
    edge_dropped = stats["edge_dropped"]
    size_dropped = stats["size_dropped"]
    initial_count = stats["initial_count"]
    post_edge_count = initial_count - edge_dropped

    print(f"Final: {final_count} cells | mean area {mean_area:.1f} um^2")
    print(f"  removed {edge_dropped} touching the border")
    print(f"  removed {size_dropped} below {cfg.min_area_um2} um^2")

    fig, ax = plt.subplots(2, 3, figsize=figsize)

    # Top row: raw inputs + merged RGB.
    ax[0, 0].imshow(clean_wga, cmap="gray")
    ax[0, 0].set_title("Step 1a: WGA channel (cell borders)")

    ax[0, 1].imshow(clean_dapi, cmap="gray")
    ax[0, 1].set_title("Step 1b: DAPI channel (nuclei)")

    ax[0, 2].imshow(rgb_img)
    ax[0, 2].set_title(f"Step 2: Merged RGB\n(WGA={wga_color}, DAPI={dapi_color})")

    # Bottom row: cellpose output, size-filtered, final overlay.
    if initial_count > 0:
        ax[1, 0].imshow(plot.mask_rgb(masks))
    else:
        ax[1, 0].imshow(np.zeros_like(rgb_img))
    ax[1, 0].set_title(f"Step 3: After edge clearing (N={post_edge_count})")

    if final_count > 0:
        ax[1, 1].imshow(plot.mask_rgb(masks))
    else:
        ax[1, 1].imshow(np.zeros_like(rgb_img))
    ax[1, 1].set_title(
        f"Step 4: Size filter > {cfg.min_area_um2:.0f} um^2 (N={final_count})"
    )

    ax[1, 2].imshow(rgb_img)
    if final_count > 0:
        for outline in utils.outlines_list(masks):
            ax[1, 2].plot(outline[:, 0], outline[:, 1],
                          color="yellow", linewidth=1.5)
    ax[1, 2].set_title(f"Step 5: Final outlines (mean area {mean_area:.0f} um^2)")

    for a in ax.flat:
        a.axis("off")

    plt.tight_layout()
    return fig
