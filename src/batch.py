"""Batch processing of a folder of Leica .lif files.

Walks an input directory recursively, opens every .lif file, runs the
segmentation pipeline on each image inside it, and writes one row per
image to a CSV. Each .lif file usually contains many fields of view
(``num_images`` per file), so the per-image granularity is what we want.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, List

import numpy as np
import pandas as pd
from readlif.reader import LifFile

from .segmentation import (
    SegmentationConfig,
    _build_model,
    apply_threshold,
    get_z_projection,
    segment_image,
)


def find_lif_files(input_dir: str) -> List[str]:
    """Recursively gather every .lif file under ``input_dir``."""
    pattern = os.path.join(input_dir, "**", "*.lif")
    return sorted(glob.glob(pattern, recursive=True))


def process_lif_file(lif_path: str,
                     cfg: SegmentationConfig,
                     model) -> Iterable[dict]:
    """Yield one result dict per image inside a single .lif file."""
    file_name = os.path.basename(lif_path)
    reader = LifFile(lif_path)
    total_images = reader.num_images

    for img_idx, lif_img in enumerate(reader.get_iter_image(), start=1):
        img_name = lif_img.name

        try:
            # Pixel size lives in the .lif metadata. Cellpose still wants
            # pixel units, but we use this to convert area -> um^2.
            pixel_size_um = 1 / lif_img.scale[0]

            wga = get_z_projection(lif_img, cfg.wga_channel, cfg.z_strategy)
            dapi = get_z_projection(lif_img, cfg.dapi_channel, cfg.z_strategy)

            _masks, _cells, stats = segment_image(
                wga=wga,
                dapi=dapi,
                cfg=cfg,
                pixel_size_um=pixel_size_um,
                model=model,
            )

            yield {
                "File_Name": file_name,
                "Image_Name": img_name,
                "Image_Index": img_idx,
                "Total_Images_In_File": total_images,
                "Pixel_Size_um": pixel_size_um,
                "Cell_Count": stats["cell_count"],
                "Edge_Cells_Removed": stats["edge_dropped"],
                "Small_Cells_Removed": stats["size_dropped"],
                "Mean_Area_um2": stats["mean_area_um2"],
            }
        except Exception as exc:  # pragma: no cover - logged, not raised
            # We don't want one bad image to abort the whole run, so we
            # log it and move on. The error column makes failures easy
            # to spot in the CSV.
            print(f"  [Error] Could not process image '{img_name}': {exc}")
            yield {
                "File_Name": file_name,
                "Image_Name": img_name,
                "Image_Index": img_idx,
                "Total_Images_In_File": total_images,
                "Pixel_Size_um": np.nan,
                "Cell_Count": 0,
                "Edge_Cells_Removed": 0,
                "Small_Cells_Removed": 0,
                "Mean_Area_um2": 0.0,
                "Error": str(exc),
            }


def run_batch(input_dir: str,
              output_csv: str,
              cfg: SegmentationConfig | None = None) -> pd.DataFrame:
    """End-to-end batch run -- find every .lif, process it, save a CSV.

    Returns the assembled DataFrame so callers can keep working with it
    in-memory (e.g. inside a notebook) without re-reading the CSV.
    """
    cfg = cfg or SegmentationConfig()

    lif_files = find_lif_files(input_dir)
    print(f"Found {len(lif_files)} LIF files under {input_dir}")
    if not lif_files:
        print("No .lif files found. Check your input path.")
        return pd.DataFrame()

    # Load Cellpose once; reuse for every image. This is the slow step,
    # so don't put it inside the loop.
    print("Loading Cellpose model...")
    model = _build_model(cfg)

    rows: list[dict] = []
    for file_idx, lif_path in enumerate(lif_files, start=1):
        print(f"\n[{file_idx}/{len(lif_files)}] {os.path.basename(lif_path)}")
        try:
            for row in process_lif_file(lif_path, cfg, model):
                rows.append(row)
                print(
                    f"  -> {row['Image_Name']}: "
                    f"{row['Cell_Count']} cells kept, "
                    f"mean area {row['Mean_Area_um2']:.1f} um^2"
                )
        except Exception as exc:
            print(f"[Error] Could not read file '{lif_path}': {exc}")

    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nDone. Results saved to {output_csv}")
    return df
