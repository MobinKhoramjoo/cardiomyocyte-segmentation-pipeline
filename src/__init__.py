"""Cardiomyocyte size pipeline.

Cellpose-based segmentation of cardiomyocytes in immunofluorescence images,
using DAPI (nuclei) and WGA (cell borders) for the segmentation step,
followed by per-image quantification of cell count and cell area.
"""

__all__ = ["segmentation", "batch", "visualization", "plotting"]
