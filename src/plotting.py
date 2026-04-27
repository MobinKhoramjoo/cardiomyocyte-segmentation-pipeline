"""Plotting helpers for the per-image results CSV.

Three things you usually want to look at after a batch run:

  1. How are cell counts and mean areas distributed across images? (QC)
  2. Is there a relationship between cell count and mean area? (sanity check)
  3. Does cell size differ between experimental groups? (the actual question)

The group-comparison plot is generic: you supply a CSV mapping each
sample identifier to a group label, plus the order in which groups
should appear on the x-axis. Nothing is hard-coded.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from statannotations.Annotator import Annotator


# ---------------------------------------------------------------------------
# Loading + QC
# ---------------------------------------------------------------------------

def load_results(csv_path: str, min_cells_per_image: int = 10) -> pd.DataFrame:
    """Load the per-image CSV and drop images that didn't yield enough cells.

    Images with very few cells are usually bad fields of view (out of focus,
    mostly empty, or stitched at a tile edge) and should not influence the
    distribution statistics.
    """
    df = pd.read_csv(csv_path)
    n_initial = len(df)
    df_clean = df[df["Cell_Count"] >= min_cells_per_image].copy()
    print(f"QC: kept {len(df_clean)} / {n_initial} images "
          f"(dropped {n_initial - len(df_clean)} with < {min_cells_per_image} cells)")
    return df_clean


def add_sample_name(df: pd.DataFrame,
                    file_column: str = "File_Name",
                    out_column: str = "Sample_Name") -> pd.DataFrame:
    """Pull the sample identifier out of the filename.

    Convention used here: the sample name is the first whitespace-separated
    token in ``File_Name``. Rename or override this if your filenames follow
    a different scheme.
    """
    df = df.copy()
    df[out_column] = df[file_column].astype(str).str.split(" ").str[0]
    return df


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def plot_distributions(df_clean: pd.DataFrame,
                       title_suffix: str = "") -> plt.Figure:
    """Side-by-side histograms of per-image cell count and mean cell area."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(data=df_clean, x="Cell_Count", kde=True,
                 element="step", stat="count", color="#1f77b4", ax=axes[0])
    axes[0].set_title(f"Cell count per image{title_suffix}",
                      fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Number of cells detected")
    axes[0].set_ylabel("Number of images")

    sns.histplot(data=df_clean, x="Mean_Area_um2", kde=True,
                 element="step", stat="count", color="#1f77b4", ax=axes[1])
    axes[1].set_title(f"Mean cardiomyocyte area{title_suffix}",
                      fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Mean area (um^2)")
    axes[1].set_ylabel("Number of images")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def plot_count_vs_area(df_clean: pd.DataFrame) -> plt.Figure:
    """Scatter of cell count vs mean area with Pearson r annotated.

    A strong negative correlation is expected: when each cell is larger,
    fewer of them fit in a fixed-size field of view.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    corr, p_val = pearsonr(df_clean["Cell_Count"], df_clean["Mean_Area_um2"])

    fig = plt.figure(figsize=(8, 6))
    ax = sns.regplot(
        data=df_clean,
        x="Cell_Count",
        y="Mean_Area_um2",
        color="#1f77b4",
        scatter_kws={"alpha": 0.6, "edgecolor": "w", "s": 60},
        line_kws={"color": "darkred", "linewidth": 2},
    )

    p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(
        0.95, 0.95,
        f"Pearson r = {corr:.2f}\n{p_text}",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="gray"),
    )

    plt.title("Cell count vs. mean cardiomyocyte area",
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of cells detected per image", fontweight="bold")
    plt.ylabel("Mean area (um^2)", fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------------

def load_group_mapping(csv_path: str,
                       sample_col: str = "Sample_Name",
                       group_col: str = "Group") -> Dict[str, str]:
    """Read a sample-to-group mapping CSV into a dict.

    The CSV must have at least two columns: one with sample identifiers
    that match the ``Sample_Name`` derived from the filenames, and one
    with the group label you want to use on the x-axis.
    """
    mapping_df = pd.read_csv(csv_path)
    return dict(zip(mapping_df[sample_col].astype(str),
                    mapping_df[group_col].astype(str)))


def plot_group_comparison(df_clean: pd.DataFrame,
                          sample_to_group: Dict[str, str],
                          group_order: List[str],
                          group_colors: Optional[Dict[str, str]] = None,
                          aggregate_by_sample: bool = True,
                          title: str = "Cardiomyocyte area by group",
                          ylabel: str = "Mean area (um^2)",
                          figsize: Optional[tuple] = None,
                          equal_var: bool = True) -> plt.Figure:
    """Boxplot + swarm of mean cell area by group, with t-test annotations.

    Works with **any number of groups**:

      - 1 group  -> plots the box/swarm with no statistics.
      - 2 groups -> single t-test annotation.
      - 3+ groups -> all pairwise t-tests are annotated.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Output of :func:`load_results` plus :func:`add_sample_name`.
    sample_to_group : dict
        Maps each sample identifier to its group label.
    group_order : list[str]
        Left-to-right order of groups on the x-axis. Pass however many
        labels you have; the figure adapts to the count.
    group_colors : dict, optional
        Map from group label to a color string. Only the labels you
        provide are used; any group without an entry falls back to a
        seaborn-default colour. Pass ``None`` to use seaborn defaults
        for every group.
    aggregate_by_sample : bool
        If True (default), each dot is one sample (image-level values
        are averaged per sample). This is the right choice for
        biological-replicate statistics. Set to False to plot one dot
        per image, which is useful for QC / sanity checking.
    title, ylabel : str
        Plot text. Defaults stay generic on purpose -- override to
        match your experiment.
    figsize : (width, height), optional
        Override the figure size. By default the width scales with the
        number of groups so the plot doesn't get squashed for many
        groups or stretched for one or two.
    equal_var : bool
        Passed through to ``scipy.stats.ttest_ind``. Set False for a
        Welch's t-test if you don't want to assume equal variances.
    """
    df = df_clean.copy()
    df["Group"] = df["Sample_Name"].map(sample_to_group)

    unassigned = df[df["Group"].isna()]["Sample_Name"].unique()
    if len(unassigned):
        print("Warning: samples without a group mapping (excluded): "
              f"{list(unassigned)}")
        df = df.dropna(subset=["Group"])

    if aggregate_by_sample:
        df = (df.groupby(["Sample_Name", "Group"])["Mean_Area_um2"]
                .mean().reset_index())
        n_label = "smpls"
        dot_size = 7
        dot_alpha = 1.0
    else:
        n_label = "imgs (ROI)"
        dot_size = 6
        dot_alpha = 0.6

    # Only keep labels that are actually present in the data; this also
    # guards against typos in `group_order` and against groups with zero
    # rows after the QC filter.
    existing_groups = [g for g in group_order if g in df["Group"].unique()]
    if not existing_groups:
        raise ValueError(
            "None of the labels in group_order were found in the data. "
            f"Got group_order={group_order}, "
            f"available={sorted(df['Group'].unique())}"
        )

    # Adaptive figure width so 2 groups don't look stretched and 6 don't
    # look squashed. Roughly 1.2 inches per group, with a sensible floor.
    if figsize is None:
        width = max(4.0, 1.2 * len(existing_groups) + 2.0)
        figsize = (width, 6)

    # Resolve the palette so any missing entries fall back to a seaborn
    # default colour rather than crashing.
    if group_colors is None:
        palette: Dict[str, object] = {}
    else:
        palette = dict(group_colors)
    seaborn_defaults = sns.color_palette(n_colors=len(existing_groups))
    for i, g in enumerate(existing_groups):
        palette.setdefault(g, seaborn_defaults[i])

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig = plt.figure(figsize=figsize)

    # Hollow boxplot so the swarm dots are easy to see.
    ax = sns.boxplot(
        data=df, x="Group", y="Mean_Area_um2",
        order=existing_groups, color="0.5", fill=False, fliersize=0,
    )
    sns.swarmplot(
        data=df, x="Group", y="Mean_Area_um2",
        order=existing_groups,
        palette=palette,
        legend=False, size=dot_size, alpha=dot_alpha,
    )

    # Sample/image counts under each group.
    for i, label in enumerate(existing_groups):
        count = df[df["Group"] == label]["Mean_Area_um2"].count()
        ax.text(i, -0.08, f"n = {count}\n{n_label}",
                ha="center", va="top", fontsize=10,
                transform=ax.get_xaxis_transform())

    # Pairwise t-tests for every pair, with custom-formatted p-values.
    # itertools.combinations yields nothing for a single group, which is
    # the behaviour we want -- the boxplot is still drawn, just without
    # stats on top.
    pairs = list(itertools.combinations(existing_groups, 2))
    if pairs:
        p_values = []
        for g1, g2 in pairs:
            data1 = df[df["Group"] == g1]["Mean_Area_um2"]
            data2 = df[df["Group"] == g2]["Mean_Area_um2"]
            _, p = ttest_ind(data1, data2,
                             equal_var=equal_var, nan_policy="omit")
            p_values.append("< 0.0001" if p < 0.0001 else f"{p:.4f}")

        try:
            annotator = Annotator(ax, pairs, data=df,
                                  x="Group", y="Mean_Area_um2",
                                  order=existing_groups)
            annotator.set_custom_annotations(p_values)
            annotator.annotate()
        except Exception as exc:
            print(f"Could not annotate stats: {exc}")

    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel("")
    plt.tight_layout()
    return fig
