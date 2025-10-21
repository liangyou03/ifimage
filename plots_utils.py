# bench_plots.py
# All comments in ENGLISH
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable

# ------------------------ helpers: data wrangling ------------------------

def _as_perimg_df(per_img: dict[str, pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    """
    Normalize input into a single per-image DataFrame with an 'algorithm' column.
    per_img can be:
      - dict: {algo_name: per-image DataFrame}, OR
      - DataFrame: with or without 'algorithm' column (will be filled with 'model' if missing).
    """
    if isinstance(per_img, dict):
        parts = []
        for algo, df in per_img.items():
            if df is None or len(df) == 0:
                continue
            df = df.copy()
            df["algorithm"] = algo  # force consistency
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    elif isinstance(per_img, pd.DataFrame):
        df = per_img.copy()
        if "algorithm" not in df.columns:
            df["algorithm"] = "model"
        return df
    else:
        raise TypeError("per_img must be a dict[str, DataFrame] or a DataFrame.")

def _infer_ap_cols(df: pd.DataFrame, thresholds: Iterable[float] | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Decide which AP columns to plot. Two modes:
      - If 'thresholds' is provided, we build exact column names 'AP@{t:.2f}' and check existence.
      - Otherwise, discover columns matching regex '^AP@\\d\\.\\d{2}$' and sort by threshold value.
    Returns (thresholds_array, ordered_column_names).
    """
    if thresholds is not None:
        thr = np.array([float(f"{t:.2f}") for t in thresholds], dtype=float)
        ap_cols = [f"AP@{t:.2f}" for t in thr]
        missing = [c for c in ap_cols if c not in df.columns]
        if missing:
            raise ValueError(f"AP columns required by thresholds not found: {missing}")
        return thr, ap_cols

    ap_cols = [c for c in df.columns if re.match(r"^AP@\d\.\d{2}$", c)]
    if not ap_cols:
        raise ValueError("No AP@xx.xx columns found in per-image table.")
    thr = np.array([float(c.split("@")[1]) for c in ap_cols], dtype=float)
    order = np.argsort(thr)
    thr = thr[order]
    ap_cols = [ap_cols[i] for i in order]
    return thr, ap_cols

# ------------------------ metrics computation ------------------------

def compute_curve_and_map(
    per_img: dict[str, pd.DataFrame] | pd.DataFrame,
    thresholds: Iterable[float] | None = None,
    group_cols: tuple[str, ...] = ("algorithm",),
) -> tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Core computation:
      - Normalize input per-image table.
      - Infer AP columns (optionally enforcing specific thresholds).
      - Compute mean AP per group at each threshold -> 'curve' DataFrame.
      - Compute mAP (mean over thresholds) per group -> 'mAP' Series.
    group_cols can be ('algorithm',) for overall, or ('celltype','algorithm') for faceted plots.
    Returns: (thresholds, curve_df, mAP_series)
      * curve_df index is a MultiIndex if you pass multiple group_cols.
    """
    df = _as_perimg_df(per_img)
    if df.empty:
        raise ValueError("Empty input: per-image table is empty after normalization.")
    thr, ap_cols = _infer_ap_cols(df, thresholds)
    curve = df.groupby(list(group_cols), sort=True)[ap_cols].mean()
    mAP = curve.mean(axis=1)
    return thr, curve, mAP

# ------------------------ plotting ------------------------

def _legend_order_from_map(index_vals, mAP: pd.Series, ascending: bool = True) -> list:
    """
    Return index values ordered by their mAP (ascending or descending).
    """
    m = mAP.reindex(index_vals)
    m = m.sort_values(ascending=ascending)
    return list(m.index)

def plot_overall_curves(
    curve: pd.DataFrame,
    thresholds: np.ndarray,
    mAP: pd.Series,
    title: str = "Cell benchmark — overall",
    legend_order: str = "asc",   # 'asc' -> small mAP at top; 'desc' -> big mAP at top
    out_png: Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot overall multi-line AP curves (index must be 'algorithm' only).
    Legend follows mAP ordering as requested.
    """
    if curve.index.nlevels != 1:
        raise ValueError("plot_overall_curves expects single-index (only 'algorithm').")
    # Determine plotting order by mAP
    asc = (legend_order == "asc")
    alg_order = _legend_order_from_map(curve.index, mAP, ascending=asc)

    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in alg_order:
        y = curve.loc[algo].values
        ax.plot(thresholds, y, marker="o", linewidth=2,
                label=f"{algo} (mAP={mAP.loc[algo]:.02f})")
    ax.set_title(title)
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Average Precision")
    ax.set_xlim(float(thresholds.min()), float(thresholds.max()))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved curve figure -> {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax

def plot_faceted_by_celltype(
    curve: pd.DataFrame,
    thresholds: np.ndarray,
    mAP: pd.Series,
    celltype_col: str = "celltype",
    title_prefix: str = "Cell benchmark —",
    legend_order: str = "asc",   # per-facet legend order by mAP (small->big)
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (5.5, 4.2),
    out_png: Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Faceted plot by cell type:
      - 'curve' must have a MultiIndex with levels (celltype, algorithm) in this order.
      - Within each facet (celltype), draw one line per algorithm.
      - Legend order within a facet is mAP ascending/descending as requested.
    Colors are consistent across facets (same algorithm -> same color).
    """
    if curve.index.nlevels != 2:
        raise ValueError("plot_faceted_by_celltype expects a two-level index (celltype, algorithm).")
    # Normalize level names/positions
    lvl_names = list(curve.index.names)
    if lvl_names[0] != celltype_col or lvl_names[1] != "algorithm":
        curve = curve.reorder_levels([celltype_col, "algorithm"]).sort_index()
        mAP = mAP.reindex(curve.index)  # align to new order

    celltypes = curve.index.get_level_values(0).unique().tolist()
    algos = curve.index.get_level_values(1).unique().tolist()

    # Assign a stable color per algorithm
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if prop_cycle is None or len(prop_cycle) == 0:
        colors = None
    else:
        colors = {algo: prop_cycle[i % len(prop_cycle)] for i, algo in enumerate(sorted(algos))}

    n_panels = len(celltypes)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_panels / ncols))
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    asc = (legend_order == "asc")
    for i, ct in enumerate(celltypes):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        # mAP ordering within this facet
        idx_here = curve.loc[ct].index  # algorithms available for this ct
        order = _legend_order_from_map(idx_here, mAP.loc[ct], ascending=asc)

        for algo in order:
            y = curve.loc[(ct, algo)].values
            kw = {}
            if colors is not None:
                kw["color"] = colors[algo]
            ax.plot(thresholds, y, marker="o", linewidth=2, label=f"{algo} (mAP={mAP.loc[(ct, algo)]:.02f})", **kw)

        ax.set_title(f"{title_prefix} {ct}")
        ax.set_xlabel("IoU Threshold")
        ax.set_ylabel("Average Precision")
        ax.set_xlim(float(thresholds.min()), float(thresholds.max()))
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()

    # Hide any unused axes if n_panels doesn't fill the grid
    for j in range(n_panels, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved faceted figure -> {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes
