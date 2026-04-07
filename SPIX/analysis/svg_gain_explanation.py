from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import wilcoxon


DEFAULT_DRIVER_COLORS: Dict[str, str] = {
    "boundary": "#1b9e77",
    "snr": "#d95f02",
    "autocorr": "#7570b3",
    "mixed": "#e6ab02",
    "weak": "#bdbdbd",
}


DEFAULT_DRIVER_METRICS: Dict[str, str] = {
    "boundary": "gain_neighbor_contrast",
    "snr": "gain_segment_snr",
    "autocorr": "gain_segment_moranI",
}


PRETTY_METRIC_LABELS: Dict[str, str] = {
    "gain_neighbor_contrast": "Boundary gain",
    "gain_segment_snr": "Segment SNR gain",
    "gain_segment_moranI": "Segment Moran's I gain",
    "gain_between_segment_var": "Between-segment variance gain",
    "gain_within_segment_var": "Within-segment noise reduction",
    "gain_segment_dynamic_range": "Dynamic range gain",
    "gain_segment_detection_rate": "Detection-rate gain",
    "gain_segment_top10_mass": "Localization concentration gain",
    "concentration_score": "Top-10% concentration score",
    "concentration_gain": "Top-10% concentration gain",
    "segment_top10_mass": "Localization concentration",
    "localization_auc": "Localization AUC",
    "localization_auc_gain": "Localization AUC gain",
    "half_mass_fraction": "Half-mass fraction",
    "half_mass_fraction_gain": "Half-mass fraction gain",
    "effective_segments": "Effective expressing segments",
    "effective_segments_gain": "Effective-segment reduction",
    "within_segment_var": "Within-segment noise",
    "neighbor_contrast": "Boundary separation",
    "between_segment_var": "Between-segment separation",
    "segment_dynamic_range": "Segment dynamic range",
    "contrast_SPIX_vs_GRID": "SPIX rank advantage",
}


DEFAULT_REASON_GAIN_METRICS: List[str] = [
    "gain_within_segment_var",
    "gain_neighbor_contrast",
    "gain_between_segment_var",
    "gain_segment_dynamic_range",
    "gain_segment_top10_mass",
]


DEFAULT_REASON_RAW_METRICS: List[str] = [
    "within_segment_var",
    "neighbor_contrast",
    "between_segment_var",
    "segment_dynamic_range",
    "segment_top10_mass",
]


DEFAULT_REASON_GAIN_COLORS: Dict[str, str] = {
    "gain_within_segment_var": "#4c78a8",
    "gain_neighbor_contrast": "#f58518",
    "gain_between_segment_var": "#54a24b",
    "gain_segment_dynamic_range": "#b279a2",
    "gain_segment_top10_mass": "#e45756",
}


DISCOVERY_REASON_LABELS: Dict[str, str] = {
    "within_segment_var": "Low within-segment noise",
    "neighbor_contrast": "Sharp boundary separation",
    "between_segment_var": "Strong between-segment separation",
    "segment_dynamic_range": "Broad segment dynamic range",
    "segment_top10_mass": "Localized segment concentration",
    "gain_within_segment_var": "Noise reduction gain",
    "gain_neighbor_contrast": "Boundary gain",
    "gain_between_segment_var": "Between-segment separation gain",
    "gain_segment_dynamic_range": "Dynamic-range gain",
    "gain_segment_top10_mass": "Localization gain",
}


DISCOVERY_PLOT_LABELS: Dict[str, str] = {
    "within_segment_var": "Within-segment homogeneity",
    "neighbor_contrast": "Boundary contrast",
    "between_segment_var": "Between-segment contrast",
    "segment_dynamic_range": "Dynamic range",
    "segment_top10_mass": "Expression concentration",
    "gain_within_segment_var": "Homogeneity gain",
    "gain_neighbor_contrast": "Boundary gain",
    "gain_between_segment_var": "Between-segment gain",
    "gain_segment_dynamic_range": "Dynamic-range gain",
    "gain_segment_top10_mass": "Concentration gain",
}


DISCOVERY_REASON_DIRECTIONS: Dict[str, float] = {
    "within_segment_var": -1.0,
    "neighbor_contrast": 1.0,
    "between_segment_var": 1.0,
    "segment_dynamic_range": 1.0,
    "segment_top10_mass": 1.0,
}


def _resolve_segment_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    candidates = [
        base_dir / p,
        base_dir.parent / p,
        base_dir / p.name,
        base_dir.parent / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Could not resolve segment path: {raw_path}")


def _extract_gene_vector(adata, gene: str, *, layer: Optional[str] = None) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene not found in adata.var_names: {gene}")

    mat = adata.layers[layer] if layer is not None else adata.X
    x = mat[:, adata.var_names.get_loc(gene)]
    if issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x


def _boundary_edges_from_label_img(label_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = np.asarray(label_img, dtype=np.int64)
    if lab.ndim != 2:
        raise ValueError("label_img must be 2D")

    def _pairs(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keep = (a >= 0) & (b >= 0) & (a != b)
        if not np.any(keep):
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        aa = a[keep]
        bb = b[keep]
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        return lo, hi

    lo_h, hi_h = _pairs(lab[:, :-1].ravel(), lab[:, 1:].ravel())
    lo_v, hi_v = _pairs(lab[:-1, :].ravel(), lab[1:, :].ravel())

    if lo_h.size + lo_v.size == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )

    lo = np.concatenate([lo_h, lo_v], axis=0)
    hi = np.concatenate([hi_h, hi_v], axis=0)
    base = int(lab.max(initial=-1)) + 1
    codes = lo * base + hi
    uniq, counts = np.unique(codes, return_counts=True)
    src = (uniq // base).astype(np.int32, copy=False)
    dst = (uniq % base).astype(np.int32, copy=False)
    weight = counts.astype(np.float64, copy=False)
    return src, dst, weight


def _load_segmentation_cache(npz_path: str | Path) -> Dict[str, np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with np.load(npz_path, allow_pickle=False) as z:
        seg_codes = z["seg_codes"].astype(np.int32, copy=True)
        label_img = z["label_img"].astype(np.int32, copy=False)

    src, dst, weight = _boundary_edges_from_label_img(label_img)
    return {
        "seg_codes": seg_codes,
        "src": src,
        "dst": dst,
        "weight": weight,
    }


def _segment_metrics(
    values: np.ndarray,
    seg_codes: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    valid = np.isfinite(values) & (seg_codes >= 0)
    if not np.any(valid):
        return {
            "within_segment_var": np.nan,
            "between_segment_var": np.nan,
            "segment_snr": np.nan,
            "neighbor_contrast": np.nan,
            "segment_moranI": np.nan,
            "segment_detection_rate": np.nan,
            "segment_dynamic_range": np.nan,
        }

    codes = seg_codes[valid].astype(np.int64, copy=False)
    x = values[valid].astype(np.float64, copy=False)
    n_segments = int(codes.max(initial=-1)) + 1

    counts = np.bincount(codes, minlength=n_segments).astype(np.float64, copy=False)
    sums = np.bincount(codes, weights=x, minlength=n_segments).astype(np.float64, copy=False)
    sums_sq = np.bincount(codes, weights=x * x, minlength=n_segments).astype(np.float64, copy=False)

    nz = counts > 0
    means = np.zeros(n_segments, dtype=np.float64)
    means[nz] = sums[nz] / counts[nz]

    total_count = float(counts[nz].sum())
    global_mean = float(sums[nz].sum() / max(total_count, eps))

    within_ss = sums_sq[nz] - 2.0 * means[nz] * sums[nz] + counts[nz] * means[nz] * means[nz]
    within_var = float(within_ss.sum() / max(total_count, eps))
    between_var = float(np.average((means[nz] - global_mean) ** 2, weights=counts[nz]))
    segment_snr = float(between_var / (within_var + eps))
    detection_rate = float(np.mean(means[nz] > 0))
    if int(nz.sum()) > 1:
        q10, q90 = np.nanpercentile(means[nz], [10.0, 90.0])
        dynamic_range = float(q90 - q10)
    else:
        dynamic_range = 0.0

    total_sum = float(sums[nz].sum())
    if total_sum > eps and int(nz.sum()) > 0:
        k = max(1, int(np.ceil(0.1 * int(nz.sum()))))
        top10_mass = float(np.sort(sums[nz])[::-1][:k].sum() / total_sum)
    else:
        top10_mass = np.nan

    in_range = (src >= 0) & (dst >= 0) & (src < n_segments) & (dst < n_segments)
    if not np.any(in_range):
        neighbor_contrast = np.nan
        moran_i = np.nan
    else:
        src_in = src[in_range]
        dst_in = dst[in_range]
        weight_in = weight[in_range]
        edge_mask = nz[src_in] & nz[dst_in]
    if not np.any(in_range) or not np.any(edge_mask):
        neighbor_contrast = np.nan
        moran_i = np.nan
    else:
        src_v = src_in[edge_mask]
        dst_v = dst_in[edge_mask]
        w_v = weight_in[edge_mask]
        neighbor_contrast = float(np.average(np.abs(means[src_v] - means[dst_v]), weights=w_v))

        centered = np.zeros_like(means)
        centered[nz] = means[nz] - means[nz].mean()
        denom = float(np.sum(centered[nz] ** 2))
        if denom <= eps:
            moran_i = np.nan
        else:
            num = float(2.0 * np.sum(w_v * centered[src_v] * centered[dst_v]))
            sum_w = float(2.0 * np.sum(w_v))
            moran_i = float((float(nz.sum()) / max(sum_w, eps)) * (num / denom))

    return {
        "within_segment_var": within_var,
        "between_segment_var": between_var,
        "segment_snr": segment_snr,
        "neighbor_contrast": neighbor_contrast,
        "segment_moranI": moran_i,
        "segment_detection_rate": detection_rate,
        "segment_dynamic_range": dynamic_range,
        "segment_top10_mass": top10_mass,
    }


def explain_svg_rank_gain(
    adata,
    segments_index_csv: str,
    genes: Iterable[str],
    *,
    focus_compactness: float = 0.1,
    reference_compactness: float = 1e7,
    layer: Optional[str] = None,
    rank_summary_df: Optional[pd.DataFrame] = None,
    compactness_tol: float = 1e-9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Explain why SPIX can outrank GRID for selected genes.

    The function matches segmentations by resolution, then compares how each
    segmentation changes expression structure for each gene:

    - lower within-segment variance        -> better denoising
    - higher between-segment variance      -> better segment separation
    - higher segment-level SNR             -> stronger biological contrast
    - higher neighbor contrast             -> sharper spatial boundaries
    - higher segment Moran's I             -> stronger spatial autocorrelation
    - higher segment dynamic range         -> more preserved expression spread

    Returns
    -------
    summary_df : pd.DataFrame
        One row per gene with mean metrics and gain columns. Positive `gain_*`
        means SPIX is better on that axis.
    long_df : pd.DataFrame
        Per-resolution matched comparison table.
    """

    index_path, matched = _load_matched_segment_index(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )

    genes = [str(g) for g in genes if str(g) in adata.var_names]
    if not genes:
        raise ValueError("No input genes were found in adata.var_names")

    seg_cache: Dict[str, Dict[str, np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    rows: List[Dict[str, object]] = []
    metric_names = [
        "within_segment_var",
        "between_segment_var",
        "segment_snr",
        "neighbor_contrast",
        "segment_moranI",
        "segment_detection_rate",
        "segment_dynamic_range",
        "segment_top10_mass",
    ]

    for gene in genes:
        values = _extract_gene_vector(adata, gene, layer=layer)

        for row in matched.itertuples(index=False):
            focus_path = _resolve_segment_path(getattr(row, "path_focus"), index_path.parent)
            ref_path = _resolve_segment_path(getattr(row, "path_reference"), index_path.parent)

            focus_key = str(focus_path)
            if focus_key not in seg_cache:
                seg_cache[focus_key] = _load_segmentation_cache(focus_path)
            ref_key = str(ref_path)
            if ref_key not in seg_cache:
                seg_cache[ref_key] = _load_segmentation_cache(ref_path)

            focus_seg = seg_cache[focus_key]
            ref_seg = seg_cache[ref_key]

            focus_metrics = _segment_metrics(
                values,
                focus_seg["seg_codes"],  # type: ignore[arg-type]
                focus_seg["src"],        # type: ignore[arg-type]
                focus_seg["dst"],        # type: ignore[arg-type]
                focus_seg["weight"],     # type: ignore[arg-type]
            )
            ref_metrics = _segment_metrics(
                values,
                ref_seg["seg_codes"],    # type: ignore[arg-type]
                ref_seg["src"],          # type: ignore[arg-type]
                ref_seg["dst"],          # type: ignore[arg-type]
                ref_seg["weight"],       # type: ignore[arg-type]
            )

            out: Dict[str, object] = {
                "gene": gene,
                "resolution": float(getattr(row, "resolution")),
                "focus_compactness": float(focus_compactness),
                "reference_compactness": float(reference_compactness),
            }
            for metric in metric_names:
                focus_val = float(focus_metrics[metric])
                ref_val = float(ref_metrics[metric])
                out[f"{metric}_SPIX"] = focus_val
                out[f"{metric}_GRID"] = ref_val

            out["gain_within_segment_var"] = out["within_segment_var_GRID"] - out["within_segment_var_SPIX"]
            out["gain_between_segment_var"] = out["between_segment_var_SPIX"] - out["between_segment_var_GRID"]
            out["gain_segment_snr"] = out["segment_snr_SPIX"] - out["segment_snr_GRID"]
            out["gain_neighbor_contrast"] = out["neighbor_contrast_SPIX"] - out["neighbor_contrast_GRID"]
            out["gain_segment_moranI"] = out["segment_moranI_SPIX"] - out["segment_moranI_GRID"]
            out["gain_segment_detection_rate"] = out["segment_detection_rate_SPIX"] - out["segment_detection_rate_GRID"]
            out["gain_segment_dynamic_range"] = out["segment_dynamic_range_SPIX"] - out["segment_dynamic_range_GRID"]
            out["gain_segment_top10_mass"] = out["segment_top10_mass_SPIX"] - out["segment_top10_mass_GRID"]
            rows.append(out)

    long_df = pd.DataFrame(rows)
    numeric_cols = [c for c in long_df.columns if c not in {"gene"}]
    summary_df = long_df.groupby("gene", sort=True)[numeric_cols].mean(numeric_only=True)

    if rank_summary_df is not None:
        summary_df = summary_df.join(rank_summary_df, how="left")

    return summary_df, long_df


def annotate_svg_gain_drivers(
    summary_df: pd.DataFrame,
    *,
    driver_metrics: Optional[Dict[str, str]] = None,
    dominance_ratio: float = 1.25,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add dominant gain-driver labels for each gene.

    Driver assignment is based on positive gains after scale-normalization:
    - `boundary`: stronger inter-segment edge contrast
    - `snr`: better segment-level signal-to-noise
    - `autocorr`: stronger segment-level spatial autocorrelation
    - `mixed`: more than one driver with similar strength
    - `weak`: no positive gain across the tracked drivers
    """

    driver_metrics = driver_metrics or DEFAULT_DRIVER_METRICS
    missing = [metric for metric in driver_metrics.values() if metric not in summary_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for driver annotation: {missing}")

    df = summary_df.copy() if copy else summary_df

    contrib_cols: List[str] = []
    for driver_name, metric in driver_metrics.items():
        vals = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        pos = np.clip(vals.to_numpy(dtype=float), 0.0, None)
        nz = pos[pos > 0]
        scale = float(np.nanpercentile(nz, 90.0)) if nz.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        contrib_col = f"{driver_name}_driver_score"
        df[contrib_col] = pos / scale
        contrib_cols.append(contrib_col)

    contrib = df[contrib_cols].to_numpy(dtype=float)
    top_idx = np.argmax(contrib, axis=1)
    top_val = contrib[np.arange(len(df)), top_idx]

    if contrib.shape[1] > 1:
        second_val = np.partition(contrib, kth=contrib.shape[1] - 2, axis=1)[:, -2]
    else:
        second_val = np.zeros(len(df), dtype=float)

    driver_names = list(driver_metrics.keys())
    dominant = np.array([driver_names[i] for i in top_idx], dtype=object)
    dominant[top_val <= 0] = "weak"

    similar = (top_val > 0) & (second_val > 0) & (top_val / np.maximum(second_val, 1e-12) < float(dominance_ratio))
    dominant[similar] = "mixed"

    df["dominant_driver"] = dominant
    df["dominant_driver_strength"] = top_val
    df["driver_balance_ratio"] = top_val / np.maximum(second_val, 1e-12)
    return df


def summarize_svg_gain_drivers(
    summary_df: pd.DataFrame,
    *,
    driver_col: str = "dominant_driver",
    contrast_col: str = "contrast_SPIX_vs_GRID",
) -> pd.DataFrame:
    """
    Summarize how many genes are explained by each gain driver.
    """

    if driver_col not in summary_df.columns:
        df = annotate_svg_gain_drivers(summary_df, copy=True)
    else:
        df = summary_df.copy()

    rows: List[Dict[str, object]] = []
    total = max(1, int(len(df)))
    metrics = [c for c in ["gain_neighbor_contrast", "gain_segment_snr", "gain_segment_moranI", contrast_col] if c in df.columns]

    for driver, sub in df.groupby(driver_col, sort=False):
        row: Dict[str, object] = {
            "driver": str(driver),
            "n_genes": int(len(sub)),
            "fraction": float(len(sub) / total),
        }
        for metric in metrics:
            row[f"median_{metric}"] = float(pd.to_numeric(sub[metric], errors="coerce").median())
        if "dominant_driver_strength" in sub.columns:
            row["median_driver_strength"] = float(pd.to_numeric(sub["dominant_driver_strength"], errors="coerce").median())
        rows.append(row)

    out = pd.DataFrame(rows).set_index("driver")
    if not out.empty:
        out = out.sort_values("n_genes", ascending=False)
    return out


def _pretty_label(metric: str) -> str:
    return PRETTY_METRIC_LABELS.get(metric, metric)


def _reason_label(metric: str) -> str:
    return DISCOVERY_REASON_LABELS.get(metric, _pretty_label(metric))


def _plot_reason_label(metric: str) -> str:
    return DISCOVERY_PLOT_LABELS.get(metric, _reason_label(metric))


def _oriented_reason_values(values: pd.Series, metric: str) -> pd.Series:
    direction = float(DISCOVERY_REASON_DIRECTIONS.get(metric, 1.0))
    vals = pd.to_numeric(values, errors="coerce")
    return vals * direction


def plot_svg_rank_gain_explanation(
    summary_df: pd.DataFrame,
    *,
    x: str = "gain_neighbor_contrast",
    y: str = "gain_segment_snr",
    size: str = "gain_segment_moranI",
    color: Optional[str] = "contrast_SPIX_vs_GRID",
    top_n_labels: int = 8,
    figsize: Tuple[float, float] = (8.0, 6.0),
    cmap: str = "viridis",
    annotate_quadrants: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a gene-level explanation scatter.

    Default interpretation:
    - x > 0: SPIX creates sharper boundaries than GRID
    - y > 0: SPIX improves segment-level signal-to-noise
    - size > 0: SPIX increases segment-level Moran's I
    """

    required = [x, y, size]
    if color is not None:
        required.append(color)
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for plot: {missing}")

    df = annotate_svg_gain_drivers(summary_df, copy=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    size_vals = pd.to_numeric(df[size], errors="coerce").fillna(0.0).to_numpy()
    size_scaled = 36.0 + 260.0 * np.clip(size_vals, 0.0, None)

    scatter_kwargs = dict(
        x=pd.to_numeric(df[x], errors="coerce"),
        y=pd.to_numeric(df[y], errors="coerce"),
        s=size_scaled,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.4,
    )
    if color is None:
        sc = ax.scatter(color="#d95f02", **scatter_kwargs)
    else:
        sc = ax.scatter(c=pd.to_numeric(df[color], errors="coerce"), cmap=cmap, **scatter_kwargs)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(_pretty_label(color))

    x_vals = pd.to_numeric(df[x], errors="coerce")
    y_vals = pd.to_numeric(df[y], errors="coerce")
    x_max = float(np.nanmax(np.abs(x_vals))) if np.isfinite(x_vals).any() else 1.0
    y_max = float(np.nanmax(np.abs(y_vals))) if np.isfinite(y_vals).any() else 1.0
    ax.axvspan(0.0, max(x_max, 1e-9), ymin=0.0, ymax=1.0, color="#1b9e77", alpha=0.04)
    ax.axhspan(0.0, max(y_max, 1e-9), xmin=0.0, xmax=1.0, color="#d95f02", alpha=0.04)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(_pretty_label(x))
    ax.set_ylabel(_pretty_label(y))
    ax.set_title("Why does SPIX outrank GRID?")
    ax.grid(alpha=0.2)

    if annotate_quadrants:
        ax.text(
            0.98,
            0.97,
            "better boundaries\n+ better SNR",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
        )
        ax.text(
            0.98,
            0.03,
            "boundary gain only",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        ax.text(
            0.02,
            0.97,
            "SNR gain only",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )

    rank_col = color if color is not None else y
    top = df.sort_values(rank_col, ascending=False).head(int(top_n_labels))
    for gene, row in top.iterrows():
        ax.annotate(
            str(gene),
            (float(row[x]), float(row[y])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#333333",
        )

    return fig, ax


def plot_svg_rank_gain_dashboard(
    summary_df: pd.DataFrame,
    *,
    x: str = "gain_neighbor_contrast",
    y: str = "gain_segment_snr",
    size: str = "gain_segment_moranI",
    color: str = "contrast_SPIX_vs_GRID",
    heatmap_metrics: Optional[List[str]] = None,
    top_n_labels: int = 8,
    top_n_heatmap: int = 12,
    figsize: Tuple[float, float] = (15.0, 10.0),
    cmap: str = "viridis",
) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    Richer interpretation figure for SPIX rank gains.

    Panels:
    1. main scatter (`x` vs `y`, colored by rank advantage)
    2. median gain bar chart across explanation metrics
    3. dominant-driver counts
    4. top-gene heatmap across gain metrics
    """

    df = annotate_svg_gain_drivers(summary_df, copy=True)
    driver_summary = summarize_svg_gain_drivers(df)

    heatmap_metrics = heatmap_metrics or [
        "gain_segment_snr",
        "gain_neighbor_contrast",
        "gain_segment_moranI",
        "gain_between_segment_var",
        "gain_within_segment_var",
    ]
    heatmap_metrics = [m for m in heatmap_metrics if m in df.columns]
    if not heatmap_metrics:
        raise ValueError("No valid heatmap metrics found in summary_df")

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax_scatter, ax_metric, ax_driver, ax_heatmap = axes.ravel()

    # Panel 1: main scatter
    size_vals = pd.to_numeric(df[size], errors="coerce").fillna(0.0).to_numpy()
    size_scaled = 30.0 + 260.0 * np.clip(size_vals, 0.0, None)
    sc = ax_scatter.scatter(
        pd.to_numeric(df[x], errors="coerce"),
        pd.to_numeric(df[y], errors="coerce"),
        c=pd.to_numeric(df[color], errors="coerce"),
        cmap=cmap,
        s=size_scaled,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.4,
    )
    cbar = fig.colorbar(sc, ax=ax_scatter)
    cbar.set_label(_pretty_label(color))
    ax_scatter.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_scatter.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_scatter.set_xlabel(_pretty_label(x))
    ax_scatter.set_ylabel(_pretty_label(y))
    ax_scatter.set_title("Gene-level Gain Map")
    ax_scatter.grid(alpha=0.2)
    ax_scatter.text(0.98, 0.97, "SPIX improves both", transform=ax_scatter.transAxes, ha="right", va="top", fontsize=9)

    label_metric = color if color in df.columns else y
    top = df.sort_values(label_metric, ascending=False).head(int(top_n_labels))
    for gene, row in top.iterrows():
        ax_scatter.annotate(
            str(gene),
            (float(row[x]), float(row[y])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#333333",
        )

    # Panel 2: median gains
    bar_metrics = [m for m in [
        "gain_segment_snr",
        "gain_neighbor_contrast",
        "gain_segment_moranI",
        "gain_between_segment_var",
        "gain_within_segment_var",
    ] if m in df.columns]
    med = pd.Series({m: float(pd.to_numeric(df[m], errors="coerce").median()) for m in bar_metrics})
    colors_bar = ["#d95f02" if v >= 0 else "#999999" for v in med.values]
    ax_metric.barh(range(len(med)), med.values, color=colors_bar, edgecolor="#333333", linewidth=0.8)
    ax_metric.set_yticks(range(len(med)))
    ax_metric.set_yticklabels([_pretty_label(m) for m in med.index])
    ax_metric.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_metric.set_title("Median Gain Across SPIX Genes")
    ax_metric.set_xlabel("Median gain (SPIX - GRID)")
    ax_metric.grid(axis="x", alpha=0.2)

    # Panel 3: dominant driver counts
    driver_counts = df["dominant_driver"].value_counts()
    ordered_drivers = [d for d in ["boundary", "snr", "autocorr", "mixed", "weak"] if d in driver_counts.index]
    ax_driver.bar(
        ordered_drivers,
        driver_counts.reindex(ordered_drivers).values,
        color=[DEFAULT_DRIVER_COLORS.get(d, "#cccccc") for d in ordered_drivers],
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax_driver.set_title("Dominant Gain Driver")
    ax_driver.set_ylabel("Gene count")
    ax_driver.grid(axis="y", alpha=0.2)
    ax_driver.tick_params(axis="x", rotation=20)

    # Panel 4: top-gene heatmap
    heat_rank = (
        pd.to_numeric(df[color], errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("gain_segment_snr", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("gain_neighbor_contrast", 0.0), errors="coerce").fillna(0.0)
    )
    top_heat = df.loc[heat_rank.sort_values(ascending=False).head(int(top_n_heatmap)).index, heatmap_metrics].copy()
    if not top_heat.empty:
        arr = top_heat.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        col_mean = np.nanmean(arr, axis=0, keepdims=True)
        col_std = np.nanstd(arr, axis=0, keepdims=True)
        col_std[col_std == 0] = 1.0
        z = (arr - col_mean) / col_std
        im = ax_heatmap.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2.0, vmax=2.0)
        fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04).set_label("column z-score")
        ax_heatmap.set_xticks(range(len(heatmap_metrics)))
        ax_heatmap.set_xticklabels([_pretty_label(m) for m in heatmap_metrics], rotation=30, ha="right")
        ax_heatmap.set_yticks(range(len(top_heat.index)))
        ax_heatmap.set_yticklabels([str(g) for g in top_heat.index])
        ax_heatmap.set_title("Top SPIX Genes: Gain Decomposition")
    else:
        ax_heatmap.text(0.5, 0.5, "No genes", ha="center", va="center", transform=ax_heatmap.transAxes)
        ax_heatmap.set_axis_off()

    return fig, axes, df


def summarize_spix_discovery_reasons(
    summary_df: pd.DataFrame,
    *,
    gain_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Non-circular summary of why SPIX-specific genes are recovered in SPIX.

    Uses only metrics not derived from Moran rank:
    - within-segment noise reduction
    - boundary separation
    - between-segment separation
    - segment dynamic range
    - localization concentration
    """

    gain_metrics = gain_metrics or DEFAULT_REASON_GAIN_METRICS
    gain_metrics = [m for m in gain_metrics if m in summary_df.columns]
    if not gain_metrics:
        raise ValueError("No valid non-circular gain metrics found")

    rows: List[Dict[str, float | str]] = []
    for metric in gain_metrics:
        vals = pd.to_numeric(summary_df[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "label": _reason_label(metric),
                "median_gain": float(vals.median()),
                "mean_gain": float(vals.mean()),
                "fraction_positive": float((vals > 0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def plot_spix_discovery_reasons(
    summary_df: pd.DataFrame,
    *,
    rank_col: str = "contrast_SPIX_vs_GRID",
    gain_metrics: Optional[List[str]] = None,
    raw_metrics: Optional[List[str]] = None,
    top_n_genes: int = 12,
    figsize: Tuple[float, float] = (16.0, 11.0),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot why SPIX-specific genes are found in SPIX but missed in GRID.

    This dashboard avoids Moran's I as an explanation axis to reduce circularity.
    It focuses on raw segmentation effects that make a gene easier to recover:

    - lower within-segment noise
    - stronger boundary separation
    - larger between-segment spread
    - larger segment dynamic range
    - stronger localization into top segments
    """

    gain_metrics = gain_metrics or DEFAULT_REASON_GAIN_METRICS
    raw_metrics = raw_metrics or DEFAULT_REASON_RAW_METRICS

    gain_metrics = [m for m in gain_metrics if m in summary_df.columns]
    raw_metrics = [m for m in raw_metrics if f"{m}_SPIX" in summary_df.columns and f"{m}_GRID" in summary_df.columns]
    if not gain_metrics:
        raise ValueError("No valid non-circular gain metrics found in summary_df")

    df = summary_df.copy()
    top_n_genes = max(1, int(top_n_genes))
    fig_w, fig_h = figsize
    fig_h = max(float(fig_h), 6.8 + 0.32 * top_n_genes)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0])
    ax_profile = fig.add_subplot(gs[0, :])
    ax_grid = fig.add_subplot(gs[1, 0])
    ax_spix = fig.add_subplot(gs[1, 1], sharey=ax_grid)

    # Panel 1: median recovery profile after metric-wise orientation/standardization
    profile_rows: List[Dict[str, float | str]] = []
    for metric in raw_metrics:
        grid_vals = _oriented_reason_values(df[f"{metric}_GRID"], metric)
        spix_vals = _oriented_reason_values(df[f"{metric}_SPIX"], metric)
        both = pd.concat([grid_vals, spix_vals], axis=0).dropna()
        if both.empty:
            continue
        mean_val = float(both.mean())
        std_val = float(both.std(ddof=0))
        if not np.isfinite(std_val) or std_val <= 0.0:
            std_val = 1.0
        profile_rows.append(
            {
                "metric": metric,
                "median_GRID": float(((grid_vals - mean_val) / std_val).median()),
                "median_SPIX": float(((spix_vals - mean_val) / std_val).median()),
            }
        )

    profile_df = pd.DataFrame(profile_rows).set_index("metric") if profile_rows else pd.DataFrame()
    if not profile_df.empty:
        y_pos = np.arange(len(profile_df), dtype=float)
        ax_profile.hlines(
            y_pos,
            profile_df["median_GRID"].to_numpy(dtype=float),
            profile_df["median_SPIX"].to_numpy(dtype=float),
            color="#9e9e9e",
            linewidth=2.0,
            alpha=0.9,
        )
        ax_profile.scatter(
            profile_df["median_GRID"].to_numpy(dtype=float),
            y_pos,
            color="#d95f02",
            s=70,
            label="GRID",
            zorder=3,
        )
        ax_profile.scatter(
            profile_df["median_SPIX"].to_numpy(dtype=float),
            y_pos,
            color="#1b9e77",
            s=70,
            label="SPIX",
            zorder=3,
        )
        ax_profile.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        ax_profile.set_yticks(y_pos)
        ax_profile.set_yticklabels([_plot_reason_label(m) for m in profile_df.index], fontsize=11)
        ax_profile.set_xlabel("Relative recoverability (oriented z-score; higher is better)")
        ax_profile.set_title("Why GRID Misses These Genes but SPIX Recovers Them", fontsize=18, pad=10)
        ax_profile.grid(axis="x", alpha=0.2)
        ax_profile.legend(frameon=False, loc="lower right", ncols=2, fontsize=11)
        x_vals = profile_df[["median_GRID", "median_SPIX"]].to_numpy(dtype=float)
        x_span = max(float(np.nanmax(np.abs(x_vals))), 1e-9)
        ax_profile.set_xlim(float(np.nanmin(x_vals)) - 0.18 * x_span, float(np.nanmax(x_vals)) + 0.55 * x_span)
    else:
        ax_profile.text(0.5, 0.5, "No valid raw metrics", ha="center", va="center", transform=ax_profile.transAxes)
        ax_profile.set_axis_off()

    gain_summary = summarize_spix_discovery_reasons(df, gain_metrics=gain_metrics)
    if not profile_df.empty:
        gain_lookup = {metric.replace("gain_", "", 1): metric for metric in gain_summary.index}
        span_profile = float(
            max(
                np.nanmax(np.abs(profile_df[["median_GRID", "median_SPIX"]].to_numpy(dtype=float))),
                1e-9,
            )
        )
        for y_i, raw_metric in enumerate(profile_df.index):
            gain_metric = gain_lookup.get(str(raw_metric))
            if gain_metric is None:
                continue
            gain_val = float(gain_summary.loc[gain_metric, "median_gain"])
            frac = float(gain_summary.loc[gain_metric, "fraction_positive"])
            right_x = max(
                float(profile_df.loc[raw_metric, "median_GRID"]),
                float(profile_df.loc[raw_metric, "median_SPIX"]),
            )
            ax_profile.text(
                right_x + 0.05 * span_profile,
                float(y_i),
                f"Δ={gain_val:.3f}  {frac:.0%}",
                va="center",
                ha="left",
                fontsize=10,
                color="#333333",
            )

    # Panel 2: top-gene recovery profile heatmaps (higher is better)
    sort_metric = rank_col if rank_col in df.columns else gain_metrics[0]
    top = df.sort_values(sort_metric, ascending=False).head(top_n_genes).copy()
    top_heat = top.copy()
    grid_arr = []
    spix_arr = []
    for _, row in top_heat.iterrows():
        grid_arr.append(
            np.array(
                [float(_oriented_reason_values(pd.Series([row[f"{m}_GRID"]]), m).iloc[0]) for m in raw_metrics],
                dtype=float,
            )
        )
        spix_arr.append(
            np.array(
                [float(_oriented_reason_values(pd.Series([row[f"{m}_SPIX"]]), m).iloc[0]) for m in raw_metrics],
                dtype=float,
            )
        )

    arr_grid = np.vstack(grid_arr) if grid_arr else np.zeros((0, len(raw_metrics)), dtype=float)
    arr_spix = np.vstack(spix_arr) if spix_arr else np.zeros((0, len(raw_metrics)), dtype=float)
    arr_both = np.vstack([arr_grid, arr_spix]) if arr_grid.size and arr_spix.size else np.zeros((0, len(raw_metrics)), dtype=float)
    if arr_both.size:
        col_mean = np.nanmean(arr_both, axis=0, keepdims=True)
        col_std = np.nanstd(arr_both, axis=0, keepdims=True)
        col_std[col_std == 0] = 1.0
        z_grid = (arr_grid - col_mean) / col_std
        z_spix = (arr_spix - col_mean) / col_std
        im_grid = ax_grid.imshow(z_grid, aspect="auto", cmap="coolwarm", vmin=-2.0, vmax=2.0)
        im_spix = ax_spix.imshow(z_spix, aspect="auto", cmap="coolwarm", vmin=-2.0, vmax=2.0)
        fig.colorbar(im_spix, ax=[ax_grid, ax_spix], fraction=0.03, pad=0.02).set_label("Oriented z-score")

        xticks = range(len(raw_metrics))
        xticklabels = [_plot_reason_label(m) for m in raw_metrics]
        for ax, title in [(ax_grid, "GRID"), (ax_spix, "SPIX")]:
            ax.set_xticks(list(xticks))
            ax.set_xticklabels(xticklabels, rotation=20, ha="right", fontsize=10)
            ax.set_title(title, fontsize=15, pad=8)
            ax.tick_params(axis="x", pad=6)

        ax_grid.set_yticks(range(len(top_heat.index)))
        ax_grid.set_yticklabels([str(g) for g in top_heat.index], fontsize=11)
        ax_grid.set_ylabel("Top SPIX-specific genes")
        ax_spix.tick_params(axis="y", left=False, labelleft=False)
        ax_grid.set_xlabel("")
        ax_spix.set_xlabel("")
        ax_grid.text(
            0.0,
            1.08,
            "Top genes: recoverability profile",
            transform=ax_grid.transAxes,
            ha="left",
            va="bottom",
            fontsize=17,
        )
    else:
        for ax in [ax_grid, ax_spix]:
            ax.text(0.5, 0.5, "No genes", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    return fig, np.array([ax_profile, ax_grid, ax_spix], dtype=object)


def _load_matched_segment_index(
    segments_index_csv: str,
    *,
    focus_compactness: float | Mapping[float, float] = 0.1,
    reference_compactness: float | Mapping[float, float] = 1e7,
    compactness_tol: float = 1e-9,
) -> Tuple[Path, pd.DataFrame]:
    index_path = Path(segments_index_csv).resolve()
    index_df = pd.read_csv(index_path)
    required = {"resolution", "compactness", "path"}
    missing = required.difference(index_df.columns)
    if missing:
        raise KeyError(f"segments_index.csv must contain columns: {sorted(missing)}")

    idx = index_df.copy()
    idx["resolution"] = pd.to_numeric(idx["resolution"], errors="coerce")
    idx["compactness"] = pd.to_numeric(idx["compactness"], errors="coerce")
    idx = idx.dropna(subset=["resolution", "compactness"])

    # Supplement missing CSV rows from on-disk segment bundles.
    file_rows: List[Dict[str, object]] = []
    pattern = re.compile(r"segments__r(?P<resolution>[^_]+)_c(?P<compactness>.+)\.npz$")
    for npz_path in sorted(index_path.parent.glob("segments__r*_c*.npz")):
        match = pattern.match(npz_path.name)
        if match is None:
            continue

        res_val = pd.to_numeric(pd.Series([match.group("resolution")]), errors="coerce").iloc[0]
        comp_val = pd.to_numeric(pd.Series([match.group("compactness")]), errors="coerce").iloc[0]
        if not np.isfinite(res_val) or not np.isfinite(comp_val):
            continue

        file_rows.append(
            {
                "scale_id": f"r{float(res_val):g}_c{float(comp_val):g}",
                "resolution": float(res_val),
                "compactness": float(comp_val),
                "n_segments": np.nan,
                "seconds": np.nan,
                "path": f"./multiscale_segments/{npz_path.name}",
            }
        )

    if file_rows:
        file_df = pd.DataFrame(file_rows)
        idx = pd.concat([idx, file_df], ignore_index=True, sort=False)
        idx = idx.drop_duplicates(subset=["resolution", "compactness"], keep="first")

    def _resolve_compactness_map(
        value: float | Mapping[float, float],
        label: str,
    ) -> Dict[float, float]:
        if isinstance(value, ABCMapping):
            resolved: Dict[float, float] = {}
            for k, v in value.items():
                try:
                    rk = float(k)
                    rv = float(v)
                except Exception as e:
                    raise ValueError(f"{label} compactness map must contain numeric resolution -> compactness values.") from e
                resolved[rk] = rv
            if not resolved:
                raise ValueError(f"{label} compactness map is empty.")
            return resolved
        return {float(res): float(value) for res in sorted(idx["resolution"].unique())}

    focus_map = _resolve_compactness_map(focus_compactness, "focus")
    ref_map = _resolve_compactness_map(reference_compactness, "reference")

    matched_rows: List[Dict[str, object]] = []
    available_res = np.asarray(sorted(idx["resolution"].unique()), dtype=float)
    for res in available_res:
        focus_target = focus_map.get(float(res), None)
        ref_target = ref_map.get(float(res), None)
        if focus_target is None or ref_target is None:
            continue

        focus_candidates = idx[
            (np.isclose(idx["resolution"], float(res), atol=float(compactness_tol)))
            & (np.isclose(idx["compactness"], float(focus_target), atol=float(compactness_tol)))
        ].copy()
        ref_candidates = idx[
            (np.isclose(idx["resolution"], float(res), atol=float(compactness_tol)))
            & (np.isclose(idx["compactness"], float(ref_target), atol=float(compactness_tol)))
        ].copy()
        if focus_candidates.empty or ref_candidates.empty:
            continue

        focus_row = focus_candidates.iloc[0]
        ref_row = ref_candidates.iloc[0]
        matched_rows.append(
            {
                "resolution": float(res),
                "scale_id_focus": focus_row.get("scale_id", None),
                "compactness_focus": float(focus_row["compactness"]),
                "path_focus": focus_row["path"],
                "scale_id_reference": ref_row.get("scale_id", None),
                "compactness_reference": float(ref_row["compactness"]),
                "path_reference": ref_row["path"],
            }
        )

    if not matched_rows:
        raise ValueError("Could not find matched focus/reference compactness entries in segments_index.csv")

    matched = pd.DataFrame(matched_rows).sort_values("resolution").reset_index(drop=True)
    if matched.empty:
        raise ValueError("No matched resolutions between focus and reference compactness groups")

    return index_path, matched


def _nearest_available_resolution(resolutions: np.ndarray, target: Optional[float]) -> float:
    vals = np.asarray(resolutions, dtype=float)
    if vals.size == 0:
        raise ValueError("No resolutions available")
    if target is None or not np.isfinite(target):
        target = float(np.median(vals))
    idx = int(np.argmin(np.abs(vals - float(target))))
    return float(vals[idx])


def _segment_expression_totals(values: np.ndarray, seg_codes: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (seg_codes >= 0)
    if not np.any(valid):
        return np.zeros(0, dtype=np.float64)

    codes = seg_codes[valid].astype(np.int64, copy=False)
    x = values[valid].astype(np.float64, copy=False)
    x = np.clip(x, 0.0, None)
    if not np.any(x > 0):
        return np.zeros(0, dtype=np.float64)

    n_segments = int(codes.max(initial=-1)) + 1
    totals = np.bincount(codes, weights=x, minlength=n_segments).astype(np.float64, copy=False)
    totals = totals[totals > 0]
    return np.sort(totals)[::-1]


def _top_fraction_mass(signal: np.ndarray, *, top_fraction: float = 0.1) -> float:
    vals = np.asarray(signal, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.nan
    total = float(vals.sum())
    if total <= 0.0:
        return np.nan
    k = max(1, int(np.ceil(float(top_fraction) * int(vals.size))))
    return float(vals[:k].sum() / total)


def _capture_curve(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(signal, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    cum = np.cumsum(vals) / float(vals.sum())
    x = np.arange(1, vals.size + 1, dtype=float) / float(vals.size)
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], cum])


def _localization_auc(signal: np.ndarray) -> float:
    x, y = _capture_curve(signal)
    if x.size == 0 or y.size == 0:
        return np.nan
    return float(np.trapezoid(y - x, x))


def _half_mass_fraction(signal: np.ndarray) -> float:
    vals = np.asarray(signal, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.nan
    cum = np.cumsum(vals) / float(vals.sum())
    idx = int(np.searchsorted(cum, 0.5, side="left"))
    idx = min(max(idx, 0), vals.size - 1)
    return float((idx + 1) / float(vals.size))


def _effective_segment_count(signal: np.ndarray) -> float:
    vals = np.asarray(signal, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.nan
    probs = vals / float(vals.sum())
    denom = float(np.sum(probs * probs))
    if denom <= 0.0:
        return np.nan
    return float(1.0 / denom)


def _paired_wilcoxon_pvalue(x: pd.Series, y: pd.Series) -> float:
    pair_df = pd.concat(
        [pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")],
        axis=1,
    ).dropna()
    if pair_df.empty:
        return np.nan
    diff = pair_df.iloc[:, 1] - pair_df.iloc[:, 0]
    if np.allclose(diff.to_numpy(dtype=float), 0.0):
        return 1.0
    try:
        return float(
            wilcoxon(
                pair_df.iloc[:, 0].to_numpy(dtype=float),
                pair_df.iloc[:, 1].to_numpy(dtype=float),
                alternative="two-sided",
                zero_method="wilcox",
            ).pvalue
        )
    except ValueError:
        return np.nan


def _repel_annotations(
    ax: plt.Axes,
    annotations: List[plt.Annotation],
    *,
    anchor_points: Optional[np.ndarray] = None,
    avoid_bboxes: Optional[Iterable[Bbox]] = None,
    max_iter: int = 45,
    expand_x: float = 1.08,
    expand_y: float = 1.18,
    point_padding_px: float = 8.0,
    boundary_padding_px: float = 4.0,
) -> None:
    if not annotations:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_pt = 72.0 / float(fig.dpi)
    base_positions = [np.array(ann.get_position(), dtype=float) for ann in annotations]
    axes_bbox = ax.get_window_extent(renderer=renderer)
    inner_bbox = Bbox.from_extents(
        axes_bbox.x0 + boundary_padding_px,
        axes_bbox.y0 + boundary_padding_px,
        axes_bbox.x1 - boundary_padding_px,
        axes_bbox.y1 - boundary_padding_px,
    )
    static_bboxes: List[Bbox] = []
    if anchor_points is not None:
        pts = np.asarray(anchor_points, dtype=float)
        if pts.size:
            for x0, y0 in pts.reshape(-1, 2):
                static_bboxes.append(
                    Bbox.from_extents(
                        x0 - point_padding_px,
                        y0 - point_padding_px,
                        x0 + point_padding_px,
                        y0 + point_padding_px,
                    )
                )
    if avoid_bboxes is not None:
        static_bboxes.extend(list(avoid_bboxes))

    for _ in range(int(max_iter)):
        bboxes = [ann.get_window_extent(renderer=renderer).expanded(expand_x, expand_y) for ann in annotations]
        centers = np.array(
            [[0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)] for bb in bboxes],
            dtype=float,
        )
        shifts = np.zeros((len(annotations), 2), dtype=float)
        moved = False

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                bb_i = bboxes[i]
                bb_j = bboxes[j]
                overlap_x = min(bb_i.x1, bb_j.x1) - max(bb_i.x0, bb_j.x0)
                overlap_y = min(bb_i.y1, bb_j.y1) - max(bb_i.y0, bb_j.y0)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                moved = True
                delta = centers[i] - centers[j]
                if np.allclose(delta, 0.0):
                    delta = np.array([1.0 if (i + j) % 2 == 0 else -1.0, 1.0 if i % 2 == 0 else -1.0], dtype=float)
                norm = float(np.hypot(delta[0], delta[1]))
                direction = delta / max(norm, 1e-6)
                push = direction * (max(overlap_x, overlap_y) * 0.12)
                shifts[i] += push
                shifts[j] -= push

        for i, bb_i in enumerate(bboxes):
            center_i = centers[i]
            for static_bb in static_bboxes:
                overlap_x = min(bb_i.x1, static_bb.x1) - max(bb_i.x0, static_bb.x0)
                overlap_y = min(bb_i.y1, static_bb.y1) - max(bb_i.y0, static_bb.y0)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                moved = True
                static_center = np.array(
                    [0.5 * (static_bb.x0 + static_bb.x1), 0.5 * (static_bb.y0 + static_bb.y1)],
                    dtype=float,
                )
                delta = center_i - static_center
                if np.allclose(delta, 0.0):
                    delta = np.array([1.0 if i % 2 == 0 else -1.0, 1.0], dtype=float)
                norm = float(np.hypot(delta[0], delta[1]))
                direction = delta / max(norm, 1e-6)
                shifts[i] += direction * (max(overlap_x, overlap_y) * 0.16)

            if bb_i.x0 < inner_bbox.x0:
                moved = True
                shifts[i, 0] += (inner_bbox.x0 - bb_i.x0) * 0.45
            if bb_i.x1 > inner_bbox.x1:
                moved = True
                shifts[i, 0] -= (bb_i.x1 - inner_bbox.x1) * 0.45
            if bb_i.y0 < inner_bbox.y0:
                moved = True
                shifts[i, 1] += (inner_bbox.y0 - bb_i.y0) * 0.45
            if bb_i.y1 > inner_bbox.y1:
                moved = True
                shifts[i, 1] -= (bb_i.y1 - inner_bbox.y1) * 0.45

            current_pos = np.array(annotations[i].get_position(), dtype=float)
            spring = (base_positions[i] - current_pos) / max(px_to_pt, 1e-6)
            shifts[i] += spring * 0.08

        if not moved:
            break

        for ann, shift in zip(annotations, shifts):
            pos = np.array(ann.get_position(), dtype=float)
            new_pos = pos + shift * px_to_pt
            new_pos = np.clip(new_pos, -70.0, 70.0)
            ann.set_position((float(new_pos[0]), float(new_pos[1])))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def _concentration_plot_rc() -> Dict[str, object]:
    return {
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.weight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 16,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12,
    }


def _select_example_genes(
    score_df: pd.DataFrame,
    *,
    top_n_examples: int,
    rank_col: str,
    example_genes: Optional[Iterable[str]] = None,
) -> Tuple[List[str], pd.DataFrame]:
    score_sort_cols = ["concentration_gain"]
    ascending = [False]
    if rank_col in score_df.columns:
        score_sort_cols.append(rank_col)
        ascending.append(False)

    ordered = score_df.sort_values(score_sort_cols, ascending=ascending)
    ordered_index = ordered.index.tolist()
    if example_genes is None:
        selected_genes = ordered_index[:top_n_examples]
    else:
        requested = [str(g) for g in example_genes if str(g) in score_df.index]
        seen = set()
        selected_genes = []
        for gene in requested:
            if gene not in seen:
                selected_genes.append(gene)
                seen.add(gene)
        for gene in ordered_index:
            if len(selected_genes) >= top_n_examples:
                break
            if gene not in seen:
                selected_genes.append(gene)
                seen.add(gene)
    return selected_genes[:top_n_examples], ordered


def _draw_expression_concentration_curves(
    axes: List[plt.Axes],
    curve_df: pd.DataFrame,
    score_df: pd.DataFrame,
    *,
    example_genes: List[str],
    top_fraction: float,
    ncols: int,
) -> None:
    method_colors = {"GRID": "#d95f02", "SPIX": "#1b9e77"}
    top_n_examples = len(example_genes)
    ncols = max(1, int(ncols))

    for i, gene in enumerate(example_genes):
        ax = axes[i]
        gene_curve = curve_df[curve_df["gene"] == gene].copy()
        if gene_curve.empty:
            ax.set_axis_off()
            continue

        for method in ["GRID", "SPIX"]:
            sub = gene_curve[gene_curve["method"] == method].sort_values("segment_fraction")
            ax.plot(
                pd.to_numeric(sub["segment_fraction"], errors="coerce"),
                pd.to_numeric(sub["captured_expression_fraction"], errors="coerce"),
                color=method_colors[method],
                linewidth=2.1,
                label=method,
            )

            if gene in score_df.index:
                score_val = float(pd.to_numeric(pd.Series([score_df.loc[gene, f"concentration_score_{method}"]]), errors="coerce").iloc[0])
                ax.scatter(
                    [float(top_fraction)],
                    [score_val],
                    color=method_colors[method],
                    s=42,
                    zorder=4,
                )

        grid_score = float(pd.to_numeric(pd.Series([score_df.loc[gene, "concentration_score_GRID"]]), errors="coerce").iloc[0])
        spix_score = float(pd.to_numeric(pd.Series([score_df.loc[gene, "concentration_score_SPIX"]]), errors="coerce").iloc[0])
        ax.axvline(float(top_fraction), color="#7f7f7f", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Top-ranked segment fraction")
        ax.set_ylabel("Captured expression")
        if i == 0:
            ax.legend(frameon=False, loc="lower right")
        ax.grid(alpha=0.2)
        ax.set_title(str(gene), fontsize=13, pad=7, fontweight="bold")
        ax.text(
            0.03,
            0.97,
            f"Top {int(round(float(top_fraction) * 100))}% score\nG {grid_score:.2f}  S {spix_score:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.6,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#dddddd", alpha=0.9),
        )
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    for ax in axes[top_n_examples:]:
        ax.set_axis_off()


def _draw_expression_concentration_scatter(
    ax: plt.Axes,
    score_df: pd.DataFrame,
    *,
    rank_col: str,
    top_fraction: float,
    label_all: bool = True,
    max_repel_iter: int = 60,
) -> Tuple[pd.DataFrame, List[plt.Annotation]]:
    x = pd.to_numeric(score_df["concentration_score_GRID"], errors="coerce")
    y = pd.to_numeric(score_df["concentration_score_SPIX"], errors="coerce")
    plot_df = pd.DataFrame({"x": x, "y": y}, index=score_df.index).dropna()
    if plot_df.empty:
        raise ValueError("No valid concentration scores available for plotting")

    fig = ax.figure
    if rank_col in score_df.columns:
        color_vals = pd.to_numeric(score_df.loc[plot_df.index, rank_col], errors="coerce").fillna(0.0)
        sc = ax.scatter(
            plot_df["x"],
            plot_df["y"],
            c=color_vals,
            cmap="viridis",
            s=68,
            edgecolors="white",
            linewidths=0.55,
            alpha=0.9,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(_pretty_label(rank_col), fontsize=11, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)
    else:
        ax.scatter(
            plot_df["x"],
            plot_df["y"],
            color="#4c78a8",
            s=68,
            edgecolors="white",
            linewidths=0.55,
            alpha=0.9,
        )

    lo = float(np.nanmin(plot_df[["x", "y"]].to_numpy(dtype=float)))
    hi = float(np.nanmax(plot_df[["x", "y"]].to_numpy(dtype=float)))
    pad = max(0.03, 0.08 * (hi - lo if hi > lo else 1.0))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#7f7f7f", linewidth=0.9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    pct = int(round(float(top_fraction) * 100))
    ax.set_xlabel(f"GRID concentration score (top {pct}% mass)", fontsize=19, labelpad=10)
    ax.set_ylabel(f"SPIX concentration score (top {pct}% mass)", fontsize=19, labelpad=10)
    ax.set_title("Quantification Across Scale-top Genes", fontsize=15, pad=14, fontweight="bold")
    ax.grid(alpha=0.2)

    annotations: List[plt.Annotation] = []
    summary_text = None
    median_grid = float(plot_df["x"].median())
    median_spix = float(plot_df["y"].median())
    frac_improved = float((plot_df["y"] > plot_df["x"]).mean())
    median_gain = float((plot_df["y"] - plot_df["x"]).median())
    summary_text = ax.text(
        0.02,
        0.98,
        f"Median GRID {median_grid:.2f}\nMedian SPIX {median_spix:.2f}\nMedian gain {median_gain:.2f}\nSPIX > GRID {frac_improved:.0%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#dddddd", alpha=0.95),
        zorder=5,
    )

    if label_all:
        center_x = float(plot_df["x"].mean())
        center_y = float(plot_df["y"].mean())
        for idx, gene in enumerate(plot_df.index.tolist()):
            dx = float(plot_df.loc[gene, "x"]) - center_x
            dy = float(plot_df.loc[gene, "y"]) - center_y
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                angle = 2.0 * np.pi * idx / max(len(plot_df), 1)
                dx = float(np.cos(angle))
                dy = float(np.sin(angle))
            offset = (
                10.0 * np.sign(dx if dx != 0 else 1.0),
                8.0 * np.sign(dy if dy != 0 else 1.0),
            )
            ann = ax.annotate(
                str(gene),
                (float(plot_df.loc[gene, "x"]), float(plot_df.loc[gene, "y"])),
                textcoords="offset points",
                xytext=offset,
                fontsize=11,
                fontweight="bold",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.78),
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, alpha=0.8),
            )
            annotations.append(ann)
        fig.canvas.draw()
        avoid_bboxes = [summary_text.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.05, 1.10)]
        anchor_points = ax.transData.transform(plot_df[["x", "y"]].to_numpy(dtype=float))
        _repel_annotations(
            ax,
            annotations,
            anchor_points=anchor_points,
            avoid_bboxes=avoid_bboxes,
            max_iter=max_repel_iter,
        )
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    return plot_df, annotations


def collect_expression_concentration_profiles(
    adata,
    segments_index_csv: str,
    genes: Iterable[str],
    *,
    rank_summary_df: Optional[pd.DataFrame] = None,
    resolution: Optional[float] = None,
    focus_compactness: float | Mapping[float, float] = 0.1,
    reference_compactness: float | Mapping[float, float] = 1e7,
    layer: Optional[str] = None,
    top_fraction: float = 0.1,
    compactness_tol: float = 1e-9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build expression-concentration curves and per-gene concentration scores.

    The concentration score is the fraction of total segment-level expression
    captured by the top `top_fraction` of segments after sorting segments by
    expression mass in descending order.
    """

    index_path, matched = _load_matched_segment_index(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )

    genes = [str(g) for g in genes if str(g) in adata.var_names]
    if not genes:
        raise ValueError("No input genes were found in adata.var_names")

    resolution_vals = matched["resolution"].to_numpy(dtype=float)
    seg_cache: Dict[str, Dict[str, np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    curve_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []

    rank_df = rank_summary_df.copy() if rank_summary_df is not None else None

    for gene in genes:
        target_resolution = resolution
        if (
            target_resolution is None
            and rank_df is not None
            and gene in rank_df.index
            and "best_resolution" in rank_df.columns
        ):
            target_resolution = pd.to_numeric(pd.Series([rank_df.loc[gene, "best_resolution"]]), errors="coerce").iloc[0]

        selected_resolution = _nearest_available_resolution(resolution_vals, target_resolution)
        row = matched.iloc[int(np.argmin(np.abs(resolution_vals - selected_resolution)))]
        values = _extract_gene_vector(adata, gene, layer=layer)

        gene_scores: Dict[str, object] = {
            "gene": gene,
            "resolution": float(selected_resolution),
            "top_fraction": float(top_fraction),
        }

        for method, path_col, compactness in [
            ("SPIX", "path_focus", float(row["compactness_focus"])),
            ("GRID", "path_reference", float(row["compactness_reference"])),
        ]:
            seg_path = _resolve_segment_path(str(row[path_col]), index_path.parent)
            seg_key = str(seg_path)
            if seg_key not in seg_cache:
                seg_cache[seg_key] = _load_segmentation_cache(seg_path)

            seg_codes = seg_cache[seg_key]["seg_codes"]  # type: ignore[index]
            signal = _segment_expression_totals(values, seg_codes)  # type: ignore[arg-type]
            x_curve, y_curve = _capture_curve(signal)
            score = _top_fraction_mass(signal, top_fraction=top_fraction)
            localization_auc = _localization_auc(signal)
            half_mass_fraction = _half_mass_fraction(signal)
            effective_segments = _effective_segment_count(signal)
            gene_scores[f"concentration_score_{method}"] = float(score)
            gene_scores[f"localization_auc_{method}"] = float(localization_auc)
            gene_scores[f"half_mass_fraction_{method}"] = float(half_mass_fraction)
            gene_scores[f"effective_segments_{method}"] = float(effective_segments)
            gene_scores[f"n_segments_{method}"] = int(signal.size)
            gene_scores[f"compactness_{method}"] = float(compactness)

            for x_val, y_val in zip(x_curve, y_curve):
                curve_rows.append(
                    {
                        "gene": gene,
                        "method": method,
                        "resolution": float(selected_resolution),
                        "top_fraction": float(top_fraction),
                        "segment_fraction": float(x_val),
                        "captured_expression_fraction": float(y_val),
                        "concentration_score": float(score),
                    }
                )

        gene_scores["concentration_gain"] = float(
            pd.to_numeric(pd.Series([gene_scores["concentration_score_SPIX"]]), errors="coerce").iloc[0]
            - pd.to_numeric(pd.Series([gene_scores["concentration_score_GRID"]]), errors="coerce").iloc[0]
        )
        gene_scores["localization_auc_gain"] = float(
            pd.to_numeric(pd.Series([gene_scores["localization_auc_SPIX"]]), errors="coerce").iloc[0]
            - pd.to_numeric(pd.Series([gene_scores["localization_auc_GRID"]]), errors="coerce").iloc[0]
        )
        gene_scores["half_mass_fraction_gain"] = float(
            pd.to_numeric(pd.Series([gene_scores["half_mass_fraction_GRID"]]), errors="coerce").iloc[0]
            - pd.to_numeric(pd.Series([gene_scores["half_mass_fraction_SPIX"]]), errors="coerce").iloc[0]
        )
        gene_scores["effective_segments_gain"] = float(
            pd.to_numeric(pd.Series([gene_scores["effective_segments_GRID"]]), errors="coerce").iloc[0]
            - pd.to_numeric(pd.Series([gene_scores["effective_segments_SPIX"]]), errors="coerce").iloc[0]
        )
        score_rows.append(gene_scores)

    curve_df = pd.DataFrame(curve_rows)
    score_df = pd.DataFrame(score_rows).set_index("gene")
    if rank_df is not None:
        extra_cols = [c for c in rank_df.columns if c not in score_df.columns]
        if extra_cols:
            score_df = score_df.join(rank_df[extra_cols], how="left")
    return curve_df, score_df


def plot_expression_concentration_curves(
    curve_df: pd.DataFrame,
    score_df: pd.DataFrame,
    *,
    top_n_examples: int = 3,
    top_fraction: float = 0.1,
    figsize: Tuple[float, float] = (15.0, 3.9),
    title_suffix: Optional[str] = None,
    example_genes: Optional[Iterable[str]] = None,
) -> Tuple[plt.Figure, np.ndarray, List[str]]:
    """
    Plot representative expression-concentration curves only.
    """

    if curve_df.empty or score_df.empty:
        raise ValueError("curve_df and score_df must be non-empty")

    top_n_examples = max(1, int(top_n_examples))
    example_genes, _ = _select_example_genes(
        score_df,
        top_n_examples=top_n_examples,
        rank_col="contrast_SPIX_vs_GRID",
        example_genes=example_genes,
    )
    top_n_examples = len(example_genes)
    ncols = min(5, top_n_examples)
    nrows = int(np.ceil(top_n_examples / ncols))

    with plt.rc_context(_concentration_plot_rc()):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(max(float(figsize[0]), 4.3 * ncols), max(float(figsize[1]), 3.6 * nrows + 0.3)),
            constrained_layout=True,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes], dtype=object)
        flat_axes = list(np.ravel(axes))
        _draw_expression_concentration_curves(
            flat_axes,
            curve_df,
            score_df,
            example_genes=example_genes,
            top_fraction=top_fraction,
            ncols=ncols,
        )
        title = "Expression Concentration Curves"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        fig.suptitle(title, fontsize=20, y=1.03, fontweight="bold")
        return fig, axes, example_genes


def plot_expression_concentration_scatter(
    score_df: pd.DataFrame,
    *,
    rank_col: str = "contrast_SPIX_vs_GRID",
    top_fraction: float = 0.1,
    figsize: Tuple[float, float] = (9.0, 6.3),
    title_suffix: Optional[str] = None,
    label_all: bool = True,
    max_repel_iter: int = 60,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the concentration-score comparison scatter only.
    """

    if score_df.empty:
        raise ValueError("score_df must be non-empty")

    with plt.rc_context(_concentration_plot_rc()):
        fig, ax = plt.subplots(
            figsize=(max(float(figsize[0]), 8.8), max(float(figsize[1]), 6.2)),
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.08, hspace=0.08, wspace=0.04)
        _draw_expression_concentration_scatter(
            ax,
            score_df,
            rank_col=rank_col,
            top_fraction=top_fraction,
            label_all=label_all,
            max_repel_iter=max_repel_iter,
        )
        title = "Concentration Score Comparison"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title, fontsize=16, pad=18, fontweight="bold")
        return fig, ax


def plot_expression_concentration_metric_summary(
    score_df: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
    figsize: Tuple[float, float] = (15.5, 8.6),
    title_suffix: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot paired metric summaries for concentration-related quantifications.
    """

    if score_df.empty:
        raise ValueError("score_df must be non-empty")

    metric_list = list(metrics) if metrics is not None else [
        "concentration_score",
        "localization_auc",
        "half_mass_fraction",
        "effective_segments",
    ]
    metric_list = [m for m in metric_list if f"{m}_GRID" in score_df.columns and f"{m}_SPIX" in score_df.columns]
    if not metric_list:
        raise ValueError("No valid paired concentration metrics available for plotting")

    ncols = min(2, len(metric_list))
    nrows = int(np.ceil(len(metric_list) / ncols))
    method_colors = {"GRID": "#d95f02", "SPIX": "#1b9e77"}
    lower_is_better = {"half_mass_fraction", "effective_segments"}

    with plt.rc_context(_concentration_plot_rc()):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(max(float(figsize[0]), 7.2 * ncols), max(float(figsize[1]), 4.7 * nrows)),
            constrained_layout=True,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes], dtype=object)
        flat_axes = list(np.ravel(axes))

        for ax, metric in zip(flat_axes, metric_list):
            grid_col = f"{metric}_GRID"
            spix_col = f"{metric}_SPIX"
            pair_df = pd.DataFrame(
                {
                    "GRID": pd.to_numeric(score_df[grid_col], errors="coerce"),
                    "SPIX": pd.to_numeric(score_df[spix_col], errors="coerce"),
                },
                index=score_df.index,
            ).dropna()
            if pair_df.empty:
                ax.set_axis_off()
                continue

            xs = np.array([0.0, 1.0], dtype=float)
            for gene, row in pair_df.iterrows():
                ax.plot(xs, row.to_numpy(dtype=float), color="#c7c7c7", linewidth=1.0, alpha=0.8, zorder=1)
                ax.scatter(xs[0], float(row["GRID"]), s=44, color=method_colors["GRID"], edgecolors="white", linewidths=0.5, zorder=2)
                ax.scatter(xs[1], float(row["SPIX"]), s=44, color=method_colors["SPIX"], edgecolors="white", linewidths=0.5, zorder=2)

            med_grid = float(pair_df["GRID"].median())
            med_spix = float(pair_df["SPIX"].median())
            benefit = (pair_df["GRID"] - pair_df["SPIX"]) if metric in lower_is_better else (pair_df["SPIX"] - pair_df["GRID"])
            pval = _paired_wilcoxon_pvalue(pair_df["GRID"], pair_df["SPIX"])
            improved = float((benefit > 0).mean())

            ax.scatter(xs[0], med_grid, s=110, color="#7a1f00", marker="_", linewidths=3.0, zorder=3)
            ax.scatter(xs[1], med_spix, s=110, color="#005a46", marker="_", linewidths=3.0, zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels(["GRID", "SPIX"], fontweight="bold")
            ax.set_ylabel(_pretty_label(metric))
            title = _pretty_label(metric)
            if metric in lower_is_better:
                title = f"{title} (lower is better)"
            ax.set_title(title, pad=12, fontweight="bold")
            ax.grid(alpha=0.2, axis="y")
            ax.text(
                0.03,
                0.97,
                f"Median G {med_grid:.3g}\nMedian S {med_spix:.3g}\nBenefit {benefit.median():.3g}\nSPIX better {improved:.0%}\nWilcoxon p={pval:.2g}" if np.isfinite(pval) else
                f"Median G {med_grid:.3g}\nMedian S {med_spix:.3g}\nBenefit {benefit.median():.3g}\nSPIX better {improved:.0%}\nWilcoxon p=NA",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.2,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#dddddd", alpha=0.95),
            )
            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")

        for ax in flat_axes[len(metric_list):]:
            ax.set_axis_off()

        title = "Concentration Metric Summary"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        fig.suptitle(title, fontsize=20, y=1.02, fontweight="bold")
        return fig, axes


def plot_expression_concentration_story(
    curve_df: pd.DataFrame,
    score_df: pd.DataFrame,
    *,
    rank_col: str = "contrast_SPIX_vs_GRID",
    top_n_examples: int = 3,
    top_fraction: float = 0.1,
    figsize: Tuple[float, float] = (15.0, 8.5),
    title_suffix: Optional[str] = None,
    example_genes: Optional[Iterable[str]] = None,
    label_all: bool = True,
    max_repel_iter: int = 60,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot the concept curves and quantification scatter in one figure.
    """

    if curve_df.empty or score_df.empty:
        raise ValueError("curve_df and score_df must be non-empty")

    top_n_examples = max(1, int(top_n_examples))
    example_genes, _ = _select_example_genes(
        score_df,
        top_n_examples=top_n_examples,
        rank_col=rank_col,
        example_genes=example_genes,
    )
    top_n_examples = len(example_genes)
    ncols = min(5, top_n_examples)
    curve_nrows = int(np.ceil(top_n_examples / ncols))

    with plt.rc_context(_concentration_plot_rc()):
        fig = plt.figure(
            figsize=(max(float(figsize[0]), 4.3 * ncols), max(float(figsize[1]), 3.5 * curve_nrows + 5.0)),
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.11, hspace=0.10, wspace=0.06)
        gs = fig.add_gridspec(curve_nrows + 1, ncols, height_ratios=[0.95] * curve_nrows + [1.32])
        axes_top: List[plt.Axes] = []
        for i in range(top_n_examples):
            row = i // ncols
            col = i % ncols
            if not axes_top:
                axes_top.append(fig.add_subplot(gs[row, col]))
            else:
                axes_top.append(fig.add_subplot(gs[row, col]))
        ax_quant = fig.add_subplot(gs[curve_nrows, :])

        _draw_expression_concentration_curves(
            axes_top,
            curve_df,
            score_df,
            example_genes=example_genes,
            top_fraction=top_fraction,
            ncols=ncols,
        )
        _draw_expression_concentration_scatter(
            ax_quant,
            score_df,
            rank_col=rank_col,
            top_fraction=top_fraction,
            label_all=label_all,
            max_repel_iter=max_repel_iter,
        )

        title = "Expression Concentration Explains Why SPIX Recovers These Genes"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        fig.suptitle(title, fontsize=20, y=1.03, fontweight="bold")
        return fig, np.array(axes_top + [ax_quant], dtype=object)
