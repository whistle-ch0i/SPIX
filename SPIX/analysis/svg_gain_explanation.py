from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping as ABCMapping
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
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
    "pearson_r": "Pearson r",
    "spearman_r": "Spearman r",
    "gain_neighbor_contrast": "Boundary gain",
    "gain_segment_snr": "Segment SNR gain",
    "gain_segment_reconstruction_r2": "Reconstruction R^2 gain",
    "gain_segment_moranI": "Segment Moran's I gain",
    "gain_segment_moranI_sizeaware": "Size-aware segment autocorr gain",
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
    "segment_reconstruction_r2": "Segment reconstruction R^2",
    "segment_moranI": "Segment Moran's I",
    "segment_moranI_sizeaware": "Size-aware segment autocorr",
    "marker_purity": "Marker purity",
    "marker_contamination": "Marker contamination",
    "marker_margin": "Marker-class margin",
    "marker_entropy_purity": "Marker entropy-purity",
    "marker_neighbor_purity": "Marker neighborhood purity",
    "marker_coverage": "Marker coverage",
    "gain_marker_purity": "Marker purity gain",
    "gain_marker_contamination": "Marker contamination reduction",
    "gain_marker_margin": "Marker margin gain",
    "gain_marker_entropy_purity": "Marker entropy-purity gain",
    "gain_marker_neighbor_purity": "Marker neighborhood purity gain",
    "gain_marker_coverage": "Marker coverage gain",
    "label_purity": "Label purity",
    "label_contamination": "Label contamination",
    "label_margin": "Label-class margin",
    "label_entropy_purity": "Label entropy-purity",
    "label_neighbor_purity": "Label neighborhood purity",
    "label_coverage": "Label coverage",
    "gain_label_purity": "Label purity gain",
    "gain_label_contamination": "Label contamination reduction",
    "gain_label_margin": "Label margin gain",
    "gain_label_entropy_purity": "Label entropy-purity gain",
    "gain_label_neighbor_purity": "Label neighborhood purity gain",
    "gain_label_coverage": "Label coverage gain",
    "contrast_SPIX_vs_GRID": "SPIX rank advantage",
}


DEFAULT_REASON_GAIN_METRICS: List[str] = [
    "gain_within_segment_var",
    "gain_neighbor_contrast",
    "gain_between_segment_var",
    "gain_segment_reconstruction_r2",
    "gain_segment_dynamic_range",
    "gain_segment_top10_mass",
]


DEFAULT_REASON_RAW_METRICS: List[str] = [
    "within_segment_var",
    "neighbor_contrast",
    "between_segment_var",
    "segment_reconstruction_r2",
    "segment_dynamic_range",
    "segment_top10_mass",
]


ALL_SEGMENT_METRIC_NAMES: List[str] = [
    "within_segment_var",
    "between_segment_var",
    "segment_snr",
    "segment_reconstruction_r2",
    "neighbor_contrast",
    "segment_moranI",
    "segment_moranI_sizeaware",
    "segment_detection_rate",
    "segment_dynamic_range",
    "segment_top10_mass",
]


GAIN_DIRECTION_BY_RAW_METRIC: Dict[str, float] = {
    "within_segment_var": -1.0,
    "between_segment_var": 1.0,
    "segment_snr": 1.0,
    "segment_reconstruction_r2": 1.0,
    "neighbor_contrast": 1.0,
    "segment_moranI": 1.0,
    "segment_moranI_sizeaware": 1.0,
    "segment_detection_rate": 1.0,
    "segment_dynamic_range": 1.0,
    "segment_top10_mass": 1.0,
}


DEFAULT_REASON_GAIN_COLORS: Dict[str, str] = {
    "gain_within_segment_var": "#4c78a8",
    "gain_neighbor_contrast": "#f58518",
    "gain_between_segment_var": "#54a24b",
    "gain_segment_reconstruction_r2": "#72b7b2",
    "gain_segment_dynamic_range": "#b279a2",
    "gain_segment_top10_mass": "#e45756",
}


DISCOVERY_REASON_LABELS: Dict[str, str] = {
    "within_segment_var": "Low within-segment noise",
    "neighbor_contrast": "Sharp boundary separation",
    "between_segment_var": "Strong between-segment separation",
    "segment_reconstruction_r2": "High reconstruction fidelity",
    "segment_dynamic_range": "Broad segment dynamic range",
    "segment_top10_mass": "Localized segment concentration",
    "gain_within_segment_var": "Noise reduction gain",
    "gain_neighbor_contrast": "Boundary gain",
    "gain_between_segment_var": "Between-segment separation gain",
    "gain_segment_reconstruction_r2": "Reconstruction gain",
    "gain_segment_dynamic_range": "Dynamic-range gain",
    "gain_segment_top10_mass": "Localization gain",
}


DISCOVERY_PLOT_LABELS: Dict[str, str] = {
    "within_segment_var": "Within-segment homogeneity",
    "neighbor_contrast": "Boundary contrast",
    "between_segment_var": "Between-segment contrast",
    "segment_reconstruction_r2": "Reconstruction fidelity",
    "segment_dynamic_range": "Dynamic range",
    "segment_top10_mass": "Expression concentration",
    "gain_within_segment_var": "Homogeneity gain",
    "gain_neighbor_contrast": "Boundary gain",
    "gain_between_segment_var": "Between-segment gain",
    "gain_segment_reconstruction_r2": "Reconstruction gain",
    "gain_segment_dynamic_range": "Dynamic-range gain",
    "gain_segment_top10_mass": "Concentration gain",
}


DISCOVERY_REASON_DIRECTIONS: Dict[str, float] = {
    "within_segment_var": -1.0,
    "neighbor_contrast": 1.0,
    "between_segment_var": 1.0,
    "segment_reconstruction_r2": 1.0,
    "segment_dynamic_range": 1.0,
    "segment_top10_mass": 1.0,
}


CRC_VISIUMHD_MARKER_PANELS: Dict[str, Dict[str, List[str]]] = {
    "broad": {
        "normal_secretory": ["PIGR", "MUC2", "FCGBP", "CLCA1", "REG4"],
        "tumor_epithelial": ["CEACAM6", "CEACAM5", "EPCAM", "KRT8", "KRT18", "KRT19", "TACSTD2"],
        "caf_stroma": ["COL1A1", "COL3A1", "DCN", "LUM", "FAP"],
        "myeloid": ["C1QC", "C1QA", "LYZ", "FCN1", "VCAN", "SPP1"],
        "lymphoid": ["CD3D", "CD3E", "MS4A1", "CD79A", "MZB1", "JCHAIN"],
    },
    "desmoplastic": {
        "tumor_epithelial": ["CEACAM6", "CEACAM5", "EPCAM", "KRT8", "KRT18"],
        "tumor_emt_like": ["MSLN", "KLK10", "TGFBI", "LCN2"],
        "caf_desmoplastic": ["FAP", "COL1A1", "COL3A1", "INHBA", "WNT5A"],
        "caf_cthrc1": ["CTHRC1", "WNT5A", "COMP", "THBS2"],
        "mac_spp1": ["SPP1", "MARCO", "PPARG", "MRC1"],
        "mac_resident": ["C1QC", "C1QA", "SELENOP", "FOLR2", "APOE", "CD163"],
    },
    "immune_tls": {
        "t_cell": ["CD3D", "CD3E", "TRAC", "IL7R"],
        "cytotoxic": ["NKG7", "CCL5", "PRF1", "GNLY", "GZMB"],
        "b_cell": ["MS4A1", "CD79A", "CD74"],
        "plasma": ["MZB1", "JCHAIN", "IGKC"],
        "dc_apc": ["FCER1A", "CD1C", "LAMP3", "CCR7"],
    },
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


def _extract_gene_matrix(adata, genes: Iterable[str], *, layer: Optional[str] = None) -> np.ndarray:
    genes_use = [str(g) for g in genes if str(g) in adata.var_names]
    if not genes_use:
        return np.zeros((int(adata.n_obs), 0), dtype=np.float64)

    mat = adata.layers[layer] if layer is not None else adata.X
    X = mat[:, [adata.var_names.get_loc(g) for g in genes_use]]
    if issparse(X):
        return X.tocsr(copy=False)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


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
        if "label_img" in z.files:
            label_img = z["label_img"].astype(np.int32, copy=False)
            src, dst, weight = _boundary_edges_from_label_img(label_img)
        else:
            src = np.zeros(0, dtype=np.int32)
            dst = np.zeros(0, dtype=np.int32)
            weight = np.zeros(0, dtype=np.float64)
    obs_idx = np.flatnonzero(seg_codes >= 0).astype(np.int64, copy=False)
    codes_valid = seg_codes[obs_idx].astype(np.int64, copy=False)
    n_segments = int(codes_valid.max(initial=-1)) + 1
    if obs_idx.size > 0 and n_segments > 0:
        row_idx = np.arange(obs_idx.size, dtype=np.int64)
        obs_to_seg = csr_matrix(
            (
                np.ones(obs_idx.size, dtype=np.float64),
                (row_idx, codes_valid),
            ),
            shape=(obs_idx.size, n_segments),
        )
    else:
        obs_to_seg = csr_matrix((0, 0), dtype=np.float64)
    seg_counts = np.asarray(obs_to_seg.sum(axis=0)).reshape(-1).astype(np.float64, copy=False)
    return {
        "seg_codes": seg_codes,
        "src": src,
        "dst": dst,
        "weight": weight,
        "obs_idx": obs_idx,
        "obs_to_seg": obs_to_seg,
        "n_segments": np.int64(n_segments),
        "seg_counts": seg_counts,
    }


def _empty_metric_arrays(
    n_genes: int,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    names = list(ALL_SEGMENT_METRIC_NAMES if metric_names is None else metric_names)
    return {str(name): np.full(n_genes, np.nan, dtype=np.float64) for name in names}


def _segment_metrics_batch(
    values: np.ndarray,
    seg_info: Dict[str, object],
    *,
    metric_names: Optional[Iterable[str]] = None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    requested = list(ALL_SEGMENT_METRIC_NAMES if metric_names is None else [str(m) for m in metric_names])
    is_sparse_values = issparse(values)
    if is_sparse_values:
        vals = values.tocsr(copy=False)
        n_genes = int(vals.shape[1])
    else:
        vals = np.asarray(values, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        n_genes = int(vals.shape[1])
    if n_genes == 0:
        return _empty_metric_arrays(0, requested)

    obs_idx = np.asarray(seg_info["obs_idx"], dtype=np.int64)
    obs_to_seg = seg_info["obs_to_seg"]
    src = np.asarray(seg_info["src"], dtype=np.int32)
    dst = np.asarray(seg_info["dst"], dtype=np.int32)
    weight = np.asarray(seg_info["weight"], dtype=np.float64)
    n_segments = int(seg_info["n_segments"])

    out = _empty_metric_arrays(n_genes, requested)
    if obs_idx.size == 0 or n_segments <= 0 or getattr(obs_to_seg, "shape", (0, 0))[0] == 0:
        return out

    requested_set = set(requested)
    need_within = bool({"within_segment_var", "segment_snr", "segment_reconstruction_r2"} & requested_set)
    need_between = bool({"between_segment_var", "segment_snr", "segment_reconstruction_r2"} & requested_set)
    need_detection = "segment_detection_rate" in requested_set
    need_dynamic = "segment_dynamic_range" in requested_set
    need_top10 = "segment_top10_mass" in requested_set
    need_neighbor = "neighbor_contrast" in requested_set
    need_moran = "segment_moranI" in requested_set
    need_moran_sizeaware = "segment_moranI_sizeaware" in requested_set
    need_edge_metrics = need_neighbor or need_moran or need_moran_sizeaware

    if is_sparse_values:
        x = vals[obs_idx, :]
        x_sq = x.copy()
        if x_sq.nnz > 0:
            x_sq.data = x_sq.data.astype(np.float64, copy=False)
            x_sq.data **= 2
        prod_sum = obs_to_seg.T @ x
        prod_sum_sq = obs_to_seg.T @ x_sq
        sums = prod_sum.toarray().astype(np.float64, copy=False) if issparse(prod_sum) else np.asarray(prod_sum, dtype=np.float64)
        sums_sq = (
            prod_sum_sq.toarray().astype(np.float64, copy=False)
            if issparse(prod_sum_sq)
            else np.asarray(prod_sum_sq, dtype=np.float64)
        )
        counts_vec = np.asarray(seg_info["seg_counts"], dtype=np.float64)
        nz = counts_vec > 0
        total_count_scalar = float(counts_vec[nz].sum())
        total_count = np.full(n_genes, total_count_scalar, dtype=np.float64)
        valid_gene_mask = np.ones(n_genes, dtype=bool)
        means = np.zeros_like(sums)
        if np.any(nz):
            means[nz, :] = sums[nz, :] / counts_vec[nz, None]
        global_mean = sums[nz, :].sum(axis=0) / max(total_count_scalar, eps) if np.any(nz) else np.zeros(n_genes, dtype=np.float64)
        within_ss = sums_sq - 2.0 * means * sums + counts_vec[:, None] * means * means
        if need_within:
            within_var = within_ss.sum(axis=0) / max(total_count_scalar, eps)
        else:
            within_var = None
        if need_between:
            between_var = (
                (counts_vec[:, None] * (means - global_mean[None, :]) ** 2)[nz, :].sum(axis=0) / max(total_count_scalar, eps)
                if np.any(nz) else np.full(n_genes, np.nan, dtype=np.float64)
            )
        else:
            between_var = None
        if "within_segment_var" in requested_set:
            out["within_segment_var"] = np.asarray(within_var, dtype=np.float64)
        if "between_segment_var" in requested_set:
            out["between_segment_var"] = np.asarray(between_var, dtype=np.float64)
        if "segment_snr" in requested_set:
            out["segment_snr"] = np.asarray(between_var, dtype=np.float64) / (np.asarray(within_var, dtype=np.float64) + eps)
        if "segment_reconstruction_r2" in requested_set:
            out["segment_reconstruction_r2"] = (
                np.asarray(between_var, dtype=np.float64)
                / np.maximum(np.asarray(within_var, dtype=np.float64) + np.asarray(between_var, dtype=np.float64), eps)
            )
        nz_count = float(np.sum(nz))
        if need_detection and nz_count > 0:
            out["segment_detection_rate"] = ((means > 0)[nz, :].sum(axis=0) / nz_count).astype(np.float64, copy=False)
    else:
        x = vals[obs_idx, :]
        finite = np.isfinite(x)
        x0 = np.where(finite, x, 0.0)
        x_sq = x0 * x0
        valid_w = finite.astype(np.float64, copy=False)

        counts = np.asarray(obs_to_seg.T @ valid_w, dtype=np.float64)
        sums = np.asarray(obs_to_seg.T @ x0, dtype=np.float64)
        sums_sq = np.asarray(obs_to_seg.T @ x_sq, dtype=np.float64)

        nz = counts > 0
        total_count = counts.sum(axis=0)
        valid_gene_mask = total_count > 0
        if not np.any(valid_gene_mask):
            return out

        means = np.zeros_like(sums)
        np.divide(sums, counts, out=means, where=nz)

        global_mean = np.zeros(n_genes, dtype=np.float64)
        global_mean[valid_gene_mask] = sums[:, valid_gene_mask].sum(axis=0) / np.maximum(total_count[valid_gene_mask], eps)

        within_ss = sums_sq - 2.0 * means * sums + counts * means * means
        within_var = np.full(n_genes, np.nan, dtype=np.float64)
        between_var = np.full(n_genes, np.nan, dtype=np.float64)
        if need_within:
            within_var[valid_gene_mask] = (
                within_ss[:, valid_gene_mask].sum(axis=0) / np.maximum(total_count[valid_gene_mask], eps)
            )
            if "within_segment_var" in requested_set:
                out["within_segment_var"][valid_gene_mask] = within_var[valid_gene_mask]
        if need_between:
            between_var[valid_gene_mask] = (
                (counts[:, valid_gene_mask] * (means[:, valid_gene_mask] - global_mean[valid_gene_mask]) ** 2).sum(axis=0)
                / np.maximum(total_count[valid_gene_mask], eps)
            )
            if "between_segment_var" in requested_set:
                out["between_segment_var"][valid_gene_mask] = between_var[valid_gene_mask]
        if "segment_snr" in requested_set:
            out["segment_snr"][valid_gene_mask] = (
                between_var[valid_gene_mask] / (within_var[valid_gene_mask] + eps)
            )
        if "segment_reconstruction_r2" in requested_set:
            out["segment_reconstruction_r2"][valid_gene_mask] = (
                between_var[valid_gene_mask]
                / np.maximum(
                    within_var[valid_gene_mask] + between_var[valid_gene_mask],
                    eps,
                )
            )

        nz_count = nz.sum(axis=0)
        positive_mean = (means > 0) & nz
        if need_detection:
            out["segment_detection_rate"][valid_gene_mask] = (
                positive_mean[:, valid_gene_mask].sum(axis=0) / np.maximum(nz_count[valid_gene_mask], 1)
            )

    in_range = (src >= 0) & (dst >= 0) & (src < n_segments) & (dst < n_segments)
    src_in = src[in_range]
    dst_in = dst[in_range]
    weight_in = weight[in_range]

    if not (need_dynamic or need_top10 or need_edge_metrics):
        return out

    for j in np.flatnonzero(valid_gene_mask):
        nz_j = nz if getattr(nz, "ndim", 1) == 1 else nz[:, j]
        n_obs_seg = int(nz_j.sum())
        if need_dynamic and n_obs_seg > 1:
            means_j = means[nz_j, j]
            q10, q90 = np.nanpercentile(means_j, [10.0, 90.0])
            out["segment_dynamic_range"][j] = float(q90 - q10)
        elif need_dynamic and n_obs_seg == 1:
            out["segment_dynamic_range"][j] = 0.0
        else:
            means_j = means[nz_j, j]

        sums_j = sums[nz_j, j]
        total_sum_j = float(sums_j.sum())
        if need_top10 and total_sum_j > eps and n_obs_seg > 0:
            k = max(1, int(np.ceil(0.1 * n_obs_seg)))
            out["segment_top10_mass"][j] = float(np.sort(sums_j)[::-1][:k].sum() / total_sum_j)

        if not need_edge_metrics or src_in.size == 0:
            continue
        edge_mask = nz_j[src_in] & nz_j[dst_in]
        if not np.any(edge_mask):
            continue

        src_v = src_in[edge_mask]
        dst_v = dst_in[edge_mask]
        w_v = weight_in[edge_mask]
        means_full = means[:, j]
        if need_neighbor:
            out["neighbor_contrast"][j] = float(np.average(np.abs(means_full[src_v] - means_full[dst_v]), weights=w_v))

        if need_moran:
            centered = means_j - means_j.mean()
            denom = float(np.sum(centered ** 2))
            if denom > eps:
                centered_full = np.zeros(n_segments, dtype=np.float64)
                centered_full[nz_j] = centered
                num = float(2.0 * np.sum(w_v * centered_full[src_v] * centered_full[dst_v]))
                sum_w = float(2.0 * np.sum(w_v))
                out["segment_moranI"][j] = float((float(n_obs_seg) / max(sum_w, eps)) * (num / denom))

        counts_j = counts_vec if is_sparse_values else counts[:, j]
        if need_moran_sizeaware:
            centered_sizeaware = np.zeros(n_segments, dtype=np.float64)
            centered_sizeaware[nz_j] = means_j - np.average(means_j, weights=counts_j[nz_j])
            denom_sizeaware = float(np.sum(counts_j[nz_j] * centered_sizeaware[nz_j] ** 2))
            edge_weight_sizeaware = w_v * np.sqrt(counts_j[src_v] * counts_j[dst_v])
            if denom_sizeaware > eps and np.any(edge_weight_sizeaware > 0):
                num_sizeaware = float(
                    2.0 * np.sum(edge_weight_sizeaware * centered_sizeaware[src_v] * centered_sizeaware[dst_v])
                )
                sum_w_sizeaware = float(2.0 * np.sum(edge_weight_sizeaware))
                out["segment_moranI_sizeaware"][j] = float(
                    (total_count[j] / max(sum_w_sizeaware, eps)) * (num_sizeaware / denom_sizeaware)
                )

    return out


def _segment_metrics(
    values: np.ndarray,
    seg_codes: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    seg_codes_arr = np.asarray(seg_codes, dtype=np.int32)
    obs_idx = np.flatnonzero(seg_codes_arr >= 0).astype(np.int64, copy=False)
    codes_valid = seg_codes_arr[obs_idx].astype(np.int64, copy=False)
    n_segments = int(codes_valid.max(initial=-1)) + 1
    if obs_idx.size > 0 and n_segments > 0:
        row_idx = np.arange(obs_idx.size, dtype=np.int64)
        obs_to_seg = csr_matrix(
            (
                np.ones(obs_idx.size, dtype=np.float64),
                (row_idx, codes_valid),
            ),
            shape=(obs_idx.size, n_segments),
        )
    else:
        obs_to_seg = csr_matrix((0, 0), dtype=np.float64)

    batch = _segment_metrics_batch(
        np.asarray(values, dtype=np.float64).reshape(-1, 1),
        {
            "obs_idx": obs_idx,
            "obs_to_seg": obs_to_seg,
            "src": np.asarray(src, dtype=np.int32),
            "dst": np.asarray(dst, dtype=np.int32),
            "weight": np.asarray(weight, dtype=np.float64),
            "n_segments": np.int64(n_segments),
        },
        eps=eps,
    )
    return {k: float(v[0]) if len(v) else np.nan for k, v in batch.items()}


def _metric_arrays_for_batch(
    gene_batch: List[str],
    values_batch: np.ndarray,
    matched_row: Dict[str, object],
    metric_names: List[str],
) -> Dict[str, np.ndarray]:
    focus_metrics = _segment_metrics_batch(
        values_batch,
        matched_row["focus_seg"],  # type: ignore[arg-type]
        metric_names=metric_names,
    )
    ref_metrics = _segment_metrics_batch(
        values_batch,
        matched_row["ref_seg"],  # type: ignore[arg-type]
        metric_names=metric_names,
    )

    n_gene = len(gene_batch)
    out: Dict[str, np.ndarray] = {
        "resolution": np.full(n_gene, float(matched_row["resolution"]), dtype=np.float64),
        "focus_compactness": np.full(n_gene, float(matched_row["focus_compactness"]), dtype=np.float64),
        "reference_compactness": np.full(n_gene, float(matched_row["reference_compactness"]), dtype=np.float64),
    }
    for metric in metric_names:
        focus_val = np.asarray(focus_metrics[metric], dtype=np.float64)
        ref_val = np.asarray(ref_metrics[metric], dtype=np.float64)
        out[f"{metric}_SPIX"] = focus_val
        out[f"{metric}_GRID"] = ref_val

    out["gain_within_segment_var"] = out["within_segment_var_GRID"] - out["within_segment_var_SPIX"]
    out["gain_between_segment_var"] = out["between_segment_var_SPIX"] - out["between_segment_var_GRID"]
    out["gain_segment_snr"] = out["segment_snr_SPIX"] - out["segment_snr_GRID"]
    out["gain_segment_reconstruction_r2"] = (
        out["segment_reconstruction_r2_SPIX"] - out["segment_reconstruction_r2_GRID"]
    )
    out["gain_neighbor_contrast"] = out["neighbor_contrast_SPIX"] - out["neighbor_contrast_GRID"]
    out["gain_segment_moranI"] = out["segment_moranI_SPIX"] - out["segment_moranI_GRID"]
    out["gain_segment_moranI_sizeaware"] = (
        out["segment_moranI_sizeaware_SPIX"] - out["segment_moranI_sizeaware_GRID"]
    )
    out["gain_segment_detection_rate"] = out["segment_detection_rate_SPIX"] - out["segment_detection_rate_GRID"]
    out["gain_segment_dynamic_range"] = out["segment_dynamic_range_SPIX"] - out["segment_dynamic_range_GRID"]
    out["gain_segment_top10_mass"] = out["segment_top10_mass_SPIX"] - out["segment_top10_mass_GRID"]
    return out


def _normalize_gain_metric_request(metrics: Optional[Iterable[str]]) -> List[str]:
    if metrics is None:
        return [f"gain_{name}" for name in ALL_SEGMENT_METRIC_NAMES]

    out: List[str] = []
    for metric in metrics:
        m = str(metric)
        if m.startswith("gain_"):
            raw = m[len("gain_"):]
        else:
            raw = m
            m = f"gain_{raw}"
        if raw not in GAIN_DIRECTION_BY_RAW_METRIC:
            raise ValueError(f"Unsupported gain metric request: {metric}")
        out.append(m)
    return out


def _gain_arrays_for_batch(
    gene_batch: List[str],
    values_batch: np.ndarray,
    matched_row: Dict[str, object],
    gain_metrics: List[str],
) -> Dict[str, np.ndarray]:
    raw_metric_names = [m[len("gain_"):] if m.startswith("gain_") else m for m in gain_metrics]
    focus_metrics = _segment_metrics_batch(
        values_batch,
        matched_row["focus_seg"],  # type: ignore[arg-type]
        metric_names=raw_metric_names,
    )
    ref_metrics = _segment_metrics_batch(
        values_batch,
        matched_row["ref_seg"],  # type: ignore[arg-type]
        metric_names=raw_metric_names,
    )

    n_gene = len(gene_batch)
    out: Dict[str, np.ndarray] = {
        "resolution": np.full(n_gene, float(matched_row["resolution"]), dtype=np.float64),
    }
    for gain_metric, raw_name in zip(gain_metrics, raw_metric_names):
        direction = float(GAIN_DIRECTION_BY_RAW_METRIC[raw_name])
        focus_val = np.asarray(focus_metrics[raw_name], dtype=np.float64)
        ref_val = np.asarray(ref_metrics[raw_name], dtype=np.float64)
        out[gain_metric] = direction * (focus_val - ref_val)
    return out


def _long_frame_for_metric_batch(
    gene_batch: List[str],
    values_batch: np.ndarray,
    matched_row: Dict[str, object],
    metric_names: List[str],
) -> pd.DataFrame:
    out = _metric_arrays_for_batch(gene_batch, values_batch, matched_row, metric_names)
    frame = pd.DataFrame(out, index=pd.Index(gene_batch, name="gene"))
    return frame.reset_index()


def _metric_arrays_for_gene_batch(
    adata,
    gene_batch: List[str],
    matched_rows: List[Dict[str, object]],
    metric_names: List[str],
    *,
    layer: Optional[str] = None,
) -> List[Dict[str, np.ndarray]]:
    values_batch = _extract_gene_matrix(adata, gene_batch, layer=layer)
    rows_local: List[Dict[str, np.ndarray]] = []
    for matched_row in matched_rows:
        rows_local.append(_metric_arrays_for_batch(gene_batch, values_batch, matched_row, metric_names))
    return rows_local


def explain_svg_rank_gain(
    adata,
    segments_index_csv: str,
    genes: Iterable[str],
    *,
    focus_compactness: float | Mapping[float, float] | str = 0.1,
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    layer: Optional[str] = None,
    rank_summary_df: Optional[pd.DataFrame] = None,
    compactness_tol: float = 1e-9,
    gene_batch_size: int = 128,
    n_jobs: int = 1,
    parallel_axis: str = "auto",
    return_long_df: bool = True,
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
    - higher size-aware segment autocorr   -> stronger autocorrelation after
                                              reweighting for unequal segment sizes
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

    seg_cache: Dict[str, Dict[str, object]] = {}
    frames: List[pd.DataFrame] = []
    metric_names = [
        "within_segment_var",
        "between_segment_var",
        "segment_snr",
        "segment_reconstruction_r2",
        "neighbor_contrast",
        "segment_moranI",
        "segment_moranI_sizeaware",
        "segment_detection_rate",
        "segment_dynamic_range",
        "segment_top10_mass",
    ]
    numeric_cols = [
        "resolution",
        "focus_compactness",
        "reference_compactness",
    ]
    for metric in metric_names:
        numeric_cols.extend([f"{metric}_SPIX", f"{metric}_GRID"])
    numeric_cols.extend(
        [
            "gain_within_segment_var",
            "gain_between_segment_var",
            "gain_segment_snr",
            "gain_segment_reconstruction_r2",
            "gain_neighbor_contrast",
            "gain_segment_moranI",
            "gain_segment_moranI_sizeaware",
            "gain_segment_detection_rate",
            "gain_segment_dynamic_range",
            "gain_segment_top10_mass",
        ]
    )

    matched_rows: List[Dict[str, object]] = []
    for row in matched.itertuples(index=False):
        focus_path = _resolve_segment_path(getattr(row, "path_focus"), index_path.parent)
        ref_path = _resolve_segment_path(getattr(row, "path_reference"), index_path.parent)

        focus_key = str(focus_path)
        if focus_key not in seg_cache:
            seg_cache[focus_key] = _load_segmentation_cache(focus_path)
        ref_key = str(ref_path)
        if ref_key not in seg_cache:
            seg_cache[ref_key] = _load_segmentation_cache(ref_path)

        matched_rows.append(
            {
                "resolution": float(getattr(row, "resolution")),
                "focus_compactness": float(getattr(row, "compactness_focus")),
                "reference_compactness": float(getattr(row, "compactness_reference")),
                "focus_seg": seg_cache[focus_key],
                "ref_seg": seg_cache[ref_key],
            }
        )

    batch_size = max(1, int(gene_batch_size))
    workers = max(1, int(n_jobs))
    gene_batches = [genes[start:start + batch_size] for start in range(0, len(genes), batch_size)]
    gene_to_pos = {gene: i for i, gene in enumerate(genes)}
    sum_arr = np.zeros((len(genes), len(numeric_cols)), dtype=np.float64)
    count_arr = np.zeros((len(genes), len(numeric_cols)), dtype=np.int32)

    axis = str(parallel_axis).lower().strip()
    if axis not in {"auto", "batches", "matches"}:
        raise ValueError("parallel_axis must be one of {'auto', 'batches', 'matches'}.")
    if axis == "auto":
        axis = "matches" if len(matched_rows) > 1 else "batches"

    def _accumulate_metric_arrays(gene_batch: List[str], arrays: Dict[str, np.ndarray]) -> None:
        pos = np.asarray([gene_to_pos[g] for g in gene_batch], dtype=np.int64)
        for col_idx, col in enumerate(numeric_cols):
            vals = np.asarray(arrays[col], dtype=np.float64)
            finite = np.isfinite(vals)
            if np.any(finite):
                sum_arr[pos[finite], col_idx] += vals[finite]
                count_arr[pos[finite], col_idx] += 1

    def _handle_metric_arrays(gene_batch: List[str], arrays: Dict[str, np.ndarray]) -> None:
        _accumulate_metric_arrays(gene_batch, arrays)
        if bool(return_long_df):
            frames.append(pd.DataFrame(arrays, index=pd.Index(gene_batch, name="gene")).reset_index())

    if workers == 1 or (axis == "matches" and len(matched_rows) <= 1) or (axis == "batches" and len(gene_batches) <= 1):
        for gene_batch in gene_batches:
            values_batch = _extract_gene_matrix(adata, gene_batch, layer=layer)
            for matched_row in matched_rows:
                _handle_metric_arrays(
                    gene_batch,
                    _metric_arrays_for_batch(gene_batch, values_batch, matched_row, metric_names),
                )
    elif axis == "batches":
        with ThreadPoolExecutor(max_workers=min(workers, len(gene_batches))) as ex:
            futures = [
                ex.submit(
                    _metric_arrays_for_gene_batch,
                    adata,
                    gene_batch,
                    matched_rows,
                    metric_names,
                    layer=layer,
                )
                for gene_batch in gene_batches
            ]
            for gene_batch, fut in zip(gene_batches, futures):
                batch_arrays = fut.result()
                for arrays in batch_arrays:
                    _accumulate_metric_arrays(gene_batch, arrays)
                    if bool(return_long_df):
                        frames.append(pd.DataFrame(arrays, index=pd.Index(gene_batch, name="gene")).reset_index())
    else:
        for gene_batch in gene_batches:
            values_batch = _extract_gene_matrix(adata, gene_batch, layer=layer)
            with ThreadPoolExecutor(max_workers=min(workers, len(matched_rows))) as ex:
                futures = [
                    ex.submit(_metric_arrays_for_batch, gene_batch, values_batch, matched_row, metric_names)
                    for matched_row in matched_rows
                ]
                for fut in futures:
                    _handle_metric_arrays(gene_batch, fut.result())

    summary_vals = np.full_like(sum_arr, np.nan, dtype=np.float64)
    np.divide(sum_arr, count_arr, out=summary_vals, where=count_arr > 0)
    summary_df = pd.DataFrame(summary_vals, index=pd.Index(genes, name="gene"), columns=numeric_cols)

    if bool(return_long_df):
        long_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=["gene"] + numeric_cols)
    else:
        long_df = pd.DataFrame(columns=["gene"] + numeric_cols)

    if rank_summary_df is not None:
        summary_df = summary_df.join(rank_summary_df, how="left")

    return summary_df, long_df


def compare_segmentation_metric_summary(
    adata,
    segments_index_csv: str,
    *,
    focus_compactness: float | Mapping[float, float] | str = 0.1,
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    genes: Optional[Iterable[str]] = None,
    layer: Optional[str] = None,
    rank_summary_df: Optional[pd.DataFrame] = None,
    compactness_tol: float = 1e-9,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    gene_batch_size: int = 128,
    n_jobs: int = 1,
    parallel_axis: str = "auto",
    return_long_df: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize SPIX-vs-GRID segmentation metrics across many genes.

    Parameters
    ----------
    adata
        Input AnnData.
    segments_index_csv
        Multiscale segment index CSV with matched SPIX/GRID segment bundles.
    focus_compactness
        Compactness value for the SPIX-like segmentation. If ``"auto"``, choose
        the non-reference candidate at each resolution, preferring rows from
        auto-compactness precomputation when available.
    reference_compactness
        Compactness value for the GRID-like segmentation. This is typically a
        fixed reference such as ``1e7``.
    genes
        Genes to score. If None, uses all genes in ``adata.var_names``.
    layer
        Optional layer to score from.
    rank_summary_df
        Optional gene-level Moran/rank summary table to join into ``summary_df``.
        This can contain columns such as ``contrast_SPIX_vs_GRID``.
    compactness_tol
        Numeric tolerance when matching compactness values from the index CSV.
    quantiles
        Quantiles to include in the aggregate summary table.
    gene_batch_size
        Number of genes to process per batch. Larger values reduce Python
        overhead but use more memory.
    n_jobs
        Number of threads to use across matched resolution pairs inside each
        gene batch. Use modest values such as 2-8.
    parallel_axis
        Parallelization axis. ``"auto"`` chooses gene-batch parallelism when
        there are many batches, otherwise matched-resolution parallelism.
    return_long_df
        If False, skip constructing the per-gene per-resolution ``long_df`` and
        return an empty long table. This is much lighter for whole-transcriptome
        runs when only ``summary_df`` / ``aggregate_df`` are needed.

    Returns
    -------
    summary_df, long_df, aggregate_df
        ``summary_df`` is gene-level mean metrics across matched resolutions,
        ``long_df`` is per-gene per-resolution, and ``aggregate_df`` summarizes
        each gain/raw metric across genes (mean/median/std and requested quantiles).
    """
    if genes is None:
        genes_use = [str(g) for g in adata.var_names]
    else:
        genes_use = [str(g) for g in genes]

    summary_df, long_df = explain_svg_rank_gain(
        adata,
        segments_index_csv,
        genes_use,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        layer=layer,
        rank_summary_df=rank_summary_df,
        compactness_tol=compactness_tol,
        gene_batch_size=gene_batch_size,
        n_jobs=n_jobs,
        parallel_axis=parallel_axis,
        return_long_df=return_long_df,
    )

    if summary_df.empty:
        return summary_df, long_df, pd.DataFrame()

    metric_cols = [
        c for c in summary_df.columns
        if (
            c.startswith("gain_")
            or c.endswith("_SPIX")
            or c.endswith("_GRID")
        )
    ]
    metric_frame = summary_df.loc[:, metric_cols].apply(pd.to_numeric, errors="coerce")
    aggregate_df = pd.DataFrame(
        {
            "mean": metric_frame.mean(axis=0, skipna=True),
            "median": metric_frame.median(axis=0, skipna=True),
            "std": metric_frame.std(axis=0, skipna=True),
            "positive_fraction": metric_frame.gt(0).where(metric_frame.notna()).mean(axis=0, skipna=True),
            "nonzero_fraction": metric_frame.ne(0).where(metric_frame.notna()).mean(axis=0, skipna=True),
        }
    )
    for q in quantiles:
        qv = float(q)
        aggregate_df[f"q{int(round(qv * 100)):02d}"] = metric_frame.quantile(qv, axis=0, interpolation="linear")

    aggregate_df.index.name = "metric"
    aggregate_df = aggregate_df.sort_index()
    return summary_df, long_df, aggregate_df


def compare_segmentation_metric_summary_with_resolution(
    adata,
    segments_index_csv: str,
    *,
    focus_compactness: float | Mapping[float, float] | str = 0.1,
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    genes: Optional[Iterable[str]] = None,
    layer: Optional[str] = None,
    rank_summary_df: Optional[pd.DataFrame] = None,
    compactness_tol: float = 1e-9,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    gene_batch_size: int = 128,
    n_jobs: int = 1,
    parallel_axis: str = "auto",
    metrics: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute gene-level and resolution-level metric summaries without building long_df.

    This is a lighter alternative to ``compare_segmentation_metric_summary(..., return_long_df=True)``
    when you want resolution-wise plots but do not need the full per-gene per-resolution table.

    metrics
        Optional subset of gain metrics to include in ``resolution_df``.
        Gene-level ``summary_df`` and overall ``aggregate_df`` are still computed
        with the full metric set for compatibility.

    Returns
    -------
    summary_df, aggregate_df, resolution_df
        ``summary_df`` is gene-level mean metrics across matched resolutions,
        ``aggregate_df`` summarizes metrics across genes,
        and ``resolution_df`` summarizes metrics across genes at each resolution.
    """
    if genes is None:
        genes_use = [str(g) for g in adata.var_names]
    else:
        genes_use = [str(g) for g in genes]

    index_path, matched = _load_matched_segment_index(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )

    genes_use = [str(g) for g in genes_use if str(g) in adata.var_names]
    if not genes_use:
        raise ValueError("No input genes were found in adata.var_names")

    seg_cache: Dict[str, Dict[str, object]] = {}
    metric_names = [
        "within_segment_var",
        "between_segment_var",
        "segment_snr",
        "segment_reconstruction_r2",
        "neighbor_contrast",
        "segment_moranI",
        "segment_moranI_sizeaware",
        "segment_detection_rate",
        "segment_dynamic_range",
        "segment_top10_mass",
    ]
    numeric_cols = [
        "resolution",
        "focus_compactness",
        "reference_compactness",
    ]
    for metric in metric_names:
        numeric_cols.extend([f"{metric}_SPIX", f"{metric}_GRID"])
    numeric_cols.extend(
        [
            "gain_within_segment_var",
            "gain_between_segment_var",
            "gain_segment_snr",
            "gain_segment_reconstruction_r2",
            "gain_neighbor_contrast",
            "gain_segment_moranI",
            "gain_segment_moranI_sizeaware",
            "gain_segment_detection_rate",
            "gain_segment_dynamic_range",
            "gain_segment_top10_mass",
        ]
    )

    metric_cols = [
        c for c in numeric_cols
        if (
            c.startswith("gain_")
            or c.endswith("_SPIX")
            or c.endswith("_GRID")
        )
    ]

    matched_rows: List[Dict[str, object]] = []
    for row in matched.itertuples(index=False):
        focus_path = _resolve_segment_path(getattr(row, "path_focus"), index_path.parent)
        ref_path = _resolve_segment_path(getattr(row, "path_reference"), index_path.parent)

        focus_key = str(focus_path)
        if focus_key not in seg_cache:
            seg_cache[focus_key] = _load_segmentation_cache(focus_path)
        ref_key = str(ref_path)
        if ref_key not in seg_cache:
            seg_cache[ref_key] = _load_segmentation_cache(ref_path)

        matched_rows.append(
            {
                "resolution": float(getattr(row, "resolution")),
                "focus_compactness": float(getattr(row, "compactness_focus")),
                "reference_compactness": float(getattr(row, "compactness_reference")),
                "focus_seg": seg_cache[focus_key],
                "ref_seg": seg_cache[ref_key],
            }
        )

    gene_to_pos = {gene: i for i, gene in enumerate(genes_use)}
    sum_arr = np.zeros((len(genes_use), len(numeric_cols)), dtype=np.float64)
    count_arr = np.zeros((len(genes_use), len(numeric_cols)), dtype=np.int32)

    resolution_store: Dict[float, Dict[str, List[np.ndarray]]] = {}
    for row in matched_rows:
        resolution_store.setdefault(float(row["resolution"]), {col: [] for col in metric_cols})

    def _accumulate_gene_arrays(gene_batch: List[str], arrays: Dict[str, np.ndarray]) -> None:
        pos = np.asarray([gene_to_pos[g] for g in gene_batch], dtype=np.int64)
        for col_idx, col in enumerate(numeric_cols):
            vals = np.asarray(arrays[col], dtype=np.float64)
            finite = np.isfinite(vals)
            if np.any(finite):
                sum_arr[pos[finite], col_idx] += vals[finite]
                count_arr[pos[finite], col_idx] += 1

    def _accumulate_resolution_arrays(arrays: Dict[str, np.ndarray]) -> None:
        res_val = float(np.asarray(arrays["resolution"], dtype=np.float64)[0])
        bucket = resolution_store[res_val]
        for col in metric_cols:
            bucket[col].append(np.asarray(arrays[col], dtype=np.float64))

    batch_size = max(1, int(gene_batch_size))
    workers = max(1, int(n_jobs))
    gene_batches = [genes_use[start:start + batch_size] for start in range(0, len(genes_use), batch_size)]
    axis = str(parallel_axis).lower().strip()
    if axis not in {"auto", "batches", "matches"}:
        raise ValueError("parallel_axis must be one of {'auto', 'batches', 'matches'}.")
    if axis == "auto":
        axis = "matches" if len(matched_rows) > 1 else "batches"

    if workers == 1 or (axis == "matches" and len(matched_rows) <= 1) or (axis == "batches" and len(gene_batches) <= 1):
        for gene_batch in gene_batches:
            values_batch = _extract_gene_matrix(adata, gene_batch, layer=layer)
            for matched_row in matched_rows:
                arrays = _metric_arrays_for_batch(gene_batch, values_batch, matched_row, metric_names)
                _accumulate_gene_arrays(gene_batch, arrays)
                _accumulate_resolution_arrays(arrays)
    elif axis == "batches":
        with ThreadPoolExecutor(max_workers=min(workers, len(gene_batches))) as ex:
            futures = [
                ex.submit(
                    _metric_arrays_for_gene_batch,
                    adata,
                    gene_batch,
                    matched_rows,
                    metric_names,
                    layer=layer,
                )
                for gene_batch in gene_batches
            ]
            for gene_batch, fut in zip(gene_batches, futures):
                batch_arrays = fut.result()
                for arrays in batch_arrays:
                    _accumulate_gene_arrays(gene_batch, arrays)
                    _accumulate_resolution_arrays(arrays)
    else:
        for gene_batch in gene_batches:
            values_batch = _extract_gene_matrix(adata, gene_batch, layer=layer)
            with ThreadPoolExecutor(max_workers=min(workers, len(matched_rows))) as ex:
                futures = [
                    ex.submit(_metric_arrays_for_batch, gene_batch, values_batch, matched_row, metric_names)
                    for matched_row in matched_rows
                ]
                for fut in futures:
                    arrays = fut.result()
                    _accumulate_gene_arrays(gene_batch, arrays)
                    _accumulate_resolution_arrays(arrays)

    summary_vals = np.full_like(sum_arr, np.nan, dtype=np.float64)
    np.divide(sum_arr, count_arr, out=summary_vals, where=count_arr > 0)
    summary_df = pd.DataFrame(summary_vals, index=pd.Index(genes_use, name="gene"), columns=numeric_cols)
    if rank_summary_df is not None:
        summary_df = summary_df.join(rank_summary_df, how="left")

    metric_frame = summary_df.loc[:, metric_cols].apply(pd.to_numeric, errors="coerce")
    aggregate_df = pd.DataFrame(
        {
            "mean": metric_frame.mean(axis=0, skipna=True),
            "median": metric_frame.median(axis=0, skipna=True),
            "std": metric_frame.std(axis=0, skipna=True),
            "positive_fraction": metric_frame.gt(0).where(metric_frame.notna()).mean(axis=0, skipna=True),
            "nonzero_fraction": metric_frame.ne(0).where(metric_frame.notna()).mean(axis=0, skipna=True),
        }
    )
    for q in quantiles:
        qv = float(q)
        aggregate_df[f"q{int(round(qv * 100)):02d}"] = metric_frame.quantile(qv, axis=0, interpolation="linear")
    aggregate_df.index.name = "metric"
    aggregate_df = aggregate_df.sort_index()

    resolution_rows: List[Dict[str, object]] = []
    for res in sorted(resolution_store):
        bucket = resolution_store[res]
        for metric in metric_cols:
            if len(bucket[metric]) == 0:
                vals = np.zeros(0, dtype=np.float64)
            else:
                vals = np.concatenate(bucket[metric], axis=0).astype(np.float64, copy=False)
            vals = vals[np.isfinite(vals)]
            row: Dict[str, object] = {
                "resolution": float(res),
                "metric": metric,
                "n_genes": int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "median": float(np.median(vals)) if vals.size else np.nan,
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "positive_fraction": float(np.mean(vals > 0)) if vals.size else np.nan,
                "nonzero_fraction": float(np.mean(vals != 0)) if vals.size else np.nan,
            }
            for q in quantiles:
                qv = float(q)
                row[f"q{int(round(qv * 100)):02d}"] = float(np.quantile(vals, qv)) if vals.size else np.nan
            resolution_rows.append(row)

    resolution_df = pd.DataFrame(resolution_rows)
    if not resolution_df.empty:
        resolution_df = resolution_df.sort_values(["metric", "resolution"]).reset_index(drop=True)

    return summary_df, aggregate_df, resolution_df


def compare_segmentation_resolution_summary(
    adata,
    segments_index_csv: str,
    *,
    focus_compactness: float | Mapping[float, float] | str = 0.1,
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    genes: Optional[Iterable[str]] = None,
    layer: Optional[str] = None,
    compactness_tol: float = 1e-9,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    gene_batch_size: int = 128,
    n_jobs: int = 1,
    parallel_axis: str = "auto",
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Fast path for resolution-wise plots without building gene-level summaries.

    This function computes only the requested gain metrics and returns only a
    resolution summary table, which is substantially faster than building
    ``summary_df`` or ``long_df`` for whole-transcriptome runs.
    """
    gain_metrics = _normalize_gain_metric_request(metrics)

    if genes is None:
        genes_use = [str(g) for g in adata.var_names]
    else:
        genes_use = [str(g) for g in genes]
    genes_use = [str(g) for g in genes_use if str(g) in adata.var_names]
    if not genes_use:
        raise ValueError("No input genes were found in adata.var_names")

    index_path, matched = _load_matched_segment_index(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )

    seg_cache: Dict[str, Dict[str, object]] = {}
    matched_rows: List[Dict[str, object]] = []
    for row in matched.itertuples(index=False):
        focus_path = _resolve_segment_path(getattr(row, "path_focus"), index_path.parent)
        ref_path = _resolve_segment_path(getattr(row, "path_reference"), index_path.parent)

        focus_key = str(focus_path)
        if focus_key not in seg_cache:
            seg_cache[focus_key] = _load_segmentation_cache(focus_path)
        ref_key = str(ref_path)
        if ref_key not in seg_cache:
            seg_cache[ref_key] = _load_segmentation_cache(ref_path)

        matched_rows.append(
            {
                "resolution": float(getattr(row, "resolution")),
                "focus_compactness": float(getattr(row, "compactness_focus")),
                "reference_compactness": float(getattr(row, "compactness_reference")),
                "focus_seg": seg_cache[focus_key],
                "ref_seg": seg_cache[ref_key],
            }
        )

    resolution_store: Dict[float, Dict[str, List[np.ndarray]]] = {}
    for row in matched_rows:
        resolution_store.setdefault(float(row["resolution"]), {metric: [] for metric in gain_metrics})

    batch_size = max(1, int(gene_batch_size))
    workers = max(1, int(n_jobs))
    gene_batches = [genes_use[start:start + batch_size] for start in range(0, len(genes_use), batch_size)]
    axis = str(parallel_axis).lower().strip()
    if axis not in {"auto", "batches", "matches"}:
        raise ValueError("parallel_axis must be one of {'auto', 'batches', 'matches'}.")
    if axis == "auto":
        axis = "matches" if len(matched_rows) > 1 else "batches"

    def _accumulate_resolution_arrays(arrays: Dict[str, np.ndarray]) -> None:
        res_val = float(np.asarray(arrays["resolution"], dtype=np.float64)[0])
        bucket = resolution_store[res_val]
        for metric in gain_metrics:
            bucket[metric].append(np.asarray(arrays[metric], dtype=np.float64))

    if workers == 1 or (axis == "matches" and len(matched_rows) <= 1) or (axis == "batches" and len(gene_batches) <= 1):
        for gene_batch in gene_batches:
            values_batch = _extract_gene_matrix(adata, gene_batch, layer=layer)
            for matched_row in matched_rows:
                _accumulate_resolution_arrays(
                    _gain_arrays_for_batch(gene_batch, values_batch, matched_row, gain_metrics)
                )
    elif axis == "batches":
        with ThreadPoolExecutor(max_workers=min(workers, len(gene_batches))) as ex:
            futures = [
                ex.submit(_extract_gene_matrix, adata, gene_batch, layer=layer)
                for gene_batch in gene_batches
            ]
            for gene_batch, fut in zip(gene_batches, futures):
                values_batch = fut.result()
                for matched_row in matched_rows:
                    _accumulate_resolution_arrays(
                        _gain_arrays_for_batch(gene_batch, values_batch, matched_row, gain_metrics)
                    )
    else:
        for gene_batch in gene_batches:
            values_batch = _extract_gene_matrix(adata, gene_batch, layer=layer)
            with ThreadPoolExecutor(max_workers=min(workers, len(matched_rows))) as ex:
                futures = [
                    ex.submit(_gain_arrays_for_batch, gene_batch, values_batch, matched_row, gain_metrics)
                    for matched_row in matched_rows
                ]
                for fut in futures:
                    _accumulate_resolution_arrays(fut.result())

    resolution_rows: List[Dict[str, object]] = []
    for res in sorted(resolution_store):
        bucket = resolution_store[res]
        for metric in gain_metrics:
            vals = np.concatenate(bucket[metric], axis=0).astype(np.float64, copy=False) if len(bucket[metric]) else np.zeros(0, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row: Dict[str, object] = {
                "resolution": float(res),
                "metric": metric,
                "n_genes": int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "median": float(np.median(vals)) if vals.size else np.nan,
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "positive_fraction": float(np.mean(vals > 0)) if vals.size else np.nan,
                "nonzero_fraction": float(np.mean(vals != 0)) if vals.size else np.nan,
            }
            for q in quantiles:
                qv = float(q)
                row[f"q{int(round(qv * 100)):02d}"] = float(np.quantile(vals, qv)) if vals.size else np.nan
            resolution_rows.append(row)

    resolution_df = pd.DataFrame(resolution_rows)
    if not resolution_df.empty:
        resolution_df = resolution_df.sort_values(["metric", "resolution"]).reset_index(drop=True)
    return resolution_df


def _prepare_matched_rows(
    segments_index_csv: str,
    *,
    focus_compactness: float | Mapping[float, float] | str,
    reference_compactness: float | Mapping[float, float] | str,
    compactness_tol: float,
) -> tuple[Path, List[Dict[str, object]]]:
    index_path, matched = _load_matched_segment_index(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )

    seg_cache: Dict[str, Dict[str, object]] = {}
    matched_rows: List[Dict[str, object]] = []
    for row in matched.itertuples(index=False):
        focus_path = _resolve_segment_path(getattr(row, "path_focus"), index_path.parent)
        ref_path = _resolve_segment_path(getattr(row, "path_reference"), index_path.parent)

        focus_key = str(focus_path)
        if focus_key not in seg_cache:
            seg_cache[focus_key] = _load_segmentation_cache(focus_path)
        ref_key = str(ref_path)
        if ref_key not in seg_cache:
            seg_cache[ref_key] = _load_segmentation_cache(ref_path)

        matched_rows.append(
            {
                "resolution": float(getattr(row, "resolution")),
                "focus_compactness": float(getattr(row, "compactness_focus")),
                "reference_compactness": float(getattr(row, "compactness_reference")),
                "focus_seg": seg_cache[focus_key],
                "ref_seg": seg_cache[ref_key],
            }
        )
    return index_path, matched_rows


def _second_largest_per_row(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if arr.shape[1] <= 1:
        return np.zeros(arr.shape[0], dtype=np.float64)
    part = np.partition(arr, kth=arr.shape[1] - 2, axis=1)
    return np.asarray(part[:, -2], dtype=np.float64)


def _weighted_average_or_nan(values: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    keep = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not np.any(keep):
        return np.nan
    vals = vals[keep]
    w = w[keep]
    return float(np.sum(vals * w) / max(float(np.sum(w)), eps))


def _resolve_marker_groups(
    adata,
    marker_groups: Mapping[str, Iterable[str]],
) -> tuple[List[str], Dict[str, np.ndarray]]:
    if marker_groups is None or len(marker_groups) == 0:
        raise ValueError("marker_groups must be a non-empty mapping of class -> genes.")

    genes_order: List[str] = []
    for genes in marker_groups.values():
        for gene in genes:
            g = str(gene)
            if g in adata.var_names and g not in genes_order:
                genes_order.append(g)
    if not genes_order:
        raise ValueError("None of the provided marker genes were found in adata.var_names.")

    gene_to_idx = {g: i for i, g in enumerate(genes_order)}
    group_indexer: Dict[str, np.ndarray] = {}
    for group_name, genes in marker_groups.items():
        idx = [gene_to_idx[str(g)] for g in genes if str(g) in gene_to_idx]
        if len(idx) == 0:
            continue
        group_indexer[str(group_name)] = np.asarray(idx, dtype=np.int64)
    if not group_indexer:
        raise ValueError("No marker group retained at least one valid gene.")
    return genes_order, group_indexer


def _resolve_obs_label_codes(
    adata,
    *,
    labels: Optional[Iterable[object]] = None,
    obs_key: Optional[str] = None,
) -> tuple[np.ndarray, List[str]]:
    if labels is not None and obs_key is not None:
        raise ValueError("Provide either `labels` or `obs_key`, not both.")
    if obs_key is not None:
        if obs_key not in adata.obs.columns:
            raise KeyError(f"obs_key not found in adata.obs: {obs_key}")
        raw = pd.Series(adata.obs[obs_key], index=adata.obs_names, copy=False)
    elif labels is not None:
        raw = pd.Series(list(labels))
        if len(raw) != int(adata.n_obs):
            raise ValueError("labels must have length equal to adata.n_obs.")
    else:
        raise ValueError("Either `labels` or `obs_key` must be provided.")

    raw = raw.astype("object")
    raw = raw.where(~pd.isna(raw), other=None)
    non_missing = raw[raw.notna()]
    if non_missing.empty:
        raise ValueError("No non-missing labels were provided.")

    codes, uniques = pd.factorize(raw, sort=True)
    codes = np.asarray(codes, dtype=np.int32)
    codes[pd.isna(raw).to_numpy()] = -1
    names = [str(v) for v in uniques.tolist()]
    return codes, names


def get_crc_visiumhd_marker_panels(
    panel_names: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Return CRC/VisiumHD-oriented marker panels for GT-free segmentation benchmarks.

    The presets are intended for biological-coherence benchmarking on colorectal
    cancer Visium HD datasets, with broad compartment, desmoplastic niche, and
    immune/TLS-focused panels.
    """
    if panel_names is None:
        names = list(CRC_VISIUMHD_MARKER_PANELS.keys())
    else:
        names = [str(name) for name in panel_names]
    missing = [name for name in names if name not in CRC_VISIUMHD_MARKER_PANELS]
    if missing:
        raise KeyError(f"Unknown CRC marker panels: {missing}")
    return {
        name: {
            group: [str(g) for g in genes]
            for group, genes in CRC_VISIUMHD_MARKER_PANELS[name].items()
        }
        for name in names
    }


def _segment_marker_metrics(
    marker_values,
    seg_info: Dict[str, object],
    *,
    group_indexer: Mapping[str, np.ndarray],
    weight_by: str = "marker_mass",
    eps: float = 1e-12,
) -> Dict[str, float]:
    obs_idx = np.asarray(seg_info["obs_idx"], dtype=np.int64)
    obs_to_seg = seg_info["obs_to_seg"]
    src = np.asarray(seg_info["src"], dtype=np.int32)
    dst = np.asarray(seg_info["dst"], dtype=np.int32)
    edge_weight = np.asarray(seg_info["weight"], dtype=np.float64)
    seg_counts = np.asarray(seg_info["seg_counts"], dtype=np.float64)
    n_segments = int(seg_info["n_segments"])
    n_groups = int(len(group_indexer))

    out = {
        "marker_purity": np.nan,
        "marker_contamination": np.nan,
        "marker_margin": np.nan,
        "marker_entropy_purity": np.nan,
        "marker_neighbor_purity": np.nan,
        "marker_coverage": np.nan,
    }
    if obs_idx.size == 0 or n_segments <= 0 or n_groups <= 0:
        return out

    x = marker_values[obs_idx, :]
    prod = obs_to_seg.T @ x
    seg_gene_sum = prod.toarray().astype(np.float64, copy=False) if issparse(prod) else np.asarray(prod, dtype=np.float64)
    if seg_gene_sum.ndim == 1:
        seg_gene_sum = seg_gene_sum.reshape(-1, 1)

    group_names = list(group_indexer.keys())
    group_sum = np.zeros((n_segments, len(group_names)), dtype=np.float64)
    for j, group_name in enumerate(group_names):
        idx = np.asarray(group_indexer[group_name], dtype=np.int64)
        group_sum[:, j] = np.asarray(seg_gene_sum[:, idx].sum(axis=1), dtype=np.float64).reshape(-1)

    total = group_sum.sum(axis=1)
    valid = total > eps
    if not np.any(valid):
        return out

    probs = np.zeros_like(group_sum)
    probs[valid] = group_sum[valid] / total[valid, None]

    top1 = np.max(probs[valid], axis=1)
    top2 = _second_largest_per_row(probs[valid])
    dominant = np.argmax(group_sum, axis=1).astype(np.int32, copy=False)
    entropy = -(probs[valid] * np.log(np.clip(probs[valid], eps, None))).sum(axis=1)
    entropy_norm = entropy / np.log(max(2, len(group_names)))
    if str(weight_by).lower().strip() == "segment_size":
        weights_seg = seg_counts[valid]
    else:
        weights_seg = total[valid]

    out["marker_purity"] = _weighted_average_or_nan(top1, weights_seg, eps=eps)
    out["marker_contamination"] = _weighted_average_or_nan(1.0 - top1, weights_seg, eps=eps)
    out["marker_margin"] = _weighted_average_or_nan(top1 - top2, weights_seg, eps=eps)
    out["marker_entropy_purity"] = _weighted_average_or_nan(1.0 - entropy_norm, weights_seg, eps=eps)
    out["marker_coverage"] = float(np.sum(total[valid]) / max(float(np.sum(total)), eps))

    in_range = (src >= 0) & (dst >= 0) & (src < n_segments) & (dst < n_segments)
    if np.any(in_range):
        src_in = src[in_range]
        dst_in = dst[in_range]
        edge_mask = valid[src_in] & valid[dst_in]
        if np.any(edge_mask):
            src_v = src_in[edge_mask]
            dst_v = dst_in[edge_mask]
            w_v = edge_weight[in_range][edge_mask]
            same = (dominant[src_v] == dominant[dst_v]).astype(np.float64, copy=False)
            out["marker_neighbor_purity"] = _weighted_average_or_nan(same, w_v, eps=eps)

    return out


def _segment_label_metrics(
    label_codes: np.ndarray,
    seg_info: Dict[str, object],
    *,
    n_labels: Optional[int] = None,
    weight_by: str = "segment_size",
    eps: float = 1e-12,
) -> Dict[str, float]:
    obs_idx = np.asarray(seg_info["obs_idx"], dtype=np.int64)
    obs_to_seg = seg_info["obs_to_seg"]
    src = np.asarray(seg_info["src"], dtype=np.int32)
    dst = np.asarray(seg_info["dst"], dtype=np.int32)
    edge_weight = np.asarray(seg_info["weight"], dtype=np.float64)
    seg_counts = np.asarray(seg_info["seg_counts"], dtype=np.float64)
    n_segments = int(seg_info["n_segments"])

    out = {
        "label_purity": np.nan,
        "label_contamination": np.nan,
        "label_margin": np.nan,
        "label_entropy_purity": np.nan,
        "label_neighbor_purity": np.nan,
        "label_coverage": np.nan,
    }
    if obs_idx.size == 0 or n_segments <= 0:
        return out

    obs_labels = np.asarray(label_codes, dtype=np.int32)[obs_idx]
    keep = obs_labels >= 0
    if not np.any(keep):
        return out

    if n_labels is None:
        n_labels = int(obs_labels[keep].max(initial=-1)) + 1
    if n_labels <= 0:
        return out

    rows = np.arange(int(np.sum(keep)), dtype=np.int64)
    label_onehot = csr_matrix(
        (
            np.ones(rows.size, dtype=np.float64),
            (rows, obs_labels[keep].astype(np.int64, copy=False)),
        ),
        shape=(rows.size, int(n_labels)),
    )
    seg_label = obs_to_seg[keep, :].T @ label_onehot
    label_counts = seg_label.toarray().astype(np.float64, copy=False) if issparse(seg_label) else np.asarray(seg_label, dtype=np.float64)
    if label_counts.ndim == 1:
        label_counts = label_counts.reshape(-1, 1)

    labeled = label_counts.sum(axis=1)
    valid = labeled > 0
    if not np.any(valid):
        return out

    probs = np.zeros_like(label_counts)
    probs[valid] = label_counts[valid] / labeled[valid, None]

    top1 = np.max(probs[valid], axis=1)
    top2 = _second_largest_per_row(probs[valid])
    dominant = np.argmax(label_counts, axis=1).astype(np.int32, copy=False)
    entropy = -(probs[valid] * np.log(np.clip(probs[valid], eps, None))).sum(axis=1)
    entropy_norm = entropy / np.log(max(2, label_counts.shape[1]))
    if str(weight_by).lower().strip() == "labeled_mass":
        weights_seg = labeled[valid]
    else:
        weights_seg = seg_counts[valid]

    out["label_purity"] = _weighted_average_or_nan(top1, weights_seg, eps=eps)
    out["label_contamination"] = _weighted_average_or_nan(1.0 - top1, weights_seg, eps=eps)
    out["label_margin"] = _weighted_average_or_nan(top1 - top2, weights_seg, eps=eps)
    out["label_entropy_purity"] = _weighted_average_or_nan(1.0 - entropy_norm, weights_seg, eps=eps)
    out["label_coverage"] = float(np.sum(labeled) / max(float(np.sum(seg_counts)), eps))

    in_range = (src >= 0) & (dst >= 0) & (src < n_segments) & (dst < n_segments)
    if np.any(in_range):
        src_in = src[in_range]
        dst_in = dst[in_range]
        edge_mask = valid[src_in] & valid[dst_in]
        if np.any(edge_mask):
            src_v = src_in[edge_mask]
            dst_v = dst_in[edge_mask]
            w_v = edge_weight[in_range][edge_mask]
            same = (dominant[src_v] == dominant[dst_v]).astype(np.float64, copy=False)
            out["label_neighbor_purity"] = _weighted_average_or_nan(same, w_v, eps=eps)

    return out


def _comparison_tables_from_rows(
    rows: List[Dict[str, object]],
    *,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return comparison_df, pd.DataFrame(), pd.DataFrame()

    metric_cols = [
        c for c in comparison_df.columns
        if c.endswith("_SPIX") or c.endswith("_GRID") or c.startswith("gain_")
    ]
    metric_frame = comparison_df.loc[:, metric_cols].apply(pd.to_numeric, errors="coerce")
    aggregate_df = pd.DataFrame(
        {
            "mean": metric_frame.mean(axis=0, skipna=True),
            "median": metric_frame.median(axis=0, skipna=True),
            "std": metric_frame.std(axis=0, skipna=True),
            "positive_fraction": metric_frame.gt(0).where(metric_frame.notna()).mean(axis=0, skipna=True),
            "nonzero_fraction": metric_frame.ne(0).where(metric_frame.notna()).mean(axis=0, skipna=True),
        }
    )
    for q in quantiles:
        qv = float(q)
        aggregate_df[f"q{int(round(qv * 100)):02d}"] = metric_frame.quantile(qv, axis=0, interpolation="linear")
    aggregate_df.index.name = "metric"
    aggregate_df = aggregate_df.sort_index()

    resolution_rows: List[Dict[str, object]] = []
    for row in comparison_df.itertuples(index=False):
        res = float(getattr(row, "resolution"))
        for metric in metric_cols:
            val = float(getattr(row, metric))
            rec: Dict[str, object] = {
                "resolution": res,
                "metric": metric,
                "n_values": 1,
                "mean": val,
                "median": val,
                "std": np.nan,
                "positive_fraction": float(val > 0) if np.isfinite(val) else np.nan,
                "nonzero_fraction": float(val != 0) if np.isfinite(val) else np.nan,
            }
            for q in quantiles:
                rec[f"q{int(round(float(q) * 100)):02d}"] = val
            resolution_rows.append(rec)

    resolution_df = pd.DataFrame(resolution_rows)
    if not resolution_df.empty:
        resolution_df = resolution_df.sort_values(["metric", "resolution"]).reset_index(drop=True)
    return comparison_df, aggregate_df, resolution_df


def _compare_segmentation_marker_purity_from_matrix(
    marker_values,
    *,
    group_indexer: Mapping[str, np.ndarray],
    matched_rows: List[Dict[str, object]],
    weight_by: str = "marker_mass",
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    meta: Optional[Mapping[str, object]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    meta_dict = dict(meta or {})
    n_marker_groups = int(len(group_indexer))
    n_marker_genes = int(marker_values.shape[1]) if hasattr(marker_values, "shape") else np.nan

    for matched_row in matched_rows:
        focus_metrics = _segment_marker_metrics(
            marker_values,
            matched_row["focus_seg"],  # type: ignore[arg-type]
            group_indexer=group_indexer,
            weight_by=weight_by,
        )
        ref_metrics = _segment_marker_metrics(
            marker_values,
            matched_row["ref_seg"],  # type: ignore[arg-type]
            group_indexer=group_indexer,
            weight_by=weight_by,
        )

        row: Dict[str, object] = {
            "resolution": float(matched_row["resolution"]),
            "focus_compactness": float(matched_row["focus_compactness"]),
            "reference_compactness": float(matched_row["reference_compactness"]),
            "n_marker_groups": n_marker_groups,
            "n_marker_genes": n_marker_genes,
        }
        row.update(meta_dict)
        for metric in [
            "marker_purity",
            "marker_contamination",
            "marker_margin",
            "marker_entropy_purity",
            "marker_neighbor_purity",
            "marker_coverage",
        ]:
            row[f"{metric}_SPIX"] = float(focus_metrics[metric])
            row[f"{metric}_GRID"] = float(ref_metrics[metric])
        row["gain_marker_purity"] = row["marker_purity_SPIX"] - row["marker_purity_GRID"]
        row["gain_marker_contamination"] = row["marker_contamination_GRID"] - row["marker_contamination_SPIX"]
        row["gain_marker_margin"] = row["marker_margin_SPIX"] - row["marker_margin_GRID"]
        row["gain_marker_entropy_purity"] = row["marker_entropy_purity_SPIX"] - row["marker_entropy_purity_GRID"]
        row["gain_marker_neighbor_purity"] = row["marker_neighbor_purity_SPIX"] - row["marker_neighbor_purity_GRID"]
        row["gain_marker_coverage"] = row["marker_coverage_SPIX"] - row["marker_coverage_GRID"]
        rows.append(row)

    return _comparison_tables_from_rows(rows, quantiles=quantiles)


def compare_segmentation_marker_purity_summary(
    adata,
    segments_index_csv: str,
    *,
    marker_groups: Mapping[str, Iterable[str]],
    focus_compactness: float | Mapping[float, float] | str = "auto",
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    layer: Optional[str] = None,
    compactness_tol: float = 1e-9,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    weight_by: str = "marker_mass",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare SPIX vs GRID using marker-based purity/contamination metrics.

    This is designed for GT-free datasets such as VisiumHD CRC where curated
    marker programs are available but segmentation ground truth is not.

    Parameters
    ----------
    marker_groups
        Mapping such as ``{"tumor": [...], "immune": [...], "stroma": [...]}``.
        Each group's genes are aggregated within segments, then segment-level
        purity/contamination metrics are computed from the resulting class mix.
    weight_by
        ``"marker_mass"`` (default) weights segments by total marker signal.
        ``"segment_size"`` weights by number of observations in the segment.

    Returns
    -------
    comparison_df, aggregate_df, resolution_df
        ``comparison_df`` has one row per matched resolution pair with raw
        SPIX/GRID values and gain columns. The other two tables mirror the
        summary format used elsewhere in this module.
    """
    marker_genes, group_indexer = _resolve_marker_groups(adata, marker_groups)
    marker_values = _extract_gene_matrix(adata, marker_genes, layer=layer)
    _, matched_rows = _prepare_matched_rows(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )
    comparison_df, aggregate_df, resolution_df = _compare_segmentation_marker_purity_from_matrix(
        marker_values,
        group_indexer=group_indexer,
        matched_rows=matched_rows,
        weight_by=weight_by,
        quantiles=quantiles,
    )
    comparison_df.attrs["marker_groups"] = list(group_indexer.keys())
    comparison_df.attrs["weight_by"] = str(weight_by)
    return comparison_df, aggregate_df, resolution_df


def compare_segmentation_label_purity_summary(
    adata,
    segments_index_csv: str,
    *,
    labels: Optional[Iterable[object]] = None,
    obs_key: Optional[str] = None,
    focus_compactness: float | Mapping[float, float] | str = "auto",
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    compactness_tol: float = 1e-9,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    weight_by: str = "segment_size",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare SPIX vs GRID using categorical label purity/neighborhood metrics.

    Parameters
    ----------
    labels, obs_key
        Observation-level categorical annotation, such as a CRC region label,
        cell-state cluster, or manual compartment assignment.
    weight_by
        ``"segment_size"`` (default) weights by all observations per segment.
        ``"labeled_mass"`` weights only by observations with non-missing labels.
    """
    label_codes, label_names = _resolve_obs_label_codes(adata, labels=labels, obs_key=obs_key)
    _, matched_rows = _prepare_matched_rows(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )

    rows: List[Dict[str, object]] = []
    for matched_row in matched_rows:
        focus_metrics = _segment_label_metrics(
            label_codes,
            matched_row["focus_seg"],  # type: ignore[arg-type]
            n_labels=len(label_names),
            weight_by=weight_by,
        )
        ref_metrics = _segment_label_metrics(
            label_codes,
            matched_row["ref_seg"],  # type: ignore[arg-type]
            n_labels=len(label_names),
            weight_by=weight_by,
        )

        row: Dict[str, object] = {
            "resolution": float(matched_row["resolution"]),
            "focus_compactness": float(matched_row["focus_compactness"]),
            "reference_compactness": float(matched_row["reference_compactness"]),
            "n_labels": int(len(label_names)),
        }
        for metric in [
            "label_purity",
            "label_contamination",
            "label_margin",
            "label_entropy_purity",
            "label_neighbor_purity",
            "label_coverage",
        ]:
            row[f"{metric}_SPIX"] = float(focus_metrics[metric])
            row[f"{metric}_GRID"] = float(ref_metrics[metric])
        row["gain_label_purity"] = row["label_purity_SPIX"] - row["label_purity_GRID"]
        row["gain_label_contamination"] = row["label_contamination_GRID"] - row["label_contamination_SPIX"]
        row["gain_label_margin"] = row["label_margin_SPIX"] - row["label_margin_GRID"]
        row["gain_label_entropy_purity"] = row["label_entropy_purity_SPIX"] - row["label_entropy_purity_GRID"]
        row["gain_label_neighbor_purity"] = row["label_neighbor_purity_SPIX"] - row["label_neighbor_purity_GRID"]
        row["gain_label_coverage"] = row["label_coverage_SPIX"] - row["label_coverage_GRID"]
        rows.append(row)

    comparison_df, aggregate_df, resolution_df = _comparison_tables_from_rows(rows, quantiles=quantiles)
    comparison_df.attrs["label_names"] = label_names
    comparison_df.attrs["weight_by"] = str(weight_by)
    return comparison_df, aggregate_df, resolution_df


def compare_segmentation_marker_panel_benchmark(
    adata,
    segments_index_csv: str,
    *,
    marker_panels: Optional[Mapping[str, Mapping[str, Iterable[str]]]] = None,
    panel_names: Optional[Iterable[str]] = None,
    focus_compactness: float | Mapping[float, float] | str = "auto",
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    layer: Optional[str] = None,
    compactness_tol: float = 1e-9,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    weight_by: str = "marker_mass",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run marker-based SPIX-vs-GRID benchmark across multiple CRC panel presets."""
    if marker_panels is None:
        marker_panels_use = get_crc_visiumhd_marker_panels(panel_names=panel_names)
    else:
        marker_panels_use = {
            str(panel): {
                str(group): [str(g) for g in genes]
                for group, genes in groups.items()
            }
            for panel, groups in marker_panels.items()
        }
        if panel_names is not None:
            keep = [str(name) for name in panel_names]
            marker_panels_use = {name: marker_panels_use[name] for name in keep if name in marker_panels_use}
    if not marker_panels_use:
        raise ValueError("No marker panels available for benchmarking.")

    panel_frames: List[pd.DataFrame] = []
    aggregate_frames: List[pd.DataFrame] = []
    resolution_frames: List[pd.DataFrame] = []
    for panel_name, groups in marker_panels_use.items():
        comparison_df, aggregate_df, resolution_df = compare_segmentation_marker_purity_summary(
            adata,
            segments_index_csv,
            marker_groups=groups,
            focus_compactness=focus_compactness,
            reference_compactness=reference_compactness,
            layer=layer,
            compactness_tol=compactness_tol,
            quantiles=quantiles,
            weight_by=weight_by,
        )
        panel_df = comparison_df.copy()
        panel_df.insert(0, "panel", str(panel_name))
        panel_df["panel_n_groups"] = int(len(groups))
        panel_df["panel_n_genes"] = int(len({gene for genes in groups.values() for gene in genes}))
        panel_frames.append(panel_df)

        agg_df = aggregate_df.reset_index().copy()
        agg_df.insert(0, "panel", str(panel_name))
        aggregate_frames.append(agg_df)

        res_df = resolution_df.copy()
        if not res_df.empty:
            res_df.insert(0, "panel", str(panel_name))
        resolution_frames.append(res_df)

    panel_df = pd.concat(panel_frames, axis=0, ignore_index=True) if panel_frames else pd.DataFrame()
    panel_aggregate_df = pd.concat(aggregate_frames, axis=0, ignore_index=True) if aggregate_frames else pd.DataFrame()
    panel_resolution_df = pd.concat(resolution_frames, axis=0, ignore_index=True) if resolution_frames else pd.DataFrame()
    if not panel_resolution_df.empty:
        panel_resolution_df = panel_resolution_df.sort_values(["panel", "metric", "resolution"]).reset_index(drop=True)
    return panel_df, panel_aggregate_df, panel_resolution_df


def add_relative_gain_columns(
    comparison_df: pd.DataFrame,
    *,
    metric_bases: Optional[Iterable[str]] = None,
    scale: float = 100.0,
    eps: float = 1e-8,
    suffix: str = "_pct",
) -> pd.DataFrame:
    """Add relative SPIX-vs-GRID gain columns such as percent improvement over GRID."""
    if comparison_df is None or len(comparison_df) == 0:
        raise ValueError("comparison_df must be a non-empty DataFrame.")

    if metric_bases is None:
        inferred = []
        for col in comparison_df.columns:
            if str(col).startswith("gain_"):
                base = str(col)[len("gain_"):]
                if f"{base}_GRID" in comparison_df.columns:
                    inferred.append(base)
        metric_bases_use = list(dict.fromkeys(inferred))
    else:
        metric_bases_use = [str(m) for m in metric_bases]

    out = comparison_df.copy()
    for base in metric_bases_use:
        gain_col = f"gain_{base}"
        ref_col = f"{base}_GRID"
        if gain_col not in out.columns or ref_col not in out.columns:
            continue
        gain_vals = pd.to_numeric(out[gain_col], errors="coerce")
        ref_vals = pd.to_numeric(out[ref_col], errors="coerce")
        denom = ref_vals.abs().clip(lower=float(eps))
        out[f"rel_gain_{base}{suffix}"] = (gain_vals / denom) * float(scale)
    return out


def _compute_gene_statistics(
    adata,
    *,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    mat = adata.layers[layer] if layer is not None else adata.X
    n_obs = max(1, int(adata.n_obs))
    if issparse(mat):
        mat_csr = mat.tocsr(copy=False)
        mean_expr = np.asarray(mat_csr.mean(axis=0)).reshape(-1).astype(np.float64, copy=False)
        detection_rate = np.asarray(mat_csr.getnnz(axis=0)).reshape(-1).astype(np.float64, copy=False) / float(n_obs)
    else:
        arr = np.asarray(mat, dtype=np.float64)
        mean_expr = np.nanmean(arr, axis=0)
        detection_rate = np.mean(np.isfinite(arr) & (arr > 0), axis=0)
    out = pd.DataFrame(
        {
            "gene": [str(g) for g in adata.var_names.tolist()],
            "mean_expr": mean_expr,
            "detection_rate": detection_rate,
        }
    ).set_index("gene")
    return out


def _assign_stat_bins(
    series: pd.Series,
    n_bins: int,
) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    ranks = vals.rank(method="average", pct=True)
    bins = np.floor(ranks * max(1, int(n_bins))).astype("Int64")
    bins = bins.clip(lower=0, upper=max(0, int(n_bins) - 1))
    bins[vals.isna()] = pd.NA
    return bins.astype("Int64")


def _sample_expression_matched_marker_groups(
    marker_groups: Mapping[str, Iterable[str]],
    gene_stats: pd.DataFrame,
    *,
    rng: np.random.Generator,
    excluded_genes: Optional[set[str]] = None,
    used_genes: Optional[set[str]] = None,
    mean_bin_col: str = "mean_bin",
    detect_bin_col: str = "detect_bin",
) -> Dict[str, List[str]]:
    def _candidate_pool(
        cand_stats: pd.DataFrame,
        mean_bin,
        detect_bin,
    ) -> List[str]:
        if pd.notna(mean_bin) and pd.notna(detect_bin):
            mask = (
                (cand_stats[mean_bin_col].astype("Int64") == int(mean_bin))
                & (cand_stats[detect_bin_col].astype("Int64") == int(detect_bin))
            )
            pool_local = cand_stats.index[mask].tolist()
            if pool_local:
                return pool_local
        if pd.notna(mean_bin):
            pool_local = cand_stats.index[cand_stats[mean_bin_col].astype("Int64") == int(mean_bin)].tolist()
            if pool_local:
                return pool_local
        if pd.notna(detect_bin):
            pool_local = cand_stats.index[cand_stats[detect_bin_col].astype("Int64") == int(detect_bin)].tolist()
            if pool_local:
                return pool_local
        return cand_stats.index.tolist()

    excluded = set() if excluded_genes is None else {str(g) for g in excluded_genes}
    used = set() if used_genes is None else {str(g) for g in used_genes}
    stats = gene_stats.copy()
    stats.index = stats.index.astype(str)

    sampled: Dict[str, List[str]] = {}
    for group_name, genes in marker_groups.items():
        group_sample: List[str] = []
        for gene in genes:
            g = str(gene)
            if g not in stats.index:
                continue
            row = stats.loc[g]
            mean_bin = row.get(mean_bin_col, pd.NA)
            detect_bin = row.get(detect_bin_col, pd.NA)

            candidates = stats.index.difference(list(excluded | used))
            cand_stats = stats.loc[candidates]
            pool = _candidate_pool(cand_stats, mean_bin, detect_bin)
            if not pool:
                candidates = stats.index.difference(list(excluded))
                cand_stats = stats.loc[candidates]
                pool = _candidate_pool(cand_stats, mean_bin, detect_bin)
            if not pool:
                pool = _candidate_pool(stats, mean_bin, detect_bin)
            if not pool:
                raise ValueError("No candidate genes available for random matched control.")
            choice = str(rng.choice(np.asarray(pool, dtype=object)))
            group_sample.append(choice)
            used.add(choice)
        if len(group_sample) == 0:
            raise ValueError(f"Failed to sample any genes for group: {group_name}")
        sampled[str(group_name)] = group_sample
    return sampled


def bootstrap_segmentation_marker_panel_benchmark(
    adata,
    segments_index_csv: str,
    *,
    marker_panels: Optional[Mapping[str, Mapping[str, Iterable[str]]]] = None,
    panel_names: Optional[Iterable[str]] = None,
    focus_compactness: float | Mapping[float, float] | str = "auto",
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    layer: Optional[str] = None,
    compactness_tol: float = 1e-9,
    n_bootstraps: int = 200,
    random_state: Optional[int] = 0,
    ci: tuple[float, float] = (0.025, 0.975),
    weight_by: str = "marker_mass",
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap marker panels by resampling genes within each marker group."""
    if marker_panels is None:
        marker_panels_use = get_crc_visiumhd_marker_panels(panel_names=panel_names)
    else:
        marker_panels_use = {
            str(panel): {
                str(group): [str(g) for g in genes]
                for group, genes in groups.items()
            }
            for panel, groups in marker_panels.items()
        }
    if not marker_panels_use:
        raise ValueError("No marker panels available for bootstrapping.")

    _, matched_rows = _prepare_matched_rows(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )
    workers = max(1, int(n_jobs))
    panel_gene_pool = {
        str(panel_name): {
            str(group): [str(g) for g in genes if str(g) in adata.var_names]
            for group, genes in groups.items()
        }
        for panel_name, groups in marker_panels_use.items()
    }
    panel_gene_pool = {
        panel_name: {group: genes for group, genes in groups.items() if len(genes) > 0}
        for panel_name, groups in panel_gene_pool.items()
    }
    panel_gene_pool = {panel_name: groups for panel_name, groups in panel_gene_pool.items() if groups}
    if not panel_gene_pool:
        raise ValueError("No marker-panel genes were found in adata.var_names for bootstrapping.")

    task_specs = [(int(boot_id), str(panel_name)) for boot_id in range(int(n_bootstraps)) for panel_name in panel_gene_pool]
    seed_seq = np.random.SeedSequence(random_state)
    child_seeds = seed_seq.spawn(len(task_specs))

    def _run_bootstrap_task(task_idx: int) -> Optional[pd.DataFrame]:
        boot_id, panel_name = task_specs[task_idx]
        rng = np.random.default_rng(child_seeds[task_idx])
        groups = panel_gene_pool[panel_name]
        sampled_groups = {
            group: [str(g) for g in rng.choice(np.asarray(genes, dtype=object), size=len(genes), replace=True)]
            for group, genes in groups.items()
            if len(genes) > 0
        }
        sampled_groups = {k: v for k, v in sampled_groups.items() if len(v) > 0}
        if not sampled_groups:
            return None
        marker_genes, group_indexer = _resolve_marker_groups(adata, sampled_groups)
        marker_values = _extract_gene_matrix(adata, marker_genes, layer=layer)
        comparison_df, _, _ = _compare_segmentation_marker_purity_from_matrix(
            marker_values,
            group_indexer=group_indexer,
            matched_rows=matched_rows,
            weight_by=weight_by,
            meta={"panel": panel_name, "bootstrap_id": boot_id},
        )
        return comparison_df

    boot_frames: List[pd.DataFrame] = []
    if workers == 1 or len(task_specs) <= 1:
        for task_idx in range(len(task_specs)):
            frame = _run_bootstrap_task(task_idx)
            if frame is not None and len(frame):
                boot_frames.append(frame)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(task_specs))) as ex:
            for frame in ex.map(_run_bootstrap_task, range(len(task_specs))):
                if frame is not None and len(frame):
                    boot_frames.append(frame)

    boot_df = pd.concat(boot_frames, axis=0, ignore_index=True) if boot_frames else pd.DataFrame()
    if boot_df.empty:
        return boot_df, pd.DataFrame()

    metric_cols = [c for c in boot_df.columns if str(c).startswith("gain_")]
    summary_rows: List[Dict[str, object]] = []
    lo_q, hi_q = float(ci[0]), float(ci[1])
    for (panel_name, resolution), sub in boot_df.groupby(["panel", "resolution"], sort=True):
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            summary_rows.append(
                {
                    "panel": str(panel_name),
                    "resolution": float(resolution),
                    "metric": str(metric),
                    "n_boot": int(vals.shape[0]),
                    "boot_mean": float(vals.mean()) if len(vals) else np.nan,
                    "boot_median": float(vals.median()) if len(vals) else np.nan,
                    "boot_std": float(vals.std()) if len(vals) > 1 else np.nan,
                    "ci_low": float(vals.quantile(lo_q)) if len(vals) else np.nan,
                    "ci_high": float(vals.quantile(hi_q)) if len(vals) else np.nan,
                    "positive_fraction_boot": float((vals > 0).mean()) if len(vals) else np.nan,
                }
            )
    boot_summary_df = pd.DataFrame(summary_rows)
    if not boot_summary_df.empty:
        boot_summary_df = boot_summary_df.sort_values(["panel", "metric", "resolution"]).reset_index(drop=True)
    return boot_df, boot_summary_df


def random_control_segmentation_marker_panel_benchmark(
    adata,
    segments_index_csv: str,
    *,
    marker_panels: Optional[Mapping[str, Mapping[str, Iterable[str]]]] = None,
    panel_names: Optional[Iterable[str]] = None,
    focus_compactness: float | Mapping[float, float] | str = "auto",
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    layer: Optional[str] = None,
    compactness_tol: float = 1e-9,
    n_controls: int = 100,
    random_state: Optional[int] = 0,
    n_mean_bins: int = 8,
    n_detect_bins: int = 8,
    weight_by: str = "marker_mass",
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Benchmark observed marker panels against expression-matched random controls."""
    if marker_panels is None:
        marker_panels_use = get_crc_visiumhd_marker_panels(panel_names=panel_names)
    else:
        marker_panels_use = {
            str(panel): {
                str(group): [str(g) for g in genes]
                for group, genes in groups.items()
            }
            for panel, groups in marker_panels.items()
        }
    if not marker_panels_use:
        raise ValueError("No marker panels available for control benchmarking.")

    observed_df, _, _ = compare_segmentation_marker_panel_benchmark(
        adata,
        segments_index_csv,
        marker_panels=marker_panels_use,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        layer=layer,
        compactness_tol=compactness_tol,
        weight_by=weight_by,
    )

    gene_stats = _compute_gene_statistics(adata, layer=layer)
    gene_stats["mean_bin"] = _assign_stat_bins(np.log1p(gene_stats["mean_expr"]), int(n_mean_bins))
    gene_stats["detect_bin"] = _assign_stat_bins(gene_stats["detection_rate"], int(n_detect_bins))
    excluded_genes = {str(g) for groups in marker_panels_use.values() for genes in groups.values() for g in genes if str(g) in adata.var_names}
    workers = max(1, int(n_jobs))
    _, matched_rows = _prepare_matched_rows(
        segments_index_csv,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        compactness_tol=compactness_tol,
    )
    seed_seq = np.random.SeedSequence(random_state)
    child_seeds = seed_seq.spawn(int(n_controls))

    def _run_control_task(control_id: int) -> List[pd.DataFrame]:
        rng = np.random.default_rng(child_seeds[control_id])
        used_genes: set[str] = set()
        frames_local: List[pd.DataFrame] = []
        for panel_name, groups in marker_panels_use.items():
            sampled_groups = _sample_expression_matched_marker_groups(
                groups,
                gene_stats,
                rng=rng,
                excluded_genes=excluded_genes,
                used_genes=used_genes,
            )
            marker_genes, group_indexer = _resolve_marker_groups(adata, sampled_groups)
            marker_values = _extract_gene_matrix(adata, marker_genes, layer=layer)
            control_df, _, _ = _compare_segmentation_marker_purity_from_matrix(
                marker_values,
                group_indexer=group_indexer,
                matched_rows=matched_rows,
                weight_by=weight_by,
                meta={"panel": str(panel_name), "control_id": int(control_id)},
            )
            frames_local.append(control_df)
        return frames_local

    control_frames: List[pd.DataFrame] = []
    if workers == 1 or int(n_controls) <= 1:
        for control_id in range(int(n_controls)):
            control_frames.extend(_run_control_task(control_id))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, int(n_controls))) as ex:
            for frames_local in ex.map(_run_control_task, range(int(n_controls))):
                control_frames.extend(frames_local)

    control_df = pd.concat(control_frames, axis=0, ignore_index=True) if control_frames else pd.DataFrame()
    if control_df.empty:
        return observed_df, control_df, pd.DataFrame()

    metric_cols = [c for c in control_df.columns if str(c).startswith("gain_")]
    summary_rows: List[Dict[str, object]] = []
    for (panel_name, resolution), sub in control_df.groupby(["panel", "resolution"], sort=True):
        obs_sub = observed_df[
            (observed_df["panel"].astype(str) == str(panel_name))
            & np.isclose(pd.to_numeric(observed_df["resolution"], errors="coerce"), float(resolution))
        ]
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            obs_val = pd.to_numeric(obs_sub[metric], errors="coerce").iloc[0] if len(obs_sub) else np.nan
            summary_rows.append(
                {
                    "panel": str(panel_name),
                    "resolution": float(resolution),
                    "metric": str(metric),
                    "n_controls": int(vals.shape[0]),
                    "observed": float(obs_val) if pd.notna(obs_val) else np.nan,
                    "control_mean": float(vals.mean()) if len(vals) else np.nan,
                    "control_median": float(vals.median()) if len(vals) else np.nan,
                    "control_std": float(vals.std()) if len(vals) > 1 else np.nan,
                    "control_q05": float(vals.quantile(0.05)) if len(vals) else np.nan,
                    "control_q95": float(vals.quantile(0.95)) if len(vals) else np.nan,
                    "empirical_p_right": float((vals >= obs_val).mean()) if len(vals) and pd.notna(obs_val) else np.nan,
                    "empirical_p_left": float((vals <= obs_val).mean()) if len(vals) and pd.notna(obs_val) else np.nan,
                    "z_score": (
                        float((obs_val - vals.mean()) / max(vals.std(ddof=1), 1e-12))
                        if len(vals) > 1 and pd.notna(obs_val) else np.nan
                    ),
                }
            )
    control_summary_df = pd.DataFrame(summary_rows)
    if not control_summary_df.empty:
        control_summary_df = control_summary_df.sort_values(["panel", "metric", "resolution"]).reset_index(drop=True)
    return observed_df, control_df, control_summary_df


def run_crc_visiumhd_segmentation_benchmark_suite(
    adata,
    segments_index_csv: str,
    *,
    marker_panels: Optional[Mapping[str, Mapping[str, Iterable[str]]]] = None,
    panel_names: Optional[Iterable[str]] = None,
    focus_compactness: float | Mapping[float, float] | str = "auto",
    reference_compactness: float | Mapping[float, float] | str = 1e7,
    layer: Optional[str] = None,
    compactness_tol: float = 1e-9,
    weight_by: str = "marker_mass",
    include_relative: bool = True,
    include_bootstrap: bool = True,
    n_bootstraps: int = 200,
    bootstrap_n_jobs: int = 1,
    include_random_control: bool = True,
    n_controls: int = 100,
    control_n_jobs: int = 1,
    random_state: Optional[int] = 0,
    include_label_benchmark: bool = False,
    label_obs_key: Optional[str] = None,
    label_weight_by: str = "segment_size",
) -> Dict[str, pd.DataFrame]:
    """Run the CRC/VisiumHD segmentation benchmark suite and return all result tables.

    Returns a dictionary whose keys may include:
    ``panel_df``, ``panel_aggregate_df``, ``panel_resolution_df``,
    ``boot_df``, ``boot_summary_df``,
    ``control_df``, ``control_summary_df``,
    ``label_df``, ``label_aggregate_df``, ``label_resolution_df``.
    """
    marker_panels_use = (
        get_crc_visiumhd_marker_panels(panel_names=panel_names)
        if marker_panels is None
        else {
            str(panel): {
                str(group): [str(g) for g in genes]
                for group, genes in groups.items()
            }
            for panel, groups in marker_panels.items()
        }
    )

    panel_df, panel_aggregate_df, panel_resolution_df = compare_segmentation_marker_panel_benchmark(
        adata,
        segments_index_csv,
        marker_panels=marker_panels_use,
        panel_names=panel_names,
        focus_compactness=focus_compactness,
        reference_compactness=reference_compactness,
        layer=layer,
        compactness_tol=compactness_tol,
        weight_by=weight_by,
    )
    if include_relative:
        panel_df = add_relative_gain_columns(panel_df)

    out: Dict[str, pd.DataFrame] = {
        "panel_df": panel_df,
        "panel_aggregate_df": panel_aggregate_df,
        "panel_resolution_df": panel_resolution_df,
    }

    if include_bootstrap:
        boot_df, boot_summary_df = bootstrap_segmentation_marker_panel_benchmark(
            adata,
            segments_index_csv,
            marker_panels=marker_panels_use,
            panel_names=panel_names,
            focus_compactness=focus_compactness,
            reference_compactness=reference_compactness,
            layer=layer,
            compactness_tol=compactness_tol,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
            weight_by=weight_by,
            n_jobs=bootstrap_n_jobs,
        )
        out["boot_df"] = boot_df
        out["boot_summary_df"] = boot_summary_df

    if include_random_control:
        observed_df, control_df, control_summary_df = random_control_segmentation_marker_panel_benchmark(
            adata,
            segments_index_csv,
            marker_panels=marker_panels_use,
            panel_names=panel_names,
            focus_compactness=focus_compactness,
            reference_compactness=reference_compactness,
            layer=layer,
            compactness_tol=compactness_tol,
            n_controls=n_controls,
            random_state=random_state,
            weight_by=weight_by,
            n_jobs=control_n_jobs,
        )
        if include_relative:
            observed_df = add_relative_gain_columns(observed_df)
        out["observed_df"] = observed_df
        out["control_df"] = control_df
        out["control_summary_df"] = control_summary_df

    if include_label_benchmark:
        if label_obs_key is None:
            raise ValueError("label_obs_key must be provided when include_label_benchmark=True.")
        label_df, label_aggregate_df, label_resolution_df = compare_segmentation_label_purity_summary(
            adata,
            segments_index_csv,
            obs_key=label_obs_key,
            focus_compactness=focus_compactness,
            reference_compactness=reference_compactness,
            compactness_tol=compactness_tol,
            weight_by=label_weight_by,
        )
        out["label_df"] = label_df
        out["label_aggregate_df"] = label_aggregate_df
        out["label_resolution_df"] = label_resolution_df

    return out


def summarize_crc_visiumhd_segmentation_benchmark_evidence(
    panel_df: pd.DataFrame,
    *,
    boot_summary_df: Optional[pd.DataFrame] = None,
    control_summary_df: Optional[pd.DataFrame] = None,
    metrics: Optional[Iterable[str]] = None,
    relative_suffix: str = "_pct",
    control_alpha: float = 0.05,
) -> pd.DataFrame:
    """Combine observed gains, bootstrap CI, and random-control evidence in one table."""
    if panel_df is None or len(panel_df) == 0:
        raise ValueError("panel_df must be a non-empty DataFrame.")
    if "panel" not in panel_df.columns or "resolution" not in panel_df.columns:
        raise KeyError("panel_df must contain 'panel' and 'resolution' columns.")

    if metrics is None:
        metrics_use = [
            "gain_marker_purity",
            "gain_marker_contamination",
            "gain_marker_margin",
            "gain_marker_entropy_purity",
            "gain_marker_neighbor_purity",
        ]
        metrics_use = [m for m in metrics_use if m in panel_df.columns]
        if not metrics_use:
            metrics_use = [c for c in panel_df.columns if str(c).startswith("gain_")]
    else:
        metrics_use = [str(m) for m in metrics if str(m) in panel_df.columns]
    if not metrics_use:
        raise ValueError("No valid metrics found in panel_df.")

    rows: List[Dict[str, object]] = []
    for _, row in panel_df.iterrows():
        panel_name = str(row["panel"])
        resolution = float(pd.to_numeric(pd.Series([row["resolution"]]), errors="coerce").iloc[0])
        for metric in metrics_use:
            observed = pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]
            rec: Dict[str, object] = {
                "panel": panel_name,
                "resolution": resolution,
                "metric": metric,
                "observed": float(observed) if pd.notna(observed) else np.nan,
            }
            rel_col = f"rel_{metric}{relative_suffix}"
            if rel_col in panel_df.columns:
                rec["relative_gain"] = float(pd.to_numeric(pd.Series([row[rel_col]]), errors="coerce").iloc[0])
                rec["relative_gain_positive"] = bool(pd.notna(rec["relative_gain"]) and float(rec["relative_gain"]) > 0)
            else:
                rec["relative_gain"] = np.nan
                rec["relative_gain_positive"] = np.nan

            if boot_summary_df is not None and len(boot_summary_df):
                boot_sub = boot_summary_df[
                    (boot_summary_df["panel"].astype(str) == panel_name)
                    & np.isclose(pd.to_numeric(boot_summary_df["resolution"], errors="coerce"), resolution)
                    & (boot_summary_df["metric"].astype(str) == metric)
                ]
                if len(boot_sub):
                    b = boot_sub.iloc[0]
                    rec["boot_mean"] = float(pd.to_numeric(pd.Series([b["boot_mean"]]), errors="coerce").iloc[0])
                    rec["boot_ci_low"] = float(pd.to_numeric(pd.Series([b["ci_low"]]), errors="coerce").iloc[0])
                    rec["boot_ci_high"] = float(pd.to_numeric(pd.Series([b["ci_high"]]), errors="coerce").iloc[0])
                    rec["boot_ci_excludes_zero"] = bool(
                        pd.notna(rec["boot_ci_low"]) and pd.notna(rec["boot_ci_high"])
                        and (float(rec["boot_ci_low"]) > 0 or float(rec["boot_ci_high"]) < 0)
                    )
                    rec["boot_positive_fraction"] = float(pd.to_numeric(pd.Series([b.get("positive_fraction_boot", np.nan)]), errors="coerce").iloc[0])
                else:
                    rec["boot_mean"] = np.nan
                    rec["boot_ci_low"] = np.nan
                    rec["boot_ci_high"] = np.nan
                    rec["boot_ci_excludes_zero"] = np.nan
                    rec["boot_positive_fraction"] = np.nan
            else:
                rec["boot_mean"] = np.nan
                rec["boot_ci_low"] = np.nan
                rec["boot_ci_high"] = np.nan
                rec["boot_ci_excludes_zero"] = np.nan
                rec["boot_positive_fraction"] = np.nan

            if control_summary_df is not None and len(control_summary_df):
                ctrl_sub = control_summary_df[
                    (control_summary_df["panel"].astype(str) == panel_name)
                    & np.isclose(pd.to_numeric(control_summary_df["resolution"], errors="coerce"), resolution)
                    & (control_summary_df["metric"].astype(str) == metric)
                ]
                if len(ctrl_sub):
                    c = ctrl_sub.iloc[0]
                    rec["control_mean"] = float(pd.to_numeric(pd.Series([c["control_mean"]]), errors="coerce").iloc[0])
                    rec["control_q95"] = float(pd.to_numeric(pd.Series([c["control_q95"]]), errors="coerce").iloc[0])
                    rec["control_z"] = float(pd.to_numeric(pd.Series([c.get("z_score", np.nan)]), errors="coerce").iloc[0])
                    rec["empirical_p_right"] = float(pd.to_numeric(pd.Series([c.get("empirical_p_right", np.nan)]), errors="coerce").iloc[0])
                    rec["beats_random_q95"] = bool(
                        pd.notna(rec["observed"]) and pd.notna(rec["control_q95"])
                        and float(rec["observed"]) > float(rec["control_q95"])
                    )
                    rec["beats_random_p"] = bool(
                        pd.notna(rec["empirical_p_right"]) and float(rec["empirical_p_right"]) <= float(control_alpha)
                    )
                else:
                    rec["control_mean"] = np.nan
                    rec["control_q95"] = np.nan
                    rec["control_z"] = np.nan
                    rec["empirical_p_right"] = np.nan
                    rec["beats_random_q95"] = np.nan
                    rec["beats_random_p"] = np.nan
            else:
                rec["control_mean"] = np.nan
                rec["control_q95"] = np.nan
                rec["control_z"] = np.nan
                rec["empirical_p_right"] = np.nan
                rec["beats_random_q95"] = np.nan
                rec["beats_random_p"] = np.nan

            evidence_score = 0
            for key in ["relative_gain_positive", "boot_ci_excludes_zero", "beats_random_q95", "beats_random_p"]:
                val = rec.get(key, np.nan)
                if isinstance(val, (bool, np.bool_)) and bool(val):
                    evidence_score += 1
            rec["evidence_score"] = int(evidence_score)
            rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["panel", "metric", "resolution"]).reset_index(drop=True)
    return out


def summarize_segmentation_metrics_by_resolution(
    long_df: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    """Aggregate SPIX-vs-GRID metric gains per resolution.

    Parameters
    ----------
    long_df
        Per-gene per-resolution table returned by ``compare_segmentation_metric_summary``
        or ``explain_svg_rank_gain`` with ``return_long_df=True``.
    metrics
        Metrics to summarize. Defaults to all ``gain_*`` columns in ``long_df``.
    quantiles
        Quantiles to include for each metric at each resolution.

    Returns
    -------
    pd.DataFrame
        Tidy resolution-level summary with columns such as ``mean``, ``median``,
        ``std``, ``positive_fraction``, and requested quantiles.
    """
    if long_df is None or len(long_df) == 0:
        raise ValueError("long_df must be a non-empty DataFrame. Run with return_long_df=True.")
    if "resolution" not in long_df.columns:
        raise KeyError("long_df must contain a 'resolution' column.")

    if metrics is None:
        metrics_use = [c for c in long_df.columns if str(c).startswith("gain_")]
    else:
        metrics_use = [str(m) for m in metrics if str(m) in long_df.columns]
    if not metrics_use:
        raise ValueError("No valid metrics were found in long_df.")

    rows: List[Dict[str, object]] = []
    resolutions = pd.to_numeric(long_df["resolution"], errors="coerce")
    for res in np.sort(resolutions.dropna().unique()):
        sub = long_df.loc[np.isclose(resolutions, float(res)), :]
        for metric in metrics_use:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            vals_valid = vals[vals.notna()]
            row: Dict[str, object] = {
                "resolution": float(res),
                "metric": metric,
                "n_genes": int(vals_valid.shape[0]),
                "mean": float(vals_valid.mean()) if len(vals_valid) else np.nan,
                "median": float(vals_valid.median()) if len(vals_valid) else np.nan,
                "std": float(vals_valid.std()) if len(vals_valid) else np.nan,
                "positive_fraction": float((vals_valid > 0).mean()) if len(vals_valid) else np.nan,
                "nonzero_fraction": float(vals_valid.ne(0).mean()) if len(vals_valid) else np.nan,
            }
            for q in quantiles:
                qv = float(q)
                row[f"q{int(round(qv * 100)):02d}"] = float(vals_valid.quantile(qv)) if len(vals_valid) else np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["metric", "resolution"]).reset_index(drop=True)
    return out


def plot_segmentation_metrics_by_resolution(
    resolution_df: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
    resolutions: Optional[List[float]] = None,
    stat: str = "median",
    figsize: Tuple[float, float] = (13.0, 8.0),
    title: str = "SPIX vs GRID by resolution",
    sharex: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot metric gains across resolutions.

    The left column shows the chosen statistic over resolution, and the right
    column shows the fraction of genes with positive gain.
    """
    if resolution_df is None or len(resolution_df) == 0:
        raise ValueError("resolution_df must be a non-empty DataFrame.")
    required = {"resolution", "metric", stat, "positive_fraction"}
    missing = sorted(required.difference(resolution_df.columns))
    if missing:
        raise KeyError(f"resolution_df is missing required columns: {missing}")

    if metrics is None:
        metrics_use = resolution_df["metric"].astype(str).drop_duplicates().tolist()
    else:
        metrics_use = [str(m) for m in metrics if str(m) in set(resolution_df["metric"].astype(str))]
    if not metrics_use:
        raise ValueError("None of the requested metrics were found in resolution_df.")

    res_df = resolution_df.copy()
    res_df["resolution"] = pd.to_numeric(res_df["resolution"], errors="coerce")
    if resolutions is not None:
        res_keep = np.asarray([float(r) for r in resolutions], dtype=float)
        res_df = res_df[res_df["resolution"].isin(res_keep)]
    if res_df.empty:
        raise ValueError("No requested resolutions were found in resolution_df.")

    n = len(metrics_use)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=figsize,
        sharex=sharex,
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [2.0, 1.1]},
    )

    for i, metric in enumerate(metrics_use):
        sub = res_df[res_df["metric"].astype(str) == metric].sort_values("resolution")
        x = sub["resolution"].to_numpy(dtype=float)
        y = pd.to_numeric(sub[stat], errors="coerce").to_numpy(dtype=float)
        y_pos = pd.to_numeric(sub["positive_fraction"], errors="coerce").to_numpy(dtype=float)

        ax0, ax1 = axes[i]
        color = "#1f77b4" if np.nanmedian(y) >= 0 else "#e67e22"
        ax0.plot(x, y, marker="o", linewidth=2.0, color=color)
        ax0.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        ax0.set_ylabel(_pretty_label(metric))
        ax0.grid(alpha=0.2)

        ax1.plot(x, y_pos, marker="o", linewidth=2.0, color="#4c78a8")
        ax1.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(alpha=0.2)

        if i == 0:
            ax0.set_title(title)
            ax1.set_title("Positive fraction")

        if i == n - 1:
            ax0.set_xlabel("Resolution")
            ax1.set_xlabel("Resolution")

    return fig, axes


def plot_segmentation_biology_metric_comparison(
    comparison_df: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "SPIX vs GRID biology metrics by resolution",
    sharex: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot raw SPIX/GRID values and gains for segmentation-level biology metrics."""
    if comparison_df is None or len(comparison_df) == 0:
        raise ValueError("comparison_df must be a non-empty DataFrame.")
    if "resolution" not in comparison_df.columns:
        raise KeyError("comparison_df must contain a 'resolution' column.")

    inferred = []
    for col in comparison_df.columns:
        if str(col).endswith("_SPIX"):
            base = str(col)[:-5]
            if f"{base}_GRID" in comparison_df.columns and f"gain_{base}" in comparison_df.columns:
                inferred.append(base)
    inferred = list(dict.fromkeys(inferred))
    if metrics is None:
        metrics_use = inferred
    else:
        metrics_use = [
            str(m) for m in metrics
            if f"{str(m)}_SPIX" in comparison_df.columns
            and f"{str(m)}_GRID" in comparison_df.columns
            and f"gain_{str(m)}" in comparison_df.columns
        ]
    if not metrics_use:
        raise ValueError("No valid metrics were found in comparison_df.")

    n = len(metrics_use)
    if figsize is None:
        figsize = (13.0, max(3.0, 2.7 * n))

    df = comparison_df.copy()
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    df = df.sort_values("resolution")

    fig, axes = plt.subplots(
        n,
        2,
        figsize=figsize,
        sharex=sharex,
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [1.6, 1.2]},
    )

    for i, metric in enumerate(metrics_use):
        x = df["resolution"].to_numpy(dtype=float)
        y_focus = pd.to_numeric(df[f"{metric}_SPIX"], errors="coerce").to_numpy(dtype=float)
        y_ref = pd.to_numeric(df[f"{metric}_GRID"], errors="coerce").to_numpy(dtype=float)
        y_gain = pd.to_numeric(df[f"gain_{metric}"], errors="coerce").to_numpy(dtype=float)

        ax0, ax1 = axes[i]
        ax0.plot(x, y_focus, marker="o", linewidth=2.0, color="#1f77b4", label="SPIX")
        ax0.plot(x, y_ref, marker="s", linewidth=2.0, color="#e67e22", label="GRID")
        ax0.set_ylabel(_pretty_label(metric))
        ax0.grid(alpha=0.2)

        gain_color = "#1f77b4" if np.nanmedian(y_gain) >= 0 else "#e67e22"
        ax1.plot(x, y_gain, marker="o", linewidth=2.0, color=gain_color)
        ax1.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        ax1.set_ylabel(_pretty_label(f"gain_{metric}"))
        ax1.grid(alpha=0.2)

        if i == 0:
            ax0.set_title(title)
            ax1.set_title("SPIX - GRID")
            ax0.legend(frameon=False, loc="best")

        if i == n - 1:
            ax0.set_xlabel("Resolution")
            ax1.set_xlabel("Resolution")

    return fig, axes


def plot_segmentation_panel_heatmap(
    panel_resolution_df: pd.DataFrame,
    *,
    metric: str = "gain_marker_purity",
    stat: str = "mean",
    panel_order: Optional[List[str]] = None,
    resolution_order: Optional[List[float]] = None,
    cmap: str = "RdBu_r",
    center: Optional[float] = 0.0,
    annotate: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot panel-by-resolution heatmap for a selected biology metric."""
    if panel_resolution_df is None or len(panel_resolution_df) == 0:
        raise ValueError("panel_resolution_df must be a non-empty DataFrame.")
    required = {"panel", "metric", "resolution", stat}
    missing = sorted(required.difference(panel_resolution_df.columns))
    if missing:
        raise KeyError(f"panel_resolution_df is missing required columns: {missing}")

    df = panel_resolution_df.copy()
    df = df[df["metric"].astype(str) == str(metric)].copy()
    if df.empty:
        raise ValueError(f"Metric not found in panel_resolution_df: {metric}")
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    df[stat] = pd.to_numeric(df[stat], errors="coerce")

    pivot = df.pivot_table(index="panel", columns="resolution", values=stat, aggfunc="first")
    if panel_order is not None:
        pivot = pivot.reindex([str(p) for p in panel_order])
    if resolution_order is not None:
        pivot = pivot.reindex(columns=[float(r) for r in resolution_order])
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if pivot.empty:
        raise ValueError("No finite values available for the requested heatmap.")

    vals = pivot.to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    if center is not None and finite_vals.size:
        vmax = float(np.nanmax(np.abs(finite_vals - float(center))))
        vmin = float(center) - vmax
        vmax = float(center) + vmax
    elif finite_vals.size:
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))
    else:
        vmin, vmax = -1.0, 1.0

    if figsize is None:
        figsize = (max(6.5, 1.15 * pivot.shape[1]), max(3.4, 0.72 * pivot.shape[0] + 1.6))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(_pretty_label(metric))

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{int(c) if float(c).is_integer() else c}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Marker panel")
    ax.set_title(title or f"{_pretty_label(metric)} across CRC marker panels")

    if annotate:
        span = max(abs(vmax), abs(vmin), 1e-12)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = vals[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(v) > 0.45 * span else "#222222",
                )

    return fig, ax


def plot_segmentation_panel_pareto(
    panel_df: pd.DataFrame,
    *,
    panels: Optional[List[str]] = None,
    x: str = "gain_marker_contamination",
    y: str = "gain_marker_purity",
    size: str = "gain_marker_margin",
    color_by: str = "resolution",
    annotate: bool = True,
    max_cols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "CRC marker-panel tradeoff: purity vs contamination reduction",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot panel-wise Pareto view for SPIX-vs-GRID marker benchmarks."""
    if panel_df is None or len(panel_df) == 0:
        raise ValueError("panel_df must be a non-empty DataFrame.")
    required = {"panel", x, y, size, color_by}
    missing = sorted(required.difference(panel_df.columns))
    if missing:
        raise KeyError(f"panel_df is missing required columns: {missing}")

    df = panel_df.copy()
    if panels is None:
        panel_use = df["panel"].astype(str).drop_duplicates().tolist()
    else:
        panel_use = [str(p) for p in panels]
        df = df[df["panel"].astype(str).isin(panel_use)].copy()
    if df.empty:
        raise ValueError("No rows left after panel filtering.")

    n = len(panel_use)
    ncols = min(max(1, int(max_cols)), n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (6.2 * ncols, 4.7 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    color_vals = pd.to_numeric(df[color_by], errors="coerce")
    finite_color = color_vals[np.isfinite(color_vals)]
    color_min = float(np.nanmin(finite_color)) if len(finite_color) else 0.0
    color_max = float(np.nanmax(finite_color)) if len(finite_color) else 1.0

    for ax, panel_name in zip(axes_flat, panel_use):
        sub = df[df["panel"].astype(str) == str(panel_name)].copy()
        xv = pd.to_numeric(sub[x], errors="coerce").to_numpy(dtype=float)
        yv = pd.to_numeric(sub[y], errors="coerce").to_numpy(dtype=float)
        sv = pd.to_numeric(sub[size], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        cv = pd.to_numeric(sub[color_by], errors="coerce").to_numpy(dtype=float)

        sv_clip = np.maximum(sv, 0.0)
        if np.nanmax(sv_clip) > 0:
            sizes = 60.0 + 260.0 * (sv_clip / np.nanmax(sv_clip))
        else:
            sizes = np.full(len(sub), 80.0, dtype=float)

        sc = ax.scatter(
            xv,
            yv,
            s=sizes,
            c=cv,
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(str(panel_name))
        ax.set_xlabel(_pretty_label(x))
        ax.set_ylabel(_pretty_label(y))
        ax.grid(alpha=0.2)

        if annotate:
            for _, row in sub.iterrows():
                rx = pd.to_numeric(pd.Series([row[x]]), errors="coerce").iloc[0]
                ry = pd.to_numeric(pd.Series([row[y]]), errors="coerce").iloc[0]
                rr = pd.to_numeric(pd.Series([row[color_by]]), errors="coerce").iloc[0]
                if np.isfinite(rx) and np.isfinite(ry) and np.isfinite(rr):
                    ax.annotate(
                        f"{int(rr) if float(rr).is_integer() else rr:g}",
                        (float(rx), float(ry)),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                    )

    for ax in axes_flat[n:]:
        ax.axis("off")

    if n > 0:
        cbar = fig.colorbar(sc, ax=axes_flat[:n].tolist(), shrink=0.92)
        cbar.set_label(_pretty_label(color_by))
    fig.suptitle(title)
    return fig, axes


def plot_segmentation_panel_control_comparison(
    observed_df: pd.DataFrame,
    control_summary_df: pd.DataFrame,
    *,
    metric: str = "gain_marker_purity",
    panels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Observed marker gain vs expression-matched random controls",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot observed panel gains against random-control envelopes by resolution."""
    if observed_df is None or len(observed_df) == 0:
        raise ValueError("observed_df must be a non-empty DataFrame.")
    if control_summary_df is None or len(control_summary_df) == 0:
        raise ValueError("control_summary_df must be a non-empty DataFrame.")
    required_obs = {"panel", "resolution", metric}
    required_ctrl = {"panel", "resolution", "metric", "observed", "control_q05", "control_q95", "control_mean"}
    missing_obs = sorted(required_obs.difference(observed_df.columns))
    missing_ctrl = sorted(required_ctrl.difference(control_summary_df.columns))
    if missing_obs:
        raise KeyError(f"observed_df is missing required columns: {missing_obs}")
    if missing_ctrl:
        raise KeyError(f"control_summary_df is missing required columns: {missing_ctrl}")

    ctrl = control_summary_df[control_summary_df["metric"].astype(str) == str(metric)].copy()
    if ctrl.empty:
        raise ValueError(f"Metric not found in control_summary_df: {metric}")

    if panels is None:
        panel_use = ctrl["panel"].astype(str).drop_duplicates().tolist()
    else:
        panel_use = [str(p) for p in panels]
    n = len(panel_use)
    if n == 0:
        raise ValueError("No panels available for plotting.")
    if figsize is None:
        figsize = (8.0, max(3.2, 2.8 * n))

    fig, axes = plt.subplots(n, 1, figsize=figsize, constrained_layout=True, squeeze=False, sharex=True)
    axes_flat = axes.reshape(-1)

    for ax, panel_name in zip(axes_flat, panel_use):
        obs_sub = observed_df[observed_df["panel"].astype(str) == str(panel_name)].copy()
        ctrl_sub = ctrl[ctrl["panel"].astype(str) == str(panel_name)].copy()
        if obs_sub.empty or ctrl_sub.empty:
            ax.axis("off")
            continue

        obs_sub["resolution"] = pd.to_numeric(obs_sub["resolution"], errors="coerce")
        ctrl_sub["resolution"] = pd.to_numeric(ctrl_sub["resolution"], errors="coerce")
        obs_sub = obs_sub.sort_values("resolution")
        ctrl_sub = ctrl_sub.sort_values("resolution")

        x = ctrl_sub["resolution"].to_numpy(dtype=float)
        obs_y = pd.to_numeric(obs_sub[metric], errors="coerce").to_numpy(dtype=float)
        ctrl_mean = pd.to_numeric(ctrl_sub["control_mean"], errors="coerce").to_numpy(dtype=float)
        q05 = pd.to_numeric(ctrl_sub["control_q05"], errors="coerce").to_numpy(dtype=float)
        q95 = pd.to_numeric(ctrl_sub["control_q95"], errors="coerce").to_numpy(dtype=float)

        ax.fill_between(x, q05, q95, color="#bdbdbd", alpha=0.35, label="Random-control 5-95%")
        ax.plot(x, ctrl_mean, color="#7f7f7f", linewidth=1.8, linestyle="--", label="Random-control mean")
        ax.plot(x, obs_y, color="#1f77b4", linewidth=2.2, marker="o", label="Observed")
        ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(_pretty_label(metric))
        ax.set_title(str(panel_name))
        ax.grid(alpha=0.2)

    axes_flat[0].legend(frameon=False, loc="best")
    axes_flat[-1].set_xlabel("Resolution")
    fig.suptitle(title)
    return fig, axes


def plot_crc_visiumhd_marker_benchmark_summary(
    panel_resolution_df: pd.DataFrame,
    panel_df: Optional[pd.DataFrame] = None,
    *,
    metrics: Optional[List[str]] = None,
    panels: Optional[List[str]] = None,
    resolutions: Optional[List[float]] = None,
    stat: str = "mean",
    figsize: Tuple[float, float] = (14.5, 8.2),
    title: str = "Visium HD CRC marker-panel benchmark: SPIX vs GRID",
    cmap: str = "RdBu_r",
) -> Tuple[plt.Figure, np.ndarray]:
    """Compact one-page summary for the CRC marker-panel benchmark.

    The figure is intended for manuscript triage rather than exhaustive reporting:
    three heatmaps summarize the main marker-panel gains, and an optional Pareto
    panel shows whether purity gain co-occurs with contamination reduction.
    """
    if panel_resolution_df is None or len(panel_resolution_df) == 0:
        raise ValueError("panel_resolution_df must be a non-empty DataFrame.")

    default_metrics = [
        "gain_marker_purity",
        "gain_marker_contamination",
        "gain_marker_margin",
    ]
    if metrics is None:
        metrics_use = [m for m in default_metrics if m in set(panel_resolution_df["metric"].astype(str))]
    else:
        metrics_use = [str(m) for m in metrics if str(m) in set(panel_resolution_df["metric"].astype(str))]
    if not metrics_use:
        raise ValueError("No requested metrics were found in panel_resolution_df.")

    df = panel_resolution_df.copy()
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    df[stat] = pd.to_numeric(df[stat], errors="coerce")
    if panels is None:
        panels_use = df["panel"].astype(str).drop_duplicates().tolist()
    else:
        panels_use = [str(p) for p in panels]
    if resolutions is None:
        resolutions_use = sorted(float(r) for r in df["resolution"].dropna().unique())
    else:
        resolutions_use = [float(r) for r in resolutions]

    n_heat = len(metrics_use)
    ncols = n_heat + (1 if panel_df is not None and len(panel_df) else 0)
    width_ratios = [1.0] * n_heat + ([1.15] if ncols > n_heat else [])
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes_flat = axes.reshape(-1)

    all_vals: List[float] = []
    pivots: List[pd.DataFrame] = []
    for metric in metrics_use:
        sub = df[df["metric"].astype(str) == metric].copy()
        pivot = sub.pivot_table(index="panel", columns="resolution", values=stat, aggfunc="first")
        pivot = pivot.reindex(index=panels_use, columns=resolutions_use)
        pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
        pivots.append(pivot)
        vals = pivot.to_numpy(dtype=float)
        all_vals.extend(vals[np.isfinite(vals)].tolist())

    finite = np.asarray(all_vals, dtype=float)
    if finite.size:
        vmax = float(np.nanmax(np.abs(finite)))
        vmax = vmax if vmax > 0 else 1.0
    else:
        vmax = 1.0
    vmin = -vmax

    for ax, metric, pivot in zip(axes_flat, metrics_use, pivots):
        vals = pivot.to_numpy(dtype=float)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(_pretty_label(metric), fontsize=12)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(
            [f"{int(c) if float(c).is_integer() else c:g}" for c in pivot.columns],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("Resolution")
        if ax is axes_flat[0]:
            ax.set_ylabel("Marker panel")
        else:
            ax.set_ylabel("")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = vals[i, j]
                if not np.isfinite(val):
                    continue
                label = f"{val:.2f}" if abs(val) >= 0.01 else f"{val:.1e}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(val) > 0.45 * vmax else "#222222",
                )

    cbar = fig.colorbar(im, ax=axes_flat[:n_heat].tolist(), shrink=0.78)
    cbar.set_label("SPIX - GRID gain")

    if ncols > n_heat:
        ax = axes_flat[-1]
        pdf = panel_df.copy()
        if panels is not None:
            pdf = pdf[pdf["panel"].astype(str).isin(panels_use)].copy()
        if resolutions is not None:
            pdf["resolution"] = pd.to_numeric(pdf["resolution"], errors="coerce")
            pdf = pdf[pdf["resolution"].isin(resolutions_use)].copy()
        x = pd.to_numeric(pdf["gain_marker_contamination"], errors="coerce")
        y = pd.to_numeric(pdf["gain_marker_purity"], errors="coerce")
        res = pd.to_numeric(pdf["resolution"], errors="coerce")
        panel_codes = pd.Categorical(pdf["panel"].astype(str), categories=panels_use)
        markers = ["o", "s", "^", "D", "P", "X"]
        for code, panel_name in enumerate(panels_use):
            mask = panel_codes == panel_name
            if not np.any(mask):
                continue
            ax.scatter(
                x[mask],
                y[mask],
                s=45 + 0.25 * res[mask].fillna(0.0),
                marker=markers[code % len(markers)],
                alpha=0.85,
                label=panel_name,
                edgecolors="white",
                linewidths=0.6,
            )
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_xlabel(_pretty_label("gain_marker_contamination"))
        ax.set_ylabel(_pretty_label("gain_marker_purity"))
        ax.set_title("Purity vs contamination", fontsize=12)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(title, fontsize=15)
    return fig, axes


def plot_crc_visiumhd_marker_benchmark_evidence(
    evidence_df: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
    panels: Optional[List[str]] = None,
    resolutions: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (16.0, 8.2),
    title: str = "Visium HD CRC marker-panel benchmark: intuitive robustness summary",
    cmap_gain: str = "RdBu_r",
    cmap_support: str = "YlGnBu",
    annotate: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot intuitive robustness summaries for the CRC marker-panel benchmark.

    Instead of an abstract evidence score, this figure reports:
    1. relative SPIX improvement over GRID in percent,
    2. the bootstrap fraction with positive SPIX-GRID gain in percent, and
    3. the random-control percentile, defined as ``100 * (1 - empirical_p_right)``.
    """
    if evidence_df is None or len(evidence_df) == 0:
        raise ValueError("evidence_df must be a non-empty DataFrame.")
    required = {"panel", "metric", "resolution"}
    missing = sorted(required.difference(evidence_df.columns))
    if missing:
        raise KeyError(f"evidence_df is missing required columns: {missing}")

    default_metrics = [
        "gain_marker_purity",
        "gain_marker_contamination",
        "gain_marker_margin",
    ]
    df = evidence_df.copy()
    if metrics is None:
        metrics_use = [m for m in default_metrics if m in set(df["metric"].astype(str))]
    else:
        metrics_use = [str(m) for m in metrics if str(m) in set(df["metric"].astype(str))]
    if not metrics_use:
        raise ValueError("No requested metrics were found in evidence_df.")
    if panels is None:
        panels_use = df["panel"].astype(str).drop_duplicates().tolist()
    else:
        panels_use = [str(p) for p in panels]
    df = df[df["panel"].astype(str).isin(panels_use) & df["metric"].astype(str).isin(metrics_use)].copy()
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    if resolutions is None:
        resolutions_use = sorted(float(r) for r in df["resolution"].dropna().unique())
    else:
        resolutions_use = [float(r) for r in resolutions]
        df = df[df["resolution"].isin(resolutions_use)].copy()

    row_order = [(p, m) for p in panels_use for m in metrics_use]
    row_labels = [f"{p}\n{_pretty_label(m)}" for p, m in row_order]
    rel = np.full((len(row_order), len(resolutions_use)), np.nan, dtype=float)
    boot_pct = np.full_like(rel, np.nan)
    random_pct = np.full_like(rel, np.nan)
    ci_sig = np.full_like(rel, False, dtype=bool)
    random_sig = np.full_like(rel, False, dtype=bool)

    for i, (panel_name, metric) in enumerate(row_order):
        sub = df[(df["panel"].astype(str) == panel_name) & (df["metric"].astype(str) == metric)]
        for j, res in enumerate(resolutions_use):
            hit = sub[np.isclose(pd.to_numeric(sub["resolution"], errors="coerce"), float(res))]
            if hit.empty:
                continue
            rec = hit.iloc[0]
            if "relative_gain" in hit.columns:
                rel[i, j] = pd.to_numeric(pd.Series([rec.get("relative_gain", np.nan)]), errors="coerce").iloc[0]
            elif "observed" in hit.columns:
                rel[i, j] = 100.0 * pd.to_numeric(pd.Series([rec.get("observed", np.nan)]), errors="coerce").iloc[0]
            if "boot_positive_fraction" in hit.columns:
                boot_val = pd.to_numeric(pd.Series([rec.get("boot_positive_fraction", np.nan)]), errors="coerce").iloc[0]
                boot_pct[i, j] = float(boot_val) * 100.0 if pd.notna(boot_val) and float(boot_val) <= 1.0 else boot_val
            if "empirical_p_right" in hit.columns:
                pval = pd.to_numeric(pd.Series([rec.get("empirical_p_right", np.nan)]), errors="coerce").iloc[0]
                random_pct[i, j] = 100.0 * (1.0 - float(pval)) if pd.notna(pval) else np.nan
            if "boot_ci_excludes_zero" in hit.columns:
                ci_sig[i, j] = bool(rec.get("boot_ci_excludes_zero", False))
            if "beats_random_p" in hit.columns:
                random_sig[i, j] = bool(rec.get("beats_random_p", False))
            elif "beats_random_q95" in hit.columns:
                random_sig[i, j] = bool(rec.get("beats_random_q95", False))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]},
    )
    ax0, ax1, ax2 = axes.reshape(-1)

    finite = rel[np.isfinite(rel)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im0 = ax0.imshow(rel, aspect="auto", cmap=cmap_gain, vmin=-vmax, vmax=vmax)
    cbar0 = fig.colorbar(im0, ax=ax0, shrink=0.80)
    cbar0.set_label("SPIX improvement over GRID (%)")

    im1 = ax1.imshow(boot_pct, aspect="auto", cmap=cmap_support, vmin=0, vmax=100)
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.80)
    cbar1.set_label("Bootstrap P(SPIX > GRID) (%)")

    im2 = ax2.imshow(random_pct, aspect="auto", cmap=cmap_support, vmin=0, vmax=100)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.80)
    cbar2.set_label("Observed vs random panels percentile (%)")

    for ax in [ax0, ax1, ax2]:
        ax.set_xticks(np.arange(len(resolutions_use)))
        ax.set_xticklabels(
            [f"{int(r) if float(r).is_integer() else r:g}" for r in resolutions_use],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Resolution")
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax0.set_title("Relative improvement")
    ax1.set_title("Bootstrap consistency")
    ax2.set_title("Random-control percentile")

    # Thin separators make the panel groups easier to scan without adding clutter.
    for panel_i in range(1, len(panels_use)):
        y = panel_i * len(metrics_use) - 0.5
        for ax in [ax0, ax1, ax2]:
            ax.axhline(y, color="white", linewidth=2.0)

    if annotate:
        for i in range(rel.shape[0]):
            for j in range(rel.shape[1]):
                if np.isfinite(rel[i, j]):
                    ax0.text(
                        j,
                        i,
                        f"{rel[i, j]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(rel[i, j]) > 0.45 * vmax else "#222222",
                    )
                if np.isfinite(boot_pct[i, j]):
                    label = f"{boot_pct[i, j]:.0f}"
                    if ci_sig[i, j]:
                        label += "*"
                    ax1.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if boot_pct[i, j] >= 70 else "#222222",
                        fontweight="bold" if ci_sig[i, j] else "normal",
                    )
                if np.isfinite(random_pct[i, j]):
                    label = f"{random_pct[i, j]:.0f}"
                    if random_sig[i, j]:
                        label += "*"
                    ax2.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if random_pct[i, j] >= 70 else "#222222",
                        fontweight="bold" if random_sig[i, j] else "normal",
                    )

    fig.suptitle(title, fontsize=15)
    fig.text(
        0.5,
        0.01,
        "* marks bootstrap CI excluding zero or random-control one-sided p <= 0.05.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    return fig, axes


def compare_segmentation_metric_moran_summary(
    summary_df: pd.DataFrame,
    *,
    moran_col: str = "contrast_SPIX_vs_GRID",
    metrics: Optional[List[str]] = None,
    quantile: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize how segmentation metrics track a Moran-derived gene score.

    Parameters
    ----------
    summary_df
        Gene-level table returned by ``compare_segmentation_metric_summary`` or
        ``explain_svg_rank_gain``. It should already contain a Moran-derived
        column such as ``contrast_SPIX_vs_GRID`` joined from a rank summary.
    moran_col
        Column to treat as the Moran/rank target. Larger values should mean a
        stronger SPIX advantage.
    metrics
        Segmentation-gain metrics to compare against ``moran_col``. If omitted,
        a curated list of gain metrics is used.
    quantile
        Fraction of genes used to define the low/high Moran groups. For
        example, ``0.2`` compares the bottom 20% vs top 20%.

    Returns
    -------
    metric_df, group_df
        ``metric_df`` contains one row per metric with correlation and
        top-vs-bottom Moran-group contrasts. ``group_df`` contains tidy
        per-group summaries for plotting.
    """
    if summary_df is None or len(summary_df) == 0:
        raise ValueError("summary_df must be a non-empty DataFrame.")
    if moran_col not in summary_df.columns:
        raise KeyError(f"Missing Moran column in summary_df: {moran_col}")

    q = float(quantile)
    if not (0.0 < q < 0.5):
        raise ValueError("quantile must be between 0 and 0.5.")

    if metrics is None:
        metrics_use = [
            "gain_segment_reconstruction_r2",
            "gain_within_segment_var",
            "gain_between_segment_var",
            "gain_segment_snr",
            "gain_neighbor_contrast",
            "gain_segment_moranI",
            "gain_segment_moranI_sizeaware",
            "gain_segment_detection_rate",
            "gain_segment_dynamic_range",
            "gain_segment_top10_mass",
        ]
        metrics_use = [m for m in metrics_use if m in summary_df.columns]
        if not metrics_use:
            metrics_use = [c for c in summary_df.columns if str(c).startswith("gain_")]
    else:
        metrics_use = [str(m) for m in metrics if str(m) in summary_df.columns]
    if not metrics_use:
        raise ValueError("No valid segmentation metrics found in summary_df.")

    frame = summary_df.loc[:, [moran_col] + metrics_use].apply(pd.to_numeric, errors="coerce")
    target = frame[moran_col]
    target_valid = target.dropna()
    if target_valid.empty:
        raise ValueError(f"Column {moran_col} has no finite values.")

    low_cut = float(target_valid.quantile(q))
    high_cut = float(target_valid.quantile(1.0 - q))

    metric_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    for metric in metrics_use:
        pair = frame.loc[:, [moran_col, metric]].dropna(how="any")
        if pair.empty:
            metric_rows.append(
                {
                    "metric": metric,
                    "n_genes": 0,
                    "pearson_r": np.nan,
                    "spearman_r": np.nan,
                    "metric_mean": np.nan,
                    "metric_median": np.nan,
                    "metric_positive_fraction": np.nan,
                    "moran_low_cut": low_cut,
                    "moran_high_cut": high_cut,
                    "median_low_moran": np.nan,
                    "median_mid_moran": np.nan,
                    "median_high_moran": np.nan,
                    "delta_high_minus_low": np.nan,
                    "positive_fraction_low_moran": np.nan,
                    "positive_fraction_high_moran": np.nan,
                }
            )
            continue

        vals = pair[metric]
        low_vals = vals[pair[moran_col] <= low_cut]
        mid_vals = vals[(pair[moran_col] > low_cut) & (pair[moran_col] < high_cut)]
        high_vals = vals[pair[moran_col] >= high_cut]

        metric_rows.append(
            {
                "metric": metric,
                "n_genes": int(len(pair)),
                "pearson_r": float(pair[moran_col].corr(vals, method="pearson")) if len(pair) >= 2 else np.nan,
                "spearman_r": float(pair[moran_col].corr(vals, method="spearman")) if len(pair) >= 2 else np.nan,
                "metric_mean": float(vals.mean()),
                "metric_median": float(vals.median()),
                "metric_positive_fraction": float((vals > 0).mean()),
                "moran_low_cut": low_cut,
                "moran_high_cut": high_cut,
                "median_low_moran": float(low_vals.median()) if len(low_vals) else np.nan,
                "median_mid_moran": float(mid_vals.median()) if len(mid_vals) else np.nan,
                "median_high_moran": float(high_vals.median()) if len(high_vals) else np.nan,
                "delta_high_minus_low": (
                    float(high_vals.median()) - float(low_vals.median())
                    if len(low_vals) and len(high_vals)
                    else np.nan
                ),
                "positive_fraction_low_moran": float((low_vals > 0).mean()) if len(low_vals) else np.nan,
                "positive_fraction_high_moran": float((high_vals > 0).mean()) if len(high_vals) else np.nan,
            }
        )

        for group_name, group_vals, group_mask in [
            ("low_moran", low_vals, pair[moran_col] <= low_cut),
            ("mid_moran", mid_vals, (pair[moran_col] > low_cut) & (pair[moran_col] < high_cut)),
            ("high_moran", high_vals, pair[moran_col] >= high_cut),
        ]:
            target_sub = pair.loc[group_mask, moran_col]
            group_rows.append(
                {
                    "metric": metric,
                    "moran_col": moran_col,
                    "moran_group": group_name,
                    "n_genes": int(len(group_vals)),
                    "moran_min": float(target_sub.min()) if len(target_sub) else np.nan,
                    "moran_median": float(target_sub.median()) if len(target_sub) else np.nan,
                    "moran_max": float(target_sub.max()) if len(target_sub) else np.nan,
                    "metric_mean": float(group_vals.mean()) if len(group_vals) else np.nan,
                    "metric_median": float(group_vals.median()) if len(group_vals) else np.nan,
                    "metric_positive_fraction": float((group_vals > 0).mean()) if len(group_vals) else np.nan,
                }
            )

    metric_df = pd.DataFrame(metric_rows).set_index("metric")
    if not metric_df.empty:
        metric_df = metric_df.sort_values(["spearman_r", "delta_high_minus_low"], ascending=[False, False])
    metric_df.index.name = "metric"

    group_df = pd.DataFrame(group_rows)
    if not group_df.empty:
        group_df = group_df.sort_values(["metric", "moran_group"]).reset_index(drop=True)

    return metric_df, group_df


def plot_segmentation_metric_moran_summary(
    metric_df: pd.DataFrame,
    group_df: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
    corr_col: str = "spearman_r",
    figsize: Tuple[float, float] = (13.0, 6.5),
    title: str = "Which segmentation gains track Moran advantage?",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot segmentation-metric association with a Moran-derived score."""
    if metric_df is None or len(metric_df) == 0:
        raise ValueError("metric_df must be a non-empty DataFrame.")
    if group_df is None or len(group_df) == 0:
        raise ValueError("group_df must be a non-empty DataFrame.")
    if corr_col not in metric_df.columns:
        raise KeyError(f"metric_df is missing correlation column: {corr_col}")

    if metrics is None:
        metrics_use = metric_df.index.astype(str).tolist()
    else:
        metrics_use = [str(m) for m in metrics if str(m) in metric_df.index]
    if not metrics_use:
        raise ValueError("None of the requested metrics were found in metric_df.")

    plot_df = metric_df.loc[metrics_use].copy()
    plot_df = plot_df.sort_values([corr_col, "delta_high_minus_low"], ascending=[True, True])

    group_plot = group_df[group_df["metric"].isin(plot_df.index)].copy()
    low_df = (
        group_plot[group_plot["moran_group"] == "low_moran"]
        .drop_duplicates(subset=["metric"])
        .set_index("metric")
    )
    high_df = (
        group_plot[group_plot["moran_group"] == "high_moran"]
        .drop_duplicates(subset=["metric"])
        .set_index("metric")
    )

    labels = [_pretty_label(metric) for metric in plot_df.index.astype(str)]
    ypos = np.arange(len(plot_df), dtype=float)
    corr_vals = plot_df[corr_col].to_numpy(dtype=float)
    low_vals = low_df.reindex(plot_df.index)["metric_median"].to_numpy(dtype=float)
    high_vals = high_df.reindex(plot_df.index)["metric_median"].to_numpy(dtype=float)
    low_frac = low_df.reindex(plot_df.index)["metric_positive_fraction"].to_numpy(dtype=float)
    high_frac = high_df.reindex(plot_df.index)["metric_positive_fraction"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.1, 1.5]},
        constrained_layout=True,
    )
    ax_corr, ax_shift = axes

    corr_colors = ["#1b9e77" if np.isfinite(v) and v >= 0 else "#d95f02" for v in corr_vals]
    ax_corr.barh(ypos, corr_vals, color=corr_colors, alpha=0.9, edgecolor="white", linewidth=0.7)
    ax_corr.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_corr.set_yticks(ypos)
    ax_corr.set_yticklabels(labels)
    ax_corr.set_xlabel(_pretty_label(corr_col))
    ax_corr.set_title("Metric-Moran correlation")
    ax_corr.grid(axis="x", alpha=0.2)

    ax_shift.hlines(ypos, low_vals, high_vals, color="#8f8f8f", linewidth=2.0, alpha=0.9)
    ax_shift.scatter(low_vals, ypos, color="#4c78a8", s=68, zorder=3, label="Low Moran group")
    ax_shift.scatter(high_vals, ypos, color="#e45756", s=68, zorder=3, label="High Moran group")
    ax_shift.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_shift.set_yticks(ypos)
    ax_shift.set_yticklabels([])
    ax_shift.set_xlabel("Metric median")
    ax_shift.set_title("Bottom vs top Moran genes")
    ax_shift.grid(axis="x", alpha=0.2)
    ax_shift.legend(frameon=False, loc="lower right")

    span_vals = np.concatenate([
        low_vals[np.isfinite(low_vals)],
        high_vals[np.isfinite(high_vals)],
    ])
    span = float(np.nanmax(np.abs(span_vals))) if span_vals.size else 1.0
    span = max(span, 1e-9)
    for y, lo, hi, lo_pf, hi_pf in zip(ypos, low_vals, high_vals, low_frac, high_frac):
        if np.isfinite(lo):
            ax_shift.text(
                lo - 0.03 * span,
                y,
                f"{lo_pf:.0%}" if np.isfinite(lo_pf) else "",
                va="center",
                ha="right",
                fontsize=8,
                color="#4c78a8",
            )
        if np.isfinite(hi):
            ax_shift.text(
                hi + 0.03 * span,
                y,
                f"{hi_pf:.0%}" if np.isfinite(hi_pf) else "",
                va="center",
                ha="left",
                fontsize=8,
                color="#e45756",
            )

    fig.suptitle(title, fontsize=15)
    return fig, axes


def plot_segmentation_metric_summary(
    aggregate_df: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
    stat: str = "median",
    lower: str = "q25",
    upper: str = "q75",
    sort: bool = False,
    figsize: Tuple[float, float] = (12.0, 6.0),
    title: str = "SPIX vs GRID Metric Summary",
    color_positive: str = "#1b9e77",
    color_negative: str = "#d95f02",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot aggregate SPIX-vs-GRID metric gains from `compare_segmentation_metric_summary`.

    The left panel shows the chosen aggregate statistic (default: median) with
    inter-quantile error bars. The right panel shows the fraction of genes with
    positive gain for each metric.
    """
    if aggregate_df is None or len(aggregate_df) == 0:
        raise ValueError("aggregate_df must be a non-empty DataFrame.")

    required = {stat, lower, upper, "positive_fraction"}
    missing_required = sorted(required.difference(aggregate_df.columns))
    if missing_required:
        raise KeyError(f"aggregate_df is missing required columns: {missing_required}")

    if metrics is None:
        metrics_use = [
            "gain_segment_reconstruction_r2",
            "gain_within_segment_var",
            "gain_between_segment_var",
            "gain_segment_snr",
            "gain_neighbor_contrast",
            "gain_segment_moranI",
            "gain_segment_moranI_sizeaware",
        ]
        metrics_use = [m for m in metrics_use if m in aggregate_df.index]
        if not metrics_use:
            metrics_use = aggregate_df.index.astype(str).tolist()
    else:
        metrics_use = [str(m) for m in metrics if str(m) in aggregate_df.index]
        if not metrics_use:
            raise ValueError("None of the requested metrics were found in aggregate_df.index.")

    plot_df = aggregate_df.loc[metrics_use].copy()
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    if bool(sort):
        plot_df = plot_df.sort_values(stat, ascending=True)

    labels = [_pretty_label(metric) for metric in plot_df.index.astype(str)]
    centers = plot_df[stat].to_numpy(dtype=float)
    lower_vals = plot_df[lower].to_numpy(dtype=float)
    upper_vals = plot_df[upper].to_numpy(dtype=float)
    pos_frac = plot_df["positive_fraction"].to_numpy(dtype=float)

    err_low = np.clip(centers - lower_vals, 0.0, None)
    err_high = np.clip(upper_vals - centers, 0.0, None)
    colors = [color_positive if val >= 0 else color_negative for val in centers]
    ypos = np.arange(len(plot_df), dtype=float)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [2.4, 1.1]},
        constrained_layout=True,
    )
    ax_main, ax_frac = axes

    ax_main.barh(
        ypos,
        centers,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
        xerr=np.vstack([err_low, err_high]),
        error_kw={"ecolor": "#444444", "elinewidth": 1.0, "capsize": 3},
    )
    ax_main.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_main.set_yticks(ypos)
    ax_main.set_yticklabels(labels)
    ax_main.set_xlabel(f"{_pretty_label(stat)} gain")
    ax_main.set_title(title)
    ax_main.grid(axis="x", alpha=0.2)

    ax_frac.barh(
        ypos,
        pos_frac,
        color="#4c78a8",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
    )
    ax_frac.axvline(0.5, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    ax_frac.set_xlim(0.0, 1.0)
    ax_frac.set_yticks(ypos)
    ax_frac.set_yticklabels([])
    ax_frac.set_xlabel("Fraction positive")
    ax_frac.set_title("Gene-wise win rate")
    ax_frac.grid(axis="x", alpha=0.2)

    for y, val, frac in zip(ypos, centers, pos_frac):
        ax_main.text(
            val + (0.01 if val >= 0 else -0.01),
            y,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
            color="#222222",
        )
        ax_frac.text(
            min(max(frac, 0.02), 0.98),
            y,
            f"{frac:.0%}",
            va="center",
            ha="right" if frac > 0.12 else "left",
            fontsize=9,
            color="white" if frac > 0.12 else "#222222",
            fontweight="bold",
        )

    return fig, axes


def compare_svg_autocorr_metrics(
    summary_df: pd.DataFrame,
    *,
    metric_a: str = "gain_segment_moranI",
    metric_b: str = "gain_segment_moranI_sizeaware",
    top_k: int = 50,
) -> pd.DataFrame:
    """
    Compare gene ranking stability between two autocorrelation gain metrics.

    The default compares the existing unweighted segment Moran gain against the
    size-aware sensitivity metric added for heterogeneous segment sizes.
    """

    missing = [metric for metric in (metric_a, metric_b) if metric not in summary_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for metric comparison: {missing}")

    cols = [metric_a, metric_b]
    df = summary_df.loc[:, cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "n_genes": 0,
                    "spearman_rank_corr": np.nan,
                    "pearson_value_corr": np.nan,
                    "sign_agreement_fraction": np.nan,
                    "top_k": int(top_k),
                    "top_k_overlap_fraction": np.nan,
                }
            ]
        )

    rank_a = df[metric_a].rank(method="average")
    rank_b = df[metric_b].rank(method="average")
    top_k_eff = max(1, min(int(top_k), len(df)))
    top_a = set(df[metric_a].nlargest(top_k_eff).index)
    top_b = set(df[metric_b].nlargest(top_k_eff).index)

    return pd.DataFrame(
        [
            {
                "metric_a": metric_a,
                "metric_b": metric_b,
                "n_genes": int(len(df)),
                "spearman_rank_corr": float(rank_a.corr(rank_b)),
                "pearson_value_corr": float(df[metric_a].corr(df[metric_b])),
                "sign_agreement_fraction": float((np.sign(df[metric_a]) == np.sign(df[metric_b])).mean()),
                "top_k": int(top_k_eff),
                "top_k_overlap_fraction": float(len(top_a & top_b) / top_k_eff),
            }
        ]
    )


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


def pretty_metric_label(metric: str) -> str:
    """Public wrapper for metric labels used in SPIX.analysis plots."""
    return _pretty_label(str(metric))


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
    focus_compactness: float | Mapping[float, float] | str = 0.1,
    reference_compactness: float | Mapping[float, float] | str = 1e7,
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
    for npz_path in sorted(index_path.parent.glob("*.npz")):
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                if "resolution" not in z.files or "compactness" not in z.files:
                    continue
                res_val = pd.to_numeric(pd.Series([z["resolution"].item()]), errors="coerce").iloc[0]
                comp_val = pd.to_numeric(pd.Series([z["compactness"].item()]), errors="coerce").iloc[0]
                if not np.isfinite(res_val) or not np.isfinite(comp_val):
                    continue
                scale_id = (
                    str(z["scale_id"].item())
                    if "scale_id" in z.files
                    else f"r{float(res_val):g}_c{float(comp_val):g}"
                )
        except Exception:
            continue

        file_rows.append(
            {
                "scale_id": scale_id,
                "resolution": float(res_val),
                "compactness": float(comp_val),
                "n_segments": np.nan,
                "seconds": np.nan,
                "path": str(npz_path.resolve()),
            }
        )

    if file_rows:
        file_df = pd.DataFrame(file_rows)
        idx = pd.concat([idx, file_df], ignore_index=True, sort=False)
        idx = idx.drop_duplicates(subset=["resolution", "compactness"], keep="first")

    def _resolve_compactness_map(
        value: float | Mapping[float, float] | str,
        label: str,
    ) -> Dict[float, float] | str:
        if isinstance(value, str):
            mode = str(value).strip().lower()
            if mode != "auto":
                raise ValueError(f"{label} compactness string must be 'auto'.")
            return mode
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

    def _select_row_for_target(
        candidates: pd.DataFrame,
        *,
        label: str,
        res: float,
    ) -> pd.Series:
        if candidates.empty:
            raise ValueError(f"No {label} candidates found at resolution {res:g}.")
        if len(candidates) == 1:
            return candidates.iloc[0]

        mode_priority = {
            "auto_reference": 0,
            "native_identity": 1,
            "manual_grid": 2,
        }
        cand = candidates.copy()
        if "mode" in cand.columns:
            cand["_mode_priority"] = cand["mode"].astype(str).map(mode_priority).fillna(99)
        else:
            cand["_mode_priority"] = 99
        if "seconds" in cand.columns:
            cand["_seconds_sort"] = pd.to_numeric(cand["seconds"], errors="coerce").fillna(np.inf)
        else:
            cand["_seconds_sort"] = np.inf
        cand = cand.sort_values(
            ["_mode_priority", "_seconds_sort", "compactness"],
            kind="mergesort",
        )
        top = cand.iloc[0]
        tied = cand[
            (cand["_mode_priority"] == top["_mode_priority"])
            & np.isclose(cand["compactness"], float(top["compactness"]), atol=float(compactness_tol))
        ]
        if len(tied) > 1:
            raise ValueError(
                f"Multiple equally plausible {label} rows found at resolution {res:g}. "
                "Pass an explicit compactness map to disambiguate."
            )
        return top

    matched_rows: List[Dict[str, object]] = []
    available_res = np.asarray(sorted(idx["resolution"].unique()), dtype=float)
    for res in available_res:
        res_candidates = idx[np.isclose(idx["resolution"], float(res), atol=float(compactness_tol))].copy()
        if res_candidates.empty:
            continue

        if isinstance(ref_map, str):
            ref_candidates = res_candidates.copy()
        else:
            ref_target = ref_map.get(float(res), None)
            if ref_target is None:
                continue
            ref_candidates = res_candidates[
                np.isclose(res_candidates["compactness"], float(ref_target), atol=float(compactness_tol))
            ].copy()
        if ref_candidates.empty:
            continue
        ref_row = _select_row_for_target(ref_candidates, label="reference", res=float(res))

        if isinstance(focus_map, str):
            focus_candidates = res_candidates[
                ~np.isclose(
                    res_candidates["compactness"],
                    float(ref_row["compactness"]),
                    atol=float(compactness_tol),
                )
            ].copy()
            if focus_candidates.empty:
                continue
        else:
            focus_target = focus_map.get(float(res), None)
            if focus_target is None:
                continue
            focus_candidates = res_candidates[
                np.isclose(res_candidates["compactness"], float(focus_target), atol=float(compactness_tol))
            ].copy()
            if focus_candidates.empty:
                continue
        focus_row = _select_row_for_target(focus_candidates, label="focus", res=float(res))

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
