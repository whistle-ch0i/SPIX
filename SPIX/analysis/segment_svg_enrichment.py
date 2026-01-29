from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScaleSegments:
    scale_id: str
    path: str
    resolution: Optional[float] = None
    compactness: Optional[float] = None
    n_segments: Optional[int] = None


def load_segments_index(segments_index_csv: str) -> pd.DataFrame:
    """Load `segments_index.csv` and resolve `path` relative to its folder."""
    p = Path(segments_index_csv).resolve()
    df = pd.read_csv(str(p))
    if "scale_id" not in df.columns or "path" not in df.columns:
        raise KeyError("segments_index.csv must contain columns: 'scale_id', 'path'")
    base = p.parent
    df = df.copy()
    df["scale_id"] = df["scale_id"].astype(str)
    df["path_resolved"] = [str((base / Path(x)).resolve()) for x in df["path"].astype(str).tolist()]
    return df


def load_seg_codes_from_npz(npz_path: str) -> np.ndarray:
    """Load `seg_codes` from a SPIX multiscale `segments__*.npz` file."""
    npz = np.load(npz_path, allow_pickle=True)
    if "seg_codes" not in npz.files:
        raise KeyError(f"'seg_codes' not found in npz: {npz_path}")
    seg_codes = np.asarray(npz["seg_codes"])
    if seg_codes.ndim != 1:
        raise ValueError(f"seg_codes must be 1D; got shape={seg_codes.shape}")
    return seg_codes.astype(np.int64, copy=False)


def load_scale_segments(segments_index_csv: str, *, scale_id: str) -> ScaleSegments:
    df = load_segments_index(segments_index_csv)
    row = df[df["scale_id"] == str(scale_id)]
    if row.empty:
        raise KeyError(f"scale_id '{scale_id}' not found in segments_index.csv")
    r0 = row.iloc[0]
    return ScaleSegments(
        scale_id=str(r0["scale_id"]),
        path=str(r0["path_resolved"]),
        resolution=float(r0["resolution"]) if "resolution" in row.columns and pd.notna(r0.get("resolution")) else None,
        compactness=float(r0["compactness"]) if "compactness" in row.columns and pd.notna(r0.get("compactness")) else None,
        n_segments=int(r0["n_segments"]) if "n_segments" in row.columns and pd.notna(r0.get("n_segments")) else None,
    )


def svg_gene_sets_from_moran_ranks(
    moran_ranks: str | pd.DataFrame,
    *,
    scale_ids: Optional[Sequence[str]] = None,
    top_n: int = 50,
    max_rank: Optional[float] = None,
) -> Dict[str, List[str]]:
    """Extract per-scale SVG gene sets from `multiscale_moran_ranks.csv`.

    The ranks file is expected to have gene names as the index and columns named `rank_<scale_id>`.
    """
    if isinstance(moran_ranks, str):
        df = pd.read_csv(moran_ranks, index_col=0)
    else:
        df = moran_ranks.copy()
    if df.index.name is None:
        # commonly comes from CSV with an unnamed index column
        df.index.name = "gene"

    wanted: Optional[set[str]] = None
    if scale_ids is not None:
        wanted = {str(s) for s in scale_ids}

    out: Dict[str, List[str]] = {}
    for col in df.columns:
        if not str(col).startswith("rank_"):
            continue
        sid = str(col)[len("rank_") :]
        if wanted is not None and sid not in wanted:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if max_rank is not None:
            s = s[s <= float(max_rank)]
        s = s.dropna().sort_values()
        genes = s.index.astype(str).tolist()[: int(top_n)]
        out[sid] = genes
    if not out:
        raise ValueError("No rank_<scale_id> columns found in the provided Moran ranks.")
    return out


def _get_X_and_var_names(adata, *, layer: Optional[str] = None):
    X = adata.layers[layer] if layer is not None else adata.X
    var_names = getattr(adata, "var_names", None)
    if var_names is None:
        raise ValueError("adata.var_names is required")
    return X, np.asarray(var_names).astype(str)


def _gene_indices(var_names: np.ndarray, genes: Sequence[str]) -> np.ndarray:
    want = {str(g) for g in genes}
    idx = np.flatnonzero(np.isin(var_names, list(want)))
    if idx.size == 0:
        raise ValueError("None of the requested genes were found in adata.var_names.")
    return idx.astype(np.int64, copy=False)


def compute_gene_set_score_per_tile(
    adata,
    genes: Sequence[str],
    *,
    layer: Optional[str] = None,
    dtype: str = "float32",
) -> np.ndarray:
    """Compute mean expression of `genes` per tile (obs), efficiently for sparse or dense X."""
    X, var_names = _get_X_and_var_names(adata, layer=layer)
    idx = _gene_indices(var_names, genes)
    try:
        from scipy.sparse import issparse  # type: ignore
    except Exception:
        issparse = lambda _: False  # noqa: E731

    if issparse(X):
        Xs = X[:, idx]
        # (n_obs, 1) matrix -> flat array
        score = np.asarray(Xs.mean(axis=1)).reshape(-1)
    else:
        Xd = np.asarray(X)
        score = Xd[:, idx].mean(axis=1)
    return score.astype(dtype, copy=False)


def aggregate_score_by_segment(
    seg_codes: np.ndarray,
    score_per_tile: np.ndarray,
    *,
    agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-tile scores into per-segment scores.

    Returns
    -------
    seg_score : np.ndarray
        Per-segment score aligned to segment code (0..max_code).
    seg_counts : np.ndarray
        Tile counts per segment.
    """
    seg = np.asarray(seg_codes, dtype=np.int64)
    if seg.ndim != 1:
        raise ValueError("seg_codes must be 1D")
    if score_per_tile.shape[0] != seg.shape[0]:
        raise ValueError("score_per_tile must match seg_codes length")
    if seg.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    max_code = int(np.nanmax(seg))
    if max_code < 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

    counts = np.bincount(seg, minlength=max_code + 1).astype(np.int64, copy=False)
    if agg == "sum":
        sums = np.bincount(seg, weights=score_per_tile.astype(np.float64, copy=False), minlength=max_code + 1)
        return sums.astype(np.float32, copy=False), counts
    if agg != "mean":
        raise ValueError("agg must be 'mean' or 'sum'")
    sums = np.bincount(seg, weights=score_per_tile.astype(np.float64, copy=False), minlength=max_code + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / np.maximum(1, counts).astype(np.float64)
    return mean.astype(np.float32, copy=False), counts


def add_scale_svg_enrichment_to_adata(
    adata,
    *,
    segments_index_csv: str,
    moran_ranks: str | pd.DataFrame,
    scale_id: str,
    top_n: int = 50,
    max_rank: Optional[float] = None,
    layer: Optional[str] = None,
    obs_key: Optional[str] = None,
    zscore_across_segments: bool = True,
) -> Dict[str, Any]:
    """Compute SVG gene-set enrichment for a given scale and store per-tile scores in `adata.obs`.

    This is intended for visualization via `SPIX.pl.image_plot(color_by=<obs_key>)`.
    """
    gene_sets = svg_gene_sets_from_moran_ranks(moran_ranks, scale_ids=[scale_id], top_n=top_n, max_rank=max_rank)
    genes = gene_sets[str(scale_id)]
    scale = load_scale_segments(segments_index_csv, scale_id=str(scale_id))
    seg_codes = load_seg_codes_from_npz(scale.path)
    if int(seg_codes.shape[0]) != int(adata.n_obs):
        raise ValueError(
            f"seg_codes length ({seg_codes.shape[0]}) does not match adata.n_obs ({adata.n_obs}). "
            f"Check that the segments npz was generated from the same AnnData."
        )

    score_tile = compute_gene_set_score_per_tile(adata, genes, layer=layer)
    score_seg, seg_counts = aggregate_score_by_segment(seg_codes, score_tile, agg="mean")
    if zscore_across_segments and score_seg.size:
        m = float(np.nanmean(score_seg))
        s = float(np.nanstd(score_seg))
        if s > 0:
            score_seg = (score_seg - m) / s

    # map back to tiles
    score_tile_from_seg = score_seg[seg_codes]
    if obs_key is None:
        obs_key = f"svg_enrich__{scale_id}"
    adata.obs[obs_key] = score_tile_from_seg

    return {
        "scale_id": str(scale_id),
        "obs_key": str(obs_key),
        "genes": list(genes),
        "segments_npz": scale.path,
        "seg_counts": seg_counts,
        "seg_score": score_seg,
    }

