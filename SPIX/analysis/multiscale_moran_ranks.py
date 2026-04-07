import os
import gc
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from .perform_pseudo_bulk_analysis import perform_pseudo_bulk_analysis

_GLOBAL_ADATA: Optional[sc.AnnData] = None
_GLOBAL_A_EDGES: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None


@contextmanager
def _scoped_thread_limits(n_threads: int):
    """
    Best-effort thread limiting that restores the previous process environment.

    Important for notebooks: setting OMP/MKL env vars permanently can make unrelated
    later computations slow in the same kernel.
    """
    if n_threads < 1:
        yield
        return

    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = str(n_threads)
        try:
            from threadpoolctl import threadpool_limits  # type: ignore

            with threadpool_limits(limits=n_threads):
                yield
        except Exception:
            yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _rank_min(values: np.ndarray, *, ascending: bool) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(values if ascending else -values, kind="mergesort")
    sorted_vals = values[order]

    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sorted_vals[1:] != sorted_vals[:-1]

    starts = np.where(change, np.arange(n, dtype=np.int32), 0)
    first = np.maximum.accumulate(starts)
    ranks_sorted = first + 1

    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = ranks_sorted
    return ranks


def _rank_min_desc(values: np.ndarray) -> np.ndarray:
    return _rank_min(values, ascending=False)


def _rank_min_asc(values: np.ndarray) -> np.ndarray:
    return _rank_min(values, ascending=True)


def _autocorr_metric_config(mode: str) -> Tuple[str, bool]:
    mode_norm = str(mode).lower()
    if mode_norm == "geary":
        return "C", True
    return "I", False


def _extract_autocorr_metric(
    df: pd.DataFrame,
    *,
    mode: str,
    thresh: float,
) -> Tuple[np.ndarray, pd.Index]:
    metric_col, ascending = _autocorr_metric_config(mode)
    if metric_col not in df.columns:
        raise KeyError(f"Expected autocorrelation column '{metric_col}' for mode='{mode}'")

    vals = df[metric_col].to_numpy(dtype=np.float64, copy=False)
    index = df.index.astype(str)
    keep = np.isfinite(vals)

    if metric_col == "C":
        # Geary's C is better when smaller; keep the default 0.0 threshold as "no extra filter"
        # so existing Moran-oriented notebook calls don't silently drop everything.
        eff_thresh = np.inf if float(thresh) == 0.0 else float(thresh)
        if np.isfinite(eff_thresh):
            keep &= vals <= eff_thresh
        ranker = _rank_min_asc
    else:
        if float(thresh) != -np.inf:
            keep &= vals > float(thresh)
        ranker = _rank_min_desc

    return ranker(vals[keep]), index[keep]


def _compute_moranI_fallback(
    X,
    W,
    var_names,
    chunk_size: int = 256,
) -> pd.DataFrame:
    """
    Fallback Moran's I for when `squidpy.gr.spatial_autocorr` returns empty/missing results.

    Computes Moran's I (no permutations) for each gene column in `X` given sparse weights `W`.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    try:
        from scipy.sparse import issparse  # type: ignore
    except Exception as e:
        raise RuntimeError(f"scipy.sparse is required for fallback MoranI: {e}") from e

    if not issparse(W):
        raise ValueError("W must be a scipy sparse matrix")

    n = X.shape[0]
    wsum = float(W.sum())
    if wsum <= 0:
        return pd.DataFrame({"I": np.full(X.shape[1], np.nan)}, index=var_names)

    row_sums = np.asarray(W.sum(axis=1)).reshape(-1)
    col_sums = np.asarray(W.sum(axis=0)).reshape(-1)
    if row_sums.shape[0] != n or col_sums.shape[0] != n:
        raise ValueError("W shape must match X rows")

    I = np.full(X.shape[1], np.nan, dtype=np.float64)

    if issparse(X):
        X = X.tocsr()

        for j0 in range(0, X.shape[1], chunk_size):
            j1 = min(X.shape[1], j0 + chunk_size)
            Xc = X[:, j0:j1]

            sum_x = np.asarray(Xc.sum(axis=0)).reshape(-1)
            sum_x2 = np.asarray(Xc.multiply(Xc).sum(axis=0)).reshape(-1)
            mean_x = sum_x / float(n)

            WX = W.dot(Xc)
            s1 = np.asarray(Xc.multiply(WX).sum(axis=0)).reshape(-1)
            xTr = np.asarray(Xc.T.dot(row_sums)).reshape(-1)
            xTc = np.asarray(Xc.T.dot(col_sums)).reshape(-1)

            num = s1 - mean_x * (xTr + xTc) + (mean_x * mean_x) * wsum
            den = sum_x2 - float(n) * (mean_x * mean_x)

            valid = den > 0
            out = np.full(j1 - j0, np.nan, dtype=np.float64)
            out[valid] = (float(n) / wsum) * (num[valid] / den[valid])
            I[j0:j1] = out
    else:
        X = np.asarray(X, dtype=np.float64)
        mean_x = X.mean(axis=0)
        Xc = X - mean_x
        WX = W.dot(Xc)
        num = np.sum(Xc * WX, axis=0)
        den = np.sum(Xc * Xc, axis=0)
        valid = den > 0
        I[valid] = (float(n) / wsum) * (num[valid] / den[valid])

    return pd.DataFrame({"I": I}, index=var_names)


def _resolve_segment_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    candidates = [
        (base_dir / p),
        (base_dir.parent / p),
        (base_dir / p.name),
        (base_dir.parent / p.name),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (base_dir / p).resolve()


def _moran_rank_worker(
    task: Tuple[str, str, float, str, Dict[str, Any], int]
) -> Tuple[str, Optional[pd.Series], Optional[str]]:
    sid, npz_path, moran_thresh, segment_key, pseudo_bulk_kwargs, threads_per_process = task
    try:
        with _scoped_thread_limits(int(threads_per_process)):
            if _GLOBAL_ADATA is None:
                raise RuntimeError(
                    "Global adata is not set. For parallel mode this requires start method 'fork'."
                )
            adata = _GLOBAL_ADATA

            with np.load(npz_path, allow_pickle=False) as z:
                seg_codes = z["seg_codes"].astype(np.int32, copy=False)

            if seg_codes.shape[0] != adata.n_obs:
                raise ValueError(
                    f"{sid}: seg_codes length ({seg_codes.shape[0]}) != adata.n_obs ({adata.n_obs})"
                )

            max_code = int(seg_codes.max(initial=-1))
            if max_code < 0:
                return sid, None, f"{sid}: no valid segments"

            new_adata, moran = perform_pseudo_bulk_analysis(
                adata,
                segment_key=segment_key,
                segment_codes=seg_codes,
                moranI_threshold=float(moran_thresh),
                **pseudo_bulk_kwargs,
            )
            mode = str(pseudo_bulk_kwargs.get("mode", "moran")).lower()
            if moran is None or moran.empty:
                results_key = f"{mode}C" if mode == "geary" else f"{mode}I"
                full = new_adata.uns.get(results_key, None)
                if isinstance(full, pd.DataFrame) and (not full.empty):
                    moran = full
                else:
                    W = new_adata.obsp.get("spatial_connectivities", None)
                    if W is None or getattr(W, "nnz", 0) == 0:
                        try:
                            sq.gr.spatial_neighbors(new_adata, coord_type="generic")
                            W = new_adata.obsp.get("spatial_connectivities", None)
                        except Exception:
                            W = None
                    if W is None or getattr(W, "nnz", 0) == 0:
                        available = list(new_adata.uns.keys())
                        del new_adata
                        return sid, None, f"{sid}: empty MoranI (uns keys: {available})"
                    if mode != "moran":
                        del new_adata
                        return sid, None, f"{sid}: fallback autocorr only supports mode='moran' (got mode='{mode}')"
                    moran = _compute_moranI_fallback(new_adata.X, W, new_adata.var_names)
            del new_adata

            ranks, rank_index = _extract_autocorr_metric(
                moran,
                mode=mode,
                thresh=float(moran_thresh),
            )
            series = pd.Series(ranks, index=rank_index, name=f"rank_{sid}")
            return sid, series, None
    except Exception as e:
        return sid, None, f"{sid}: {type(e).__name__}: {e}"
    finally:
        gc.collect()


def _csr_topk(mat, k: int):
    if k is None or k <= 0:
        return mat
    try:
        from scipy.sparse import csr_matrix  # type: ignore
    except Exception as e:
        raise RuntimeError(f"scipy is required for top-k sparsification: {e}") from e

    mat = mat.tocsr()
    n_rows = mat.shape[0]
    indptr = mat.indptr
    indices = mat.indices
    data = mat.data

    new_indptr = np.zeros(n_rows + 1, dtype=indptr.dtype)
    new_indices_parts = []
    new_data_parts = []

    nnz_out = 0
    for i in range(n_rows):
        start = int(indptr[i])
        end = int(indptr[i + 1])
        row_nnz = end - start
        if row_nnz <= k:
            cols = indices[start:end]
            vals = data[start:end]
        else:
            row_vals = data[start:end]
            pick = np.argpartition(row_vals, -k)[-k:]
            # stable-ish ordering (largest first) helps reproducibility
            pick = pick[np.argsort(row_vals[pick])[::-1]]
            cols = indices[start:end][pick]
            vals = row_vals[pick]
        new_indices_parts.append(cols)
        new_data_parts.append(vals)
        nnz_out += cols.size
        new_indptr[i + 1] = nnz_out

    new_indices = np.concatenate(new_indices_parts) if nnz_out else np.zeros(0, dtype=indices.dtype)
    new_data = np.concatenate(new_data_parts) if nnz_out else np.zeros(0, dtype=data.dtype)
    out = csr_matrix((new_data, new_indices, new_indptr), shape=mat.shape)
    out.eliminate_zeros()
    return out


def _weights_transform_csr(W, transformation: Optional[str]):
    """
    Apply a libpysal-style weights transformation to a CSR matrix.

    Supported:
    - 'R' / 'r': row-standardized
    - 'B' / 'b': binary
    - 'D' / 'd': double-standardized (global sum = 1)
    """
    if transformation is None:
        transformation = "R"
    t = str(transformation).strip().lower()
    if t in {"", "none", "n"}:
        return W
    W = W.tocsr()
    if t in {"b", "binary"}:
        if W.nnz:
            W.data = np.ones_like(W.data, dtype=np.float32)
        return W
    if t in {"r", "row"}:
        try:
            from scipy.sparse import csr_matrix  # type: ignore
        except Exception as e:
            raise RuntimeError(f"scipy is required for weights transform: {e}") from e
        row_sums = np.asarray(W.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
        nz = row_sums > 0
        inv = np.zeros_like(row_sums, dtype=np.float32)
        inv[nz] = (1.0 / row_sums[nz]).astype(np.float32)
        Dinv = csr_matrix((inv, (np.arange(W.shape[0]), np.arange(W.shape[0]))), shape=W.shape)
        out = (Dinv @ W).tocsr()
        out.eliminate_zeros()
        return out
    if t in {"d", "double", "global"}:
        s = float(W.sum())
        if s > 0:
            W = W.copy()
            W.data = (W.data / np.float32(s)).astype(np.float32, copy=False)
        return W
    raise ValueError(f"Unsupported weights transformation '{transformation}'. Use 'R', 'B', 'D', or None.")


def _collapse_edges_to_segments(
    n_cells: int,
    n_segs: int,
    seg_codes_compact: np.ndarray,
    edges: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    make_binary: bool = False,
    topk_per_row: Optional[int] = None,
    row_normalize: bool = False,
):
    """
    Collapse a cell-cell adjacency (provided as upper-triangular edge list) to segment adjacency.

    `edges` must be (row, col, weight) with row < col (upper triangle).
    """
    try:
        from scipy.sparse import coo_matrix, csr_matrix  # type: ignore
    except Exception as e:
        raise RuntimeError(f"scipy is required for fast graph collapse: {e}") from e

    row, col, w = edges
    if row.size == 0:
        return csr_matrix((n_segs, n_segs), dtype=np.float32)

    if seg_codes_compact.shape[0] != n_cells:
        raise ValueError("seg_codes_compact length must equal n_cells")

    s_row = seg_codes_compact[row]
    s_col = seg_codes_compact[col]
    valid = (s_row >= 0) & (s_col >= 0) & (s_row != s_col)
    if not np.any(valid):
        return csr_matrix((n_segs, n_segs), dtype=np.float32)

    s_row = s_row[valid].astype(np.int32, copy=False)
    s_col = s_col[valid].astype(np.int32, copy=False)
    ww = w[valid].astype(np.float32, copy=False)

    # Symmetrize (segment graph should be undirected)
    rows = np.concatenate([s_row, s_col])
    cols = np.concatenate([s_col, s_row])
    data = np.concatenate([ww, ww])

    B = coo_matrix((data, (rows, cols)), shape=(n_segs, n_segs), dtype=np.float32)
    B.sum_duplicates()
    B = B.tocsr()
    B.setdiag(0)
    B.eliminate_zeros()

    if make_binary and B.nnz:
        B.data = np.ones_like(B.data, dtype=np.float32)

    if topk_per_row is not None and int(topk_per_row) > 0:
        B = _csr_topk(B, int(topk_per_row))

    if row_normalize:
        row_sums = np.asarray(B.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
        nz = row_sums > 0
        inv = np.zeros_like(row_sums, dtype=np.float32)
        inv[nz] = (1.0 / row_sums[nz]).astype(np.float32)
        Dinv = csr_matrix((inv, (np.arange(n_segs), np.arange(n_segs))), shape=(n_segs, n_segs))
        B = Dinv @ B

    return B


def _aggregate_X_to_segments(
    X,
    seg_codes_compact: np.ndarray,
    n_segs: int,
    *,
    expr_agg: str = "sum",
):
    """Aggregate cell x gene `X` into segment x gene, using `seg_codes_compact` (0..n_segs-1, or -1)."""
    try:
        from scipy.sparse import csr_matrix, issparse  # type: ignore
    except Exception as e:
        raise RuntimeError(f"scipy is required for fast pseudo-bulk aggregation: {e}") from e

    expr_agg = str(expr_agg).lower()
    if expr_agg not in {"sum", "mean"}:
        raise ValueError("expr_agg must be 'sum' or 'mean'")

    n_cells = X.shape[0]
    if seg_codes_compact.shape[0] != n_cells:
        raise ValueError("seg_codes_compact length must match X rows")

    valid_cells = seg_codes_compact >= 0
    cell_idx = np.flatnonzero(valid_cells)
    if cell_idx.size == 0:
        return csr_matrix((n_segs, X.shape[1]), dtype=np.float32)

    seg_idx = seg_codes_compact[cell_idx].astype(np.int32, copy=False)
    M = csr_matrix(
        (np.ones(cell_idx.size, dtype=np.float32), (seg_idx, cell_idx)),
        shape=(n_segs, n_cells),
    )

    if issparse(X):
        X_bulk = (M @ X).tocsr()
    else:
        X_bulk = M @ np.asarray(X)
        X_bulk = csr_matrix(X_bulk, dtype=np.float32)

    if expr_agg == "mean":
        counts = np.bincount(seg_idx, minlength=n_segs).astype(np.float32)
        counts[counts <= 0] = 1.0
        inv = 1.0 / counts
        Dinv = csr_matrix((inv, (np.arange(n_segs), np.arange(n_segs))), shape=(n_segs, n_segs))
        X_bulk = (Dinv @ X_bulk).tocsr()

    X_bulk.eliminate_zeros()
    return X_bulk


def _moran_rank_fast_thread(
    adata: sc.AnnData,
    task: Tuple[str, str, float, str, Dict[str, Any], int],
    edges: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[str, Optional[pd.Series], Optional[str]]:
    sid, npz_path, moran_thresh, _segment_key, pseudo_bulk_kwargs, threads_per_process = task
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            seg_codes_raw = z["seg_codes"].astype(np.int32, copy=False)

        if seg_codes_raw.shape[0] != adata.n_obs:
            raise ValueError(
                f"{sid}: seg_codes length ({seg_codes_raw.shape[0]}) != adata.n_obs ({adata.n_obs})"
            )

        valid = seg_codes_raw >= 0
        if not np.any(valid):
            series = pd.Series(np.zeros(0, dtype=np.int32), index=pd.Index([], dtype=str), name=f"rank_{sid}")
            return sid, series, None

        # Compact segment ids to 0..n_segs-1 (saves memory if codes are sparse)
        raw_ids = seg_codes_raw[valid].astype(np.int64, copy=False)
        uniq, inv = np.unique(raw_ids, return_inverse=True)
        n_segs = int(uniq.size)
        seg_codes = np.full(seg_codes_raw.shape[0], -1, dtype=np.int32)
        seg_codes[np.flatnonzero(valid)] = inv.astype(np.int32, copy=False)

        expr_agg = pseudo_bulk_kwargs.get("expr_agg", "sum")
        normalize_total = bool(pseudo_bulk_kwargs.get("normalize_total", True))
        log_transform = bool(pseudo_bulk_kwargs.get("log_transform", True))
        min_genes_in_segment = int(pseudo_bulk_kwargs.get("min_genes_in_segment", 1))

        collapse_make_binary = bool(pseudo_bulk_kwargs.get("collapse_make_binary", False))
        collapse_topk_per_segment = pseudo_bulk_kwargs.get("collapse_topk_per_segment", None)
        collapse_row_normalize = bool(pseudo_bulk_kwargs.get("collapse_row_normalize", False))
        sq_autocorr_kwargs = pseudo_bulk_kwargs.get("sq_autocorr_kwargs", {}) or {}
        try:
            weights_transform = sq_autocorr_kwargs.get("transformation", None)
        except Exception:
            weights_transform = None

        X_bulk = _aggregate_X_to_segments(adata.X, seg_codes, n_segs, expr_agg=str(expr_agg))

        if min_genes_in_segment > 1:
            nnz_per_gene = np.asarray(X_bulk.getnnz(axis=0)).reshape(-1)
            keep = nnz_per_gene >= min_genes_in_segment
            if not np.any(keep):
                series = pd.Series(np.zeros(0, dtype=np.int32), index=pd.Index([], dtype=str), name=f"rank_{sid}")
                return sid, series, None
            X_bulk = X_bulk[:, keep]
            var_names = np.asarray(adata.var_names).astype(str)[keep]
        else:
            var_names = np.asarray(adata.var_names).astype(str)

        if normalize_total:
            row_sums = np.asarray(X_bulk.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
            nz = row_sums > 0
            scale = np.ones_like(row_sums, dtype=np.float64)
            scale[nz] = 1e4 / row_sums[nz]
            from scipy.sparse import csr_matrix  # type: ignore

            D = csr_matrix((scale.astype(np.float32), (np.arange(n_segs), np.arange(n_segs))), shape=(n_segs, n_segs))
            X_bulk = (D @ X_bulk).tocsr()
            X_bulk.eliminate_zeros()

        if log_transform:
            X_bulk.data = np.log1p(X_bulk.data)

        W = _collapse_edges_to_segments(
            n_cells=int(adata.n_obs),
            n_segs=n_segs,
            seg_codes_compact=seg_codes,
            edges=edges,
            make_binary=collapse_make_binary,
            topk_per_row=collapse_topk_per_segment,
            row_normalize=collapse_row_normalize,
        )
        # Match squidpy/libpysal default behavior more closely: apply weights transformation (default row-standardized).
        W = _weights_transform_csr(W, weights_transform)

        moran = _compute_moranI_fallback(X_bulk, W, var_names)
        i_vals = moran["I"].to_numpy(dtype=np.float64, copy=False)
        moran_index = moran.index.astype(str)

        finite = np.isfinite(i_vals)
        if float(moran_thresh) != -np.inf:
            finite = finite & (i_vals > float(moran_thresh))

        if not np.any(finite):
            series = pd.Series(np.zeros(0, dtype=np.int32), index=pd.Index([], dtype=str), name=f"rank_{sid}")
            return sid, series, None

        ranks = _rank_min_desc(i_vals[finite])
        series = pd.Series(ranks, index=moran_index[finite], name=f"rank_{sid}")
        return sid, series, None
    except Exception as e:
        return sid, None, f"{sid}: {type(e).__name__}: {e}"
    finally:
        gc.collect()


def _moran_rank_fast_worker(
    task: Tuple[str, str, float, str, Dict[str, Any], int]
) -> Tuple[str, Optional[pd.Series], Optional[str]]:
    global _GLOBAL_A_EDGES
    if _GLOBAL_A_EDGES is None:
        return task[0], None, f"{task[0]}: Global edges are not set"
    if _GLOBAL_ADATA is None:
        return task[0], None, f"{task[0]}: Global adata is not set"
    threads_per_process = task[5]
    with _scoped_thread_limits(int(threads_per_process)):
        return _moran_rank_fast_thread(_GLOBAL_ADATA, task, _GLOBAL_A_EDGES)


def _moran_rank_thread(
    adata: sc.AnnData,
    task: Tuple[str, str, float, str, Dict[str, Any], int],
) -> Tuple[str, Optional[pd.Series], Optional[str]]:
    sid, npz_path, moran_thresh, segment_key, pseudo_bulk_kwargs, _threads_per_process = task
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            seg_codes = z["seg_codes"].astype(np.int32, copy=False)

        if seg_codes.shape[0] != adata.n_obs:
            raise ValueError(
                f"{sid}: seg_codes length ({seg_codes.shape[0]}) != adata.n_obs ({adata.n_obs})"
            )

        max_code = int(seg_codes.max(initial=-1))
        if max_code < 0:
            return sid, None, f"{sid}: no valid segments"

        new_adata, moran = perform_pseudo_bulk_analysis(
            adata,
            segment_key=segment_key,
            segment_codes=seg_codes,
            moranI_threshold=float(moran_thresh),
            **pseudo_bulk_kwargs,
        )
        mode = str(pseudo_bulk_kwargs.get("mode", "moran")).lower()
        if moran is None or moran.empty:
            results_key = f"{mode}C" if mode == "geary" else f"{mode}I"
            full = new_adata.uns.get(results_key, None)
            if isinstance(full, pd.DataFrame) and (not full.empty):
                moran = full
            else:
                W = new_adata.obsp.get("spatial_connectivities", None)
                if W is None or getattr(W, "nnz", 0) == 0:
                    try:
                        sq.gr.spatial_neighbors(new_adata, coord_type="generic")
                        W = new_adata.obsp.get("spatial_connectivities", None)
                    except Exception:
                        W = None
                if W is None or getattr(W, "nnz", 0) == 0:
                    available = list(new_adata.uns.keys())
                    del new_adata
                    return sid, None, f"{sid}: empty MoranI (uns keys: {available})"
                if mode != "moran":
                    del new_adata
                    return sid, None, f"{sid}: fallback autocorr only supports mode='moran' (got mode='{mode}')"
                moran = _compute_moranI_fallback(new_adata.X, W, new_adata.var_names)
        del new_adata

        ranks, rank_index = _extract_autocorr_metric(
            moran,
            mode=mode,
            thresh=float(moran_thresh),
        )
        series = pd.Series(ranks, index=rank_index, name=f"rank_{sid}")
        return sid, series, None
    except Exception as e:
        return sid, None, f"{sid}: {type(e).__name__}: {e}"
    finally:
        gc.collect()


def multiscale_moran_ranks(
    adata: sc.AnnData,
    segments_index_csv: str,
    out_csv: Optional[str] = None,
    moran_thresh: float = 0.0,
    backend: str = "threads",  # "threads" | "process" | "sequential"
    n_jobs: Optional[int] = None,
    chunksize: int = 1,
    maxtasksperchild: int = 1,
    threads_per_process: int = 1,
    validate_paths: bool = True,
    scale_ids: Optional[List[str]] = None,
    sq_coord_type: str = "generic",
    sq_neighbors_kwargs: Optional[Dict[str, Any]] = None,
    pseudo_bulk_kwargs: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
    engine: str = "squidpy",  # "squidpy" | "fast"
) -> pd.DataFrame:
    """
    Compute Moran's I ranks for every multiscale segmentation (in parallel across scales).

    Notes
    -----
    - Default `backend="threads"` avoids `fork()` crashes in environments already using OpenMP.
    - `backend="process"` uses multiprocessing with start method 'fork' (fastest/lowest-copy on Unix),
      but can terminate with: "fork() called from a process already using GNU OpenMP".
    - Set `threads_per_process=1` to avoid CPU over-subscription (common with MKL/OpenBLAS).
    - For speed, we default to **no permutation testing** in `squidpy.gr.spatial_autocorr`
      by setting `pseudo_bulk_kwargs["sq_autocorr_kwargs"]["n_perms"] = 0` unless you override it.

    To speed up further, consider:
    - `backend="process"` (if your environment can safely `fork()`), since threads can be GIL-limited.
    - Passing `pseudo_bulk_kwargs={"collapse_topk_per_segment": 10}` to sparsify the segment graph.
    - Setting `engine="fast"` to bypass `perform_pseudo_bulk_analysis`/`squidpy.gr.spatial_autocorr` and compute Moran's I
      directly on a segment-collapsed graph (usually much faster and less memory, but should match the default
      `segment_graph_strategy="collapsed"` semantics).
    """

    if sq_neighbors_kwargs is None:
        sq_neighbors_kwargs = {}
    if pseudo_bulk_kwargs is None:
        pseudo_bulk_kwargs = {}
    if "segment_obs_columns" not in pseudo_bulk_kwargs:
        pseudo_bulk_kwargs["segment_obs_columns"] = []
    if "perform_pca" not in pseudo_bulk_kwargs:
        pseudo_bulk_kwargs["perform_pca"] = False
    if "highly_variable" not in pseudo_bulk_kwargs:
        pseudo_bulk_kwargs["highly_variable"] = False
    if "sq_autocorr_kwargs" not in pseudo_bulk_kwargs:
        pseudo_bulk_kwargs["sq_autocorr_kwargs"] = {"n_jobs": 1, "n_perms": 0}
    else:
        try:
            pseudo_bulk_kwargs["sq_autocorr_kwargs"] = dict(pseudo_bulk_kwargs["sq_autocorr_kwargs"])
            pseudo_bulk_kwargs["sq_autocorr_kwargs"].setdefault("n_jobs", 1)
            # squidpy's default is typically permutation-heavy; ranks only need the point estimate.
            pseudo_bulk_kwargs["sq_autocorr_kwargs"].setdefault("n_perms", 0)
        except Exception:
            pass

    if out_csv is None:
        out_csv = str(Path(segments_index_csv).resolve().parent / "multiscale_moran_ranks.csv")

    if "spatial_connectivities" not in adata.obsp:
        if not quiet:
            print("[neighbors] computing squidpy spatial_neighbors once...", flush=True)
        sq.gr.spatial_neighbors(adata, coord_type=sq_coord_type, **sq_neighbors_kwargs)

    index_df = pd.read_csv(segments_index_csv)
    if "scale_id" not in index_df.columns or "path" not in index_df.columns:
        raise KeyError("segments_index.csv must contain columns: 'scale_id', 'path'")

    if scale_ids is not None:
        wanted = {str(s) for s in scale_ids}
        index_df = index_df[index_df["scale_id"].astype(str).isin(wanted)].copy()
        if index_df.empty:
            raise ValueError(f"No matching scale_ids found in segments_index.csv: {sorted(wanted)}")

    base_dir = Path(segments_index_csv).resolve().parent
    index_order = [str(s) for s in index_df["scale_id"].tolist()]
    tasks: List[Tuple[str, str, float, str, Dict[str, Any], int]] = []

    segment_key = "__spix_segment__"
    if segment_key in adata.obs:
        segment_key = f"{segment_key}{os.getpid()}"

    for row in index_df.itertuples(index=False):
        sid = str(getattr(row, "scale_id"))
        raw = getattr(row, "path")
        p = _resolve_segment_path(raw, base_dir)
        if validate_paths and not p.exists():
            raise FileNotFoundError(
                f"Segment npz not found for scale_id='{sid}': resolved='{p}' (raw='{raw}', base_dir='{base_dir}')"
            )
        tasks.append((sid, str(p), float(moran_thresh), segment_key, dict(pseudo_bulk_kwargs), int(threads_per_process)))

    if not tasks:
        rank_df = pd.DataFrame()
        rank_df.to_csv(out_csv)
        return rank_df

    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)
    n_jobs = max(1, int(n_jobs))

    series_by_sid: Dict[str, pd.Series] = {}

    backend = str(backend).lower()
    if backend not in {"threads", "process", "sequential"}:
        raise ValueError("backend must be one of: 'threads', 'process', 'sequential'")

    engine = str(engine).lower()
    if engine not in {"squidpy", "fast"}:
        raise ValueError("engine must be one of: 'squidpy', 'fast'")
    mode = str(pseudo_bulk_kwargs.get("mode", "moran")).lower()
    if engine == "fast" and mode != "moran":
        raise ValueError("engine='fast' currently supports only mode='moran'")

    edges: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if engine == "fast":
        A = adata.obsp.get("spatial_connectivities", None)
        if A is None:
            raise RuntimeError("engine='fast' requires adata.obsp['spatial_connectivities']")
        try:
            from scipy.sparse import issparse  # type: ignore
        except Exception as e:
            raise RuntimeError(f"scipy is required for engine='fast': {e}") from e
        if not issparse(A):
            raise ValueError("adata.obsp['spatial_connectivities'] must be a scipy sparse matrix for engine='fast'")
        A_coo = A.tocoo()
        mask = A_coo.row < A_coo.col
        edges = (
            A_coo.row[mask].astype(np.int64, copy=False),
            A_coo.col[mask].astype(np.int64, copy=False),
            A_coo.data[mask].astype(np.float32, copy=False),
        )

    # Scope thread limits to this call so notebook kernels aren't left in a low-thread state.
    with _scoped_thread_limits(int(threads_per_process)):
        if backend == "sequential":
            for i, task in enumerate(tasks, start=1):
                if engine == "fast":
                    if edges is None:
                        raise RuntimeError("Fast engine edges were not initialized")
                    sid, series, err = _moran_rank_fast_thread(adata, task, edges)
                else:
                    sid, series, err = _moran_rank_thread(adata, task)
                if not quiet:
                    print(f"[{i}/{len(tasks)}] MoranI ranks for {sid}", flush=True)
                if err is not None:
                    warnings.warn(err)
                    continue
                if series is not None:
                    series_by_sid[sid] = series

        elif backend == "threads":
            if not quiet:
                print(f"[threads] n_jobs={n_jobs}", flush=True)
            with ThreadPoolExecutor(max_workers=min(n_jobs, len(tasks))) as ex:
                if engine == "fast":
                    if edges is None:
                        raise RuntimeError("Fast engine edges were not initialized")
                    futures = [ex.submit(_moran_rank_fast_thread, adata, task, edges) for task in tasks]
                else:
                    futures = [ex.submit(_moran_rank_thread, adata, task) for task in tasks]
                done_n = 0
                for fut in as_completed(futures):
                    sid, series, err = fut.result()
                    done_n += 1
                    if err is not None:
                        warnings.warn(err)
                        continue
                    if series is not None:
                        series_by_sid[sid] = series
                    if not quiet:
                        print(f"[{done_n}/{len(tasks)}] done: {sid}", flush=True)

        else:  # backend == "process"
            all_methods = set(mp.get_all_start_methods())
            if "fork" not in all_methods or mp.get_start_method(allow_none=True) == "spawn":
                warnings.warn(
                    "multiprocessing start method 'fork' is unavailable (or 'spawn' is active). "
                    "Falling back to threads to avoid pickling the full AnnData."
                )
                return multiscale_moran_ranks(
                    adata,
                    segments_index_csv,
                    out_csv=out_csv,
                    moran_thresh=moran_thresh,
                    backend="threads",
                    n_jobs=n_jobs,
                    chunksize=chunksize,
                    maxtasksperchild=maxtasksperchild,
                    threads_per_process=threads_per_process,
                    sq_coord_type=sq_coord_type,
                    sq_neighbors_kwargs=sq_neighbors_kwargs,
                    pseudo_bulk_kwargs=pseudo_bulk_kwargs,
                    quiet=quiet,
                    engine=engine,
                )

            global _GLOBAL_ADATA
            _GLOBAL_ADATA = adata
            global _GLOBAL_A_EDGES
            if engine == "fast":
                if edges is None:
                    raise RuntimeError("Fast engine edges were not initialized")
                _GLOBAL_A_EDGES = edges
            else:
                _GLOBAL_A_EDGES = None
            ctx = mp.get_context("fork")
            if not quiet:
                print(f"[process] n_jobs={n_jobs}, maxtasksperchild={maxtasksperchild}", flush=True)
            with ctx.Pool(processes=n_jobs, maxtasksperchild=maxtasksperchild) as pool:
                for done, (sid, series, err) in enumerate(
                    pool.imap_unordered(
                        _moran_rank_fast_worker if engine == "fast" else _moran_rank_worker,
                        tasks,
                        chunksize=chunksize,
                    ),
                    start=1,
                ):
                    if err is not None:
                        warnings.warn(err)
                        continue
                    if series is not None:
                        series_by_sid[sid] = series
                    if not quiet:
                        print(f"[{done}/{len(tasks)}] done: {sid}", flush=True)

    ordered_cols = [series_by_sid[sid] for sid in index_order if sid in series_by_sid]
    rank_df = pd.concat(ordered_cols, axis=1, sort=False) if ordered_cols else pd.DataFrame()
    if rank_df.empty and tasks:
        warnings.warn(
            "multiscale_moran_ranks produced an empty table (no scale returned non-empty MoranI). "
            "Try `moran_thresh=-1`, set `quiet=False` to see warnings, or run a single scale sequentially to debug."
        )
    rank_df.to_csv(out_csv)
    return rank_df


def fill_missing_ranks_with_worst(rank_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaNs (gene absent at a scale) with `max_rank + 1`, matching typical plotting/averaging usage.
    """

    if rank_df.empty:
        return rank_df
    arr = rank_df.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return rank_df
    max_rank = int(np.nanmax(arr))
    return rank_df.fillna(max_rank + 1)


def ranks_to_resolution_trajectory(
    rank_df: pd.DataFrame,
    segments_index_csv: str,
    fillna_worst: bool = True,
    prefix: str = "rank_",
    col_prefix: str = "res_",
) -> pd.DataFrame:
    """
    Convert per-scale rank table -> resolution trajectory table, compatible with `plot_traj(res_rank, ...)`.

    This averages ranks across all segmentations sharing the same `resolution` (e.g. different compactness),
    using `segments_index.csv` columns: `scale_id`, `resolution` (and optionally `compactness`).
    """

    if fillna_worst:
        rank_df = fill_missing_ranks_with_worst(rank_df)

    index_df = pd.read_csv(segments_index_csv)
    if "scale_id" not in index_df.columns or "resolution" not in index_df.columns:
        raise KeyError("segments_index.csv must contain columns: 'scale_id', 'resolution'")

    res_vals = pd.to_numeric(index_df["resolution"], errors="coerce")
    if res_vals.isna().any():
        raise ValueError("segments_index.csv 'resolution' must be numeric")

    scale_ids = [str(s) for s in index_df["scale_id"].tolist()]
    col_names = [f"{prefix}{sid}" for sid in scale_ids]
    existing = [c for c in col_names if c in rank_df.columns]
    if not existing:
        return pd.DataFrame(index=rank_df.index)

    tmp = pd.DataFrame({"scale_id": scale_ids, "resolution": res_vals.to_numpy()})
    tmp["col"] = col_names
    tmp = tmp[tmp["col"].isin(rank_df.columns)]

    res_rank = pd.DataFrame(index=rank_df.index)
    for res in sorted(tmp["resolution"].unique()):
        cols = tmp.loc[tmp["resolution"] == res, "col"].tolist()
        if not cols:
            continue
        label = f"{col_prefix}{format(float(res), 'g')}"
        res_rank[label] = rank_df[cols].mean(axis=1)
    return res_rank


def ranks_to_compactness_trajectory(
    rank_df: pd.DataFrame,
    segments_index_csv: str,
    fillna_worst: bool = True,
    prefix: str = "rank_",
    col_prefix: str = "comp_",
) -> pd.DataFrame:
    """
    Convert per-scale rank table -> compactness trajectory table (if `compactness` exists in index CSV).
    """

    if fillna_worst:
        rank_df = fill_missing_ranks_with_worst(rank_df)

    index_df = pd.read_csv(segments_index_csv)
    required = {"scale_id", "compactness"}
    if not required.issubset(index_df.columns):
        raise KeyError("segments_index.csv must contain columns: 'scale_id', 'compactness'")

    comp_vals = pd.to_numeric(index_df["compactness"], errors="coerce")
    if comp_vals.isna().any():
        raise ValueError("segments_index.csv 'compactness' must be numeric")

    scale_ids = [str(s) for s in index_df["scale_id"].tolist()]
    col_names = [f"{prefix}{sid}" for sid in scale_ids]

    tmp = pd.DataFrame({"scale_id": scale_ids, "compactness": comp_vals.to_numpy()})
    tmp["col"] = col_names
    tmp = tmp[tmp["col"].isin(rank_df.columns)]

    comp_rank = pd.DataFrame(index=rank_df.index)
    for comp in sorted(tmp["compactness"].unique()):
        cols = tmp.loc[tmp["compactness"] == comp, "col"].tolist()
        if not cols:
            continue
        label = f"{col_prefix}{format(float(comp), 'g')}"
        comp_rank[label] = rank_df[cols].mean(axis=1)
    return comp_rank


def _fdr_bh_simple(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    n = int(p.size)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * float(n) / (np.arange(n, dtype=np.float64) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _extract_resolution_columns(
    res_rank: pd.DataFrame,
    *,
    resolution_cols: Optional[List[str]] = None,
    prefix: str = "res_",
) -> tuple[list[str], np.ndarray]:
    if resolution_cols is None:
        resolution_cols = [c for c in res_rank.columns if str(c).startswith(str(prefix))]
    cols = [str(c) for c in resolution_cols if c in res_rank.columns]
    if len(cols) < 3:
        raise ValueError("Need at least 3 resolution columns to estimate continuous scale specificity.")
    res_vals = []
    for c in cols:
        try:
            res_vals.append(float(str(c)[len(prefix):]))
        except Exception as e:
            raise ValueError(f"Could not parse resolution value from column '{c}' with prefix '{prefix}'.") from e
    return cols, np.asarray(res_vals, dtype=np.float64)


def continuous_scale_specificity(
    res_rank: pd.DataFrame,
    *,
    resolution_cols: Optional[List[str]] = None,
    prefix: str = "res_",
    fillna_worst: bool = True,
    n_permutations: int = 2000,
    random_state: Optional[int] = 0,
    alpha: float = 0.05,
    weight_power: float = 1.0,
    min_weight_eps: float = 1e-6,
    permutation_chunk_size: int = 512,
) -> pd.DataFrame:
    """Estimate continuous multiscale specificity from per-resolution ranks.

    This avoids arbitrary early/mid/late bins and instead summarizes each gene by:
    - preferred_scale_um: weighted center of preference on the log2-resolution axis
    - breadth_log2_um: weighted spread across scales (smaller = more localized)
    - peak_prominence: separation between best and second-best resolution preference
    - specificity_stat: prominence divided by breadth
    - empirical p/q-value: permutation test against randomized scale ordering
    - leave-one-out stability metrics

    Parameters
    ----------
    res_rank
        Gene x resolution-rank table, typically from `ranks_to_resolution_trajectory`.
        Lower ranks are better.
    resolution_cols
        Optional explicit resolution columns. If omitted, uses columns starting with `prefix`.
    prefix
        Prefix used to parse resolution values from columns, default `"res_"`.
    fillna_worst
        If True, fill missing ranks with worst rank per table before scoring.
    n_permutations
        Number of within-gene permutation null draws over the scale axis.
    random_state
        Random seed for reproducible permutation nulls.
    alpha
        Significance threshold used only for the `specificity_class` summary column.
    weight_power
        Exponent applied to preference weights `1 - rank_pct`.
    min_weight_eps
        Small additive constant to stabilize weighted summaries.
    permutation_chunk_size
        Number of genes per chunk when evaluating permutation nulls.
    """
    if res_rank is None or len(res_rank) == 0:
        return pd.DataFrame(index=getattr(res_rank, "index", None))

    cols, resolutions = _extract_resolution_columns(res_rank, resolution_cols=resolution_cols, prefix=prefix)
    work = res_rank.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    if fillna_worst:
        work = fill_missing_ranks_with_worst(work)
    else:
        work = work.copy()
        max_per_col = work.max(axis=0, skipna=True)
        for c in work.columns:
            worst = float(max_per_col.get(c, np.nan))
            if not np.isfinite(worst):
                worst = 1.0
            work[c] = work[c].fillna(worst + 1.0)

    rank_mat = work.to_numpy(dtype=np.float64, copy=False)
    col_max = np.nanmax(rank_mat, axis=0)
    col_max = np.where(np.isfinite(col_max) & (col_max > 0), col_max, 1.0)
    rank_pct = rank_mat / col_max[None, :]
    pref = np.clip(1.0 - rank_pct, 0.0, 1.0)
    if float(weight_power) != 1.0:
        pref = np.power(pref, float(weight_power))

    log_res = np.log2(np.asarray(resolutions, dtype=np.float64))
    eps = float(max(min_weight_eps, 1e-12))
    weights = pref + eps
    weight_sum = np.sum(weights, axis=1)
    preferred_log = np.sum(weights * log_res[None, :], axis=1) / np.maximum(weight_sum, eps)
    preferred_scale = np.power(2.0, preferred_log)
    breadth = np.sqrt(
        np.sum(weights * (log_res[None, :] - preferred_log[:, None]) ** 2, axis=1) / np.maximum(weight_sum, eps)
    )

    best_idx = np.argmax(pref, axis=1)
    if pref.shape[1] > 1:
        second_pref = np.partition(pref, kth=pref.shape[1] - 2, axis=1)[:, -2]
    else:
        second_pref = np.zeros(pref.shape[0], dtype=np.float64)
    best_pref = pref[np.arange(pref.shape[0]), best_idx]
    peak_prominence = best_pref - second_pref
    specificity_stat = peak_prominence / (breadth + 1e-6)
    nearest_idx = np.argmin(np.abs(log_res[None, :] - preferred_log[:, None]), axis=1)

    m = int(pref.shape[1])
    rng = np.random.default_rng(random_state)
    perms = np.empty((int(max(1, n_permutations)), m), dtype=np.int64)
    base_idx = np.arange(m, dtype=np.int64)
    for i in range(perms.shape[0]):
        perms[i] = rng.permutation(base_idx)

    null_ge = np.zeros(pref.shape[0], dtype=np.int32)
    null_mean = np.zeros(pref.shape[0], dtype=np.float64)
    null_m2 = np.zeros(pref.shape[0], dtype=np.float64)

    chunk = int(max(1, permutation_chunk_size))
    for i in range(0, pref.shape[0], chunk):
        j = min(i + chunk, pref.shape[0])
        pref_chunk = pref[i:j]
        permuted = pref_chunk[:, perms]
        perm_weights = permuted + eps
        perm_wsum = np.sum(perm_weights, axis=2)
        perm_pref_log = np.sum(perm_weights * log_res[None, None, :], axis=2) / np.maximum(perm_wsum, eps)
        perm_breadth = np.sqrt(
            np.sum(perm_weights * (log_res[None, None, :] - perm_pref_log[:, :, None]) ** 2, axis=2)
            / np.maximum(perm_wsum, eps)
        )
        perm_best = np.max(permuted, axis=2)
        if m > 1:
            perm_second = np.partition(permuted, kth=m - 2, axis=2)[:, :, -2]
        else:
            perm_second = np.zeros((j - i, perms.shape[0]), dtype=np.float64)
        perm_prom = perm_best - perm_second
        perm_stat = perm_prom / (perm_breadth + 1e-6)

        obs = specificity_stat[i:j][:, None]
        null_ge[i:j] = np.sum(perm_stat >= obs, axis=1, dtype=np.int32)
        null_mean[i:j] = np.mean(perm_stat, axis=1)
        null_m2[i:j] = np.var(perm_stat, axis=1, ddof=0)

    pvalue = (null_ge.astype(np.float64) + 1.0) / (float(perms.shape[0]) + 1.0)
    qvalue = _fdr_bh_simple(pvalue)
    zscore = (specificity_stat - null_mean) / np.sqrt(np.maximum(null_m2, 1e-12))

    loo_pref = np.full((pref.shape[0], m), np.nan, dtype=np.float64)
    loo_nearest = np.full((pref.shape[0], m), -1, dtype=np.int32)
    for drop_idx in range(m):
        keep = np.arange(m) != int(drop_idx)
        pref_sub = pref[:, keep]
        log_sub = log_res[keep]
        w_sub = pref_sub + eps
        wsum_sub = np.sum(w_sub, axis=1)
        pref_log_sub = np.sum(w_sub * log_sub[None, :], axis=1) / np.maximum(wsum_sub, eps)
        loo_pref[:, drop_idx] = pref_log_sub
        loo_nearest[:, drop_idx] = np.argmin(np.abs(log_res[None, :] - pref_log_sub[:, None]), axis=1)

    stability_abs_shift = np.abs(loo_pref - preferred_log[:, None])
    stability_median_abs_shift = np.nanmedian(stability_abs_shift, axis=1)
    stability_max_abs_shift = np.nanmax(stability_abs_shift, axis=1)
    stability_fraction_same = np.mean(loo_nearest == nearest_idx[:, None], axis=1)

    out = pd.DataFrame(index=res_rank.index.copy())
    out["preferred_scale_um"] = preferred_scale
    out["preferred_log2_scale"] = preferred_log
    out["breadth_log2_um"] = breadth
    out["best_resolution"] = np.asarray(cols, dtype=object)[best_idx]
    out["nearest_resolution"] = np.asarray(cols, dtype=object)[nearest_idx]
    out["best_resolution_pref"] = best_pref
    out["second_resolution_pref"] = second_pref
    out["peak_prominence"] = peak_prominence
    out["specificity_stat"] = specificity_stat
    out["null_mean"] = null_mean
    out["null_sd"] = np.sqrt(np.maximum(null_m2, 0.0))
    out["zscore"] = zscore
    out["pvalue"] = pvalue
    out["qvalue"] = qvalue
    out["stability_median_abs_shift_log2_um"] = stability_median_abs_shift
    out["stability_max_abs_shift_log2_um"] = stability_max_abs_shift
    out["stability_fraction_same_resolution"] = stability_fraction_same
    out["specificity_class"] = np.where(qvalue <= float(alpha), "scale_specific", "broad_or_flat")
    return out.sort_values(
        ["qvalue", "specificity_stat", "peak_prominence", "stability_fraction_same_resolution"],
        ascending=[True, False, False, False],
    )


def classify_scale_specific_genes_quantile(
    res_rank: pd.DataFrame,
    *,
    region_cols: Dict[str, List[str]],
    fillna_worst: bool = True,
    q_gap_abs: float = 0.50,
    q_gap_rel: float = 0.50,
    q_best_mean: float = 0.05,
    q_best_std: float = 0.50,
    q_spec: float = 0.50,
) -> pd.DataFrame:
    """Classify scale-specific genes from resolution-rank trajectories using quantile thresholds.

    This packages the region-based notebook workflow the user had been using for
    `early / mid / late` scale specificity, but replaces hard thresholds with
    data-adaptive quantiles.

    Parameters
    ----------
    res_rank
        Gene x resolution-rank table, typically from `ranks_to_resolution_trajectory`.
        Lower ranks are better.
    region_cols
        Required mapping from region name -> resolution columns.
    fillna_worst
        If True, fill missing ranks with the worst observed rank + 1 before scoring.
    q_gap_abs
        Quantile for `gap_abs`; larger is stricter because `gap_abs` should be high.
    q_gap_rel
        Quantile for `gap_rel`; larger is stricter because `gap_rel` should be high.
    q_best_mean
        Quantile for `best_mean`; smaller is stricter because `best_mean` should be low.
    q_best_std
        Quantile for `best_std`; smaller is stricter because `best_std` should be low.
    q_spec
        Quantile for `specificity_score`; larger is stricter because `specificity_score` should be high.

    Returns
    -------
    pd.DataFrame
        The input table augmented with region-level specificity metrics and final
        `category`. The resolved thresholds are stored in `out.attrs["quantile_thresholds"]`.
    """
    if res_rank is None or len(res_rank) == 0:
        return pd.DataFrame(index=getattr(res_rank, "index", None))

    if region_cols is None:
        raise ValueError("region_cols must be provided explicitly.")

    region_cols = {
        str(region): [str(c) for c in cols if str(c) in res_rank.columns]
        for region, cols in region_cols.items()
    }
    missing = [region for region, cols in region_cols.items() if len(cols) == 0]
    if missing:
        raise ValueError(f"Missing columns for groups: {missing}")

    rank_cols = [c for cols in region_cols.values() for c in cols]
    rank_mat = res_rank.loc[:, rank_cols].apply(pd.to_numeric, errors="coerce")
    if fillna_worst:
        rank_mat = fill_missing_ranks_with_worst(rank_mat)

    rank_arr = rank_mat.to_numpy(dtype=np.float64, copy=False)
    col_max = np.nanmax(rank_arr, axis=0)
    col_max = np.where(np.isfinite(col_max) & (col_max > 0), col_max, 1.0)
    rank_pct = rank_mat.divide(col_max, axis=1).fillna(1.0)

    out = res_rank.copy()

    for region, cols in region_cols.items():
        out[f"mean_{region}"] = rank_pct[cols].mean(axis=1)
        out[f"std_{region}"] = rank_pct[cols].std(axis=1, ddof=0)

    mean_tbl = pd.DataFrame({region: out[f"mean_{region}"] for region in region_cols})
    std_tbl = pd.DataFrame({region: out[f"std_{region}"] for region in region_cols})

    sorted_means = np.sort(mean_tbl.to_numpy(dtype=np.float64, copy=False), axis=1)
    out["best_region"] = mean_tbl.idxmin(axis=1)
    out["best_mean"] = sorted_means[:, 0]
    out["second_best_mean"] = sorted_means[:, 1]
    out["gap_abs"] = out["second_best_mean"] - out["best_mean"]
    out["gap_rel"] = out["gap_abs"] / (out["second_best_mean"] + 1e-8)

    region_to_idx = {region: i for i, region in enumerate(mean_tbl.columns)}
    best_idx = out["best_region"].map(region_to_idx).to_numpy()
    out["best_std"] = std_tbl.to_numpy(dtype=np.float64, copy=False)[np.arange(len(out)), best_idx]
    out["specificity_score"] = out["gap_abs"] / (out["best_std"] + 1e-3)

    rp = rank_pct.to_numpy(dtype=np.float64, copy=False)
    others_mean = (rp.sum(axis=1, keepdims=True) - rp) / max(1, rp.shape[1] - 1)
    res_score = others_mean - rp
    best_res_idx = np.argmax(res_score, axis=1)
    if res_score.shape[1] > 1:
        second_res = np.partition(res_score, kth=res_score.shape[1] - 2, axis=1)[:, -2]
    else:
        second_res = np.zeros(res_score.shape[0], dtype=np.float64)

    out["best_resolution"] = np.asarray(rank_cols, dtype=object)[best_res_idx]
    out["best_resolution_score"] = res_score[np.arange(res_score.shape[0]), best_res_idx]
    out["best_resolution_margin"] = out["best_resolution_score"] - second_res

    min_gap_abs = float(out["gap_abs"].quantile(float(q_gap_abs)))
    min_gap_rel = float(out["gap_rel"].quantile(float(q_gap_rel)))
    max_best_mean = float(out["best_mean"].quantile(float(q_best_mean)))
    max_best_std = float(out["best_std"].quantile(float(q_best_std)))
    min_spec_score = float(out["specificity_score"].quantile(float(q_spec)))

    is_specific = (
        (out["gap_abs"] >= min_gap_abs) &
        (out["gap_rel"] >= min_gap_rel) &
        (out["best_mean"] <= max_best_mean) &
        (out["best_std"] <= max_best_std) &
        (out["specificity_score"] >= min_spec_score)
    )

    out["category"] = np.where(is_specific, out["best_region"], "mixed")
    out.attrs["quantile_thresholds"] = {
        "q_gap_abs": float(q_gap_abs),
        "q_gap_rel": float(q_gap_rel),
        "q_best_mean": float(q_best_mean),
        "q_best_std": float(q_best_std),
        "q_spec": float(q_spec),
        "min_gap_abs": min_gap_abs,
        "min_gap_rel": min_gap_rel,
        "max_best_mean": max_best_mean,
        "max_best_std": max_best_std,
        "min_spec_score": min_spec_score,
        "region_cols": {k: list(v) for k, v in region_cols.items()},
    }
    return out
