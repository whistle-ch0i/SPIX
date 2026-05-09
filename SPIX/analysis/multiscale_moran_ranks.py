import os
import gc
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

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
_NATIVE_IDENTITY_PATH = "__native_identity__"


def _running_inside_ipykernel() -> bool:
    """Return True when called from a live Jupyter/IPython kernel."""
    if "ipykernel" in sys.modules:
        return True
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return False
        return "IPKernelApp" in getattr(shell, "config", {})
    except Exception:
        return False


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _normalize_size_residualize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in {"", "none", "false", "off", "no"}:
        return None
    aliases = {
        "log_n_spots": "log_n_spots",
        "n_spots": "log_n_spots",
        "segment_size": "log_n_spots",
        "log_segment_size": "log_n_spots",
        "log_total_counts": "log_total_counts",
        "total_counts": "log_total_counts",
        "library_size": "log_total_counts",
        "log_library_size": "log_total_counts",
    }
    if key not in aliases:
        allowed = sorted(set(aliases.values()))
        raise ValueError(f"Unsupported size_residualize='{value}'. Use one of {allowed} or None.")
    return aliases[key]


def _extract_autocorr_metric(
    df: pd.DataFrame,
    *,
    mode: str,
    thresh: float,
) -> Tuple[np.ndarray, pd.Index]:
    score_series, rank_series = _extract_autocorr_metric_series(
        df,
        mode=mode,
        thresh=thresh,
    )
    return rank_series.to_numpy(dtype=np.int32, copy=False), rank_series.index


def _extract_autocorr_metric_series(
    df: pd.DataFrame,
    *,
    mode: str,
    thresh: float,
) -> Tuple[pd.Series, pd.Series]:
    metric_col, _ascending = _autocorr_metric_config(mode)
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

    kept_vals = vals[keep]
    kept_index = index[keep]
    score_series = pd.Series(kept_vals, index=kept_index, name=metric_col, dtype=np.float64)
    rank_series = pd.Series(ranker(kept_vals), index=kept_index, name="rank", dtype=np.int32)
    return score_series, rank_series


def _compute_moranI_fallback(
    X,
    W,
    var_names,
    chunk_size: int = 256,
    residualize_covariate: Optional[np.ndarray] = None,
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

    z = None
    inv00 = inv01 = inv11 = 0.0
    wz = None
    wtz = None
    c00 = c01 = c10 = c11 = 0.0
    if residualize_covariate is not None:
        z = np.asarray(residualize_covariate, dtype=np.float64).reshape(-1)
        if z.shape[0] != n:
            raise ValueError("residualize_covariate length must match X rows")
        sum_z = float(z.sum())
        sum_z2 = float(np.dot(z, z))
        det = float(n) * sum_z2 - sum_z * sum_z
        if det <= 1e-12:
            z = None
        else:
            inv00 = sum_z2 / det
            inv01 = -sum_z / det
            inv11 = float(n) / det
            wz = np.asarray(W.dot(z)).reshape(-1)
            wtz = np.asarray(W.T.dot(z)).reshape(-1)
            c00 = wsum
            c01 = float(np.dot(col_sums, z))
            c10 = float(np.dot(row_sums, z))
            c11 = float(np.dot(z, wz))

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

            if z is None:
                num = s1 - mean_x * (xTr + xTc) + (mean_x * mean_x) * wsum
                den = sum_x2 - float(n) * (mean_x * mean_x)
            else:
                zTx = np.asarray(Xc.T.dot(z)).reshape(-1)
                beta0 = inv00 * sum_x + inv01 * zTx
                beta1 = inv01 * sum_x + inv11 * zTx
                left1 = np.asarray(Xc.T.dot(wtz)).reshape(-1)
                right1 = np.asarray(Xc.T.dot(wz)).reshape(-1)
                quad = (
                    c00 * beta0 * beta0
                    + (c01 + c10) * beta0 * beta1
                    + c11 * beta1 * beta1
                )
                num = s1 - (beta0 * xTc + beta1 * left1) - (beta0 * xTr + beta1 * right1) + quad
                den = sum_x2 - (beta0 * sum_x + beta1 * zTx)

            valid = den > 0
            out = np.full(j1 - j0, np.nan, dtype=np.float64)
            out[valid] = (float(n) / wsum) * (num[valid] / den[valid])
            I[j0:j1] = out
    else:
        X = np.asarray(X, dtype=np.float64)
        if z is None:
            mean_x = X.mean(axis=0)
            Xc = X - mean_x
            WX = W.dot(Xc)
            num = np.sum(Xc * WX, axis=0)
            den = np.sum(Xc * Xc, axis=0)
        else:
            sum_x = X.sum(axis=0)
            sum_x2 = np.sum(X * X, axis=0)
            zTx = z @ X
            beta0 = inv00 * sum_x + inv01 * zTx
            beta1 = inv01 * sum_x + inv11 * zTx
            WX = W.dot(X)
            num = np.sum(X * WX, axis=0)
            num = (
                num
                - (beta0 * (col_sums @ X) + beta1 * (wtz @ X))
                - (beta0 * (row_sums @ X) + beta1 * (wz @ X))
                + (
                    c00 * beta0 * beta0
                    + (c01 + c10) * beta0 * beta1
                    + c11 * beta1 * beta1
                )
            )
            den = sum_x2 - (beta0 * sum_x + beta1 * zTx)
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
) -> Tuple[str, Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    sid, npz_path, moran_thresh, segment_key, pseudo_bulk_kwargs, threads_per_process = task
    try:
        with _scoped_thread_limits(int(threads_per_process)):
            if _GLOBAL_ADATA is None:
                raise RuntimeError(
                    "Global adata is not set. For parallel mode this requires start method 'fork'."
                )
            adata = _GLOBAL_ADATA
            if str(npz_path) == _NATIVE_IDENTITY_PATH:
                A = adata.obsp.get("spatial_connectivities", None)
                if A is None:
                    raise RuntimeError("native_identity scale requires adata.obsp['spatial_connectivities']")
                try:
                    from scipy.sparse import issparse  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"scipy is required for native_identity MoranI: {e}") from e
                if not issparse(A):
                    raise ValueError("adata.obsp['spatial_connectivities'] must be sparse for native_identity")
                A = A.maximum(A.T)
                A_coo = A.tocoo()
                mask = A_coo.row < A_coo.col
                edges = (
                    A_coo.row[mask].astype(np.int64, copy=False),
                    A_coo.col[mask].astype(np.int64, copy=False),
                    A_coo.data[mask].astype(np.float32, copy=False),
                )
                return _moran_rank_fast_native_thread(
                    adata,
                    task,
                    edges,
                )

            with np.load(npz_path, allow_pickle=False) as z:
                seg_codes = z["seg_codes"].astype(np.int32, copy=False)

            if seg_codes.shape[0] != adata.n_obs:
                raise ValueError(
                    f"{sid}: seg_codes length ({seg_codes.shape[0]}) != adata.n_obs ({adata.n_obs})"
                )

            max_code = int(seg_codes.max(initial=-1))
            if max_code < 0:
                return sid, None, None, f"{sid}: no valid segments"

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
                        return sid, None, None, f"{sid}: empty MoranI (uns keys: {available})"
                    if mode != "moran":
                        del new_adata
                        return sid, None, None, f"{sid}: fallback autocorr only supports mode='moran' (got mode='{mode}')"
                    moran = _compute_moranI_fallback(new_adata.X, W, new_adata.var_names)
            del new_adata

            score_series, rank_series = _extract_autocorr_metric_series(
                moran,
                mode=mode,
                thresh=float(moran_thresh),
            )
            metric_col, _ = _autocorr_metric_config(mode)
            rank_series.name = f"rank_{sid}"
            score_series.name = f"{metric_col}_{sid}"
            return sid, rank_series, score_series, None
    except Exception as e:
        return sid, None, None, f"{sid}: {type(e).__name__}: {e}"
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


def _edges_to_cell_graph(
    n_cells: int,
    edges: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    make_binary: bool = False,
):
    try:
        from scipy.sparse import coo_matrix  # type: ignore
    except Exception as e:
        raise RuntimeError(f"scipy is required for native graph construction: {e}") from e

    row, col, w = edges
    rows = np.concatenate([row, col]).astype(np.int64, copy=False)
    cols = np.concatenate([col, row]).astype(np.int64, copy=False)
    data = np.concatenate([w, w]).astype(np.float32, copy=False)
    W = coo_matrix((data, (rows, cols)), shape=(int(n_cells), int(n_cells)), dtype=np.float32)
    W.sum_duplicates()
    W = W.tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    if make_binary and W.nnz:
        W.data = np.ones_like(W.data, dtype=np.float32)
    return W


def _moran_rank_fast_native_thread(
    adata: sc.AnnData,
    task: Tuple[str, str, float, str, Dict[str, Any], int],
    edges: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[str, Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    sid, _npz_path, moran_thresh, _segment_key, pseudo_bulk_kwargs, _threads_per_process = task
    try:
        try:
            from scipy.sparse import csr_matrix, issparse  # type: ignore
        except Exception as e:
            raise RuntimeError(f"scipy is required for native_identity fast MoranI: {e}") from e

        mode = str(pseudo_bulk_kwargs.get("mode", "moran")).lower()
        if mode != "moran":
            return sid, None, None, f"{sid}: native_identity fast path supports mode='moran' only (got mode='{mode}')"

        normalize_total = bool(pseudo_bulk_kwargs.get("normalize_total", True))
        log_transform = bool(pseudo_bulk_kwargs.get("log_transform", True))
        min_genes_in_segment = int(pseudo_bulk_kwargs.get("min_genes_in_segment", 1))
        size_residualize = _normalize_size_residualize(pseudo_bulk_kwargs.get("size_residualize", None))
        sq_autocorr_kwargs = pseudo_bulk_kwargs.get("sq_autocorr_kwargs", {}) or {}
        try:
            weights_transform = sq_autocorr_kwargs.get("transformation", None)
        except Exception:
            weights_transform = None

        X = adata.X
        if issparse(X):
            X_work = X.tocsr(copy=True)
        else:
            X_work = csr_matrix(np.asarray(X, dtype=np.float32))

        if min_genes_in_segment > 1:
            nnz_per_gene = np.asarray(X_work.getnnz(axis=0)).reshape(-1)
            keep = nnz_per_gene >= min_genes_in_segment
            if not np.any(keep):
                rank_series = pd.Series(np.zeros(0, dtype=np.int32), index=pd.Index([], dtype=str), name=f"rank_{sid}")
                score_series = pd.Series(np.zeros(0, dtype=np.float64), index=pd.Index([], dtype=str), name=f"I_{sid}")
                return sid, rank_series, score_series, None
            X_work = X_work[:, keep].tocsr()
            var_names = np.asarray(adata.var_names).astype(str)[keep]
        else:
            var_names = np.asarray(adata.var_names).astype(str)

        raw_row_sums = np.asarray(X_work.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
        if normalize_total:
            row_sums = raw_row_sums
            nz = row_sums > 0
            scale = np.ones_like(row_sums, dtype=np.float64)
            scale[nz] = 1e4 / row_sums[nz]
            D = csr_matrix(
                (scale.astype(np.float32), (np.arange(X_work.shape[0]), np.arange(X_work.shape[0]))),
                shape=(X_work.shape[0], X_work.shape[0]),
            )
            X_work = (D @ X_work).tocsr()
            X_work.eliminate_zeros()

        if log_transform:
            X_work.data = np.log1p(X_work.data)

        residualize_covariate = None
        if size_residualize == "log_n_spots":
            residualize_covariate = np.zeros(int(adata.n_obs), dtype=np.float64)
        elif size_residualize == "log_total_counts":
            residualize_covariate = np.log1p(raw_row_sums)

        W = _edges_to_cell_graph(
            int(adata.n_obs),
            edges,
            make_binary=bool(pseudo_bulk_kwargs.get("collapse_make_binary", False)),
        )
        topk = pseudo_bulk_kwargs.get("collapse_topk_per_segment", None)
        if topk is not None and int(topk) > 0:
            W = _csr_topk(W, int(topk))
        if bool(pseudo_bulk_kwargs.get("collapse_row_normalize", False)):
            W = _weights_transform_csr(W, "R")
        W = _weights_transform_csr(W, weights_transform)

        moran = _compute_moranI_fallback(
            X_work,
            W,
            var_names,
            residualize_covariate=residualize_covariate,
        )
        score_series, rank_series = _extract_autocorr_metric_series(
            moran,
            mode=mode,
            thresh=float(moran_thresh),
        )
        metric_col, _ = _autocorr_metric_config(mode)
        rank_series.name = f"rank_{sid}"
        score_series.name = f"{metric_col}_{sid}"
        return sid, rank_series, score_series, None
    except Exception as e:
        return sid, None, None, f"{sid}: {type(e).__name__}: {e}"
    finally:
        gc.collect()


def _moran_rank_fast_thread(
    adata: sc.AnnData,
    task: Tuple[str, str, float, str, Dict[str, Any], int],
    edges: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[str, Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    sid, npz_path, moran_thresh, _segment_key, pseudo_bulk_kwargs, threads_per_process = task
    try:
        if str(npz_path) == _NATIVE_IDENTITY_PATH:
            return _moran_rank_fast_native_thread(adata, task, edges)

        with np.load(npz_path, allow_pickle=False) as z:
            seg_codes_raw = z["seg_codes"].astype(np.int32, copy=False)

        if seg_codes_raw.shape[0] != adata.n_obs:
            raise ValueError(
                f"{sid}: seg_codes length ({seg_codes_raw.shape[0]}) != adata.n_obs ({adata.n_obs})"
            )

        valid = seg_codes_raw >= 0
        if not np.any(valid):
            rank_series = pd.Series(np.zeros(0, dtype=np.int32), index=pd.Index([], dtype=str), name=f"rank_{sid}")
            score_series = pd.Series(np.zeros(0, dtype=np.float64), index=pd.Index([], dtype=str), name=f"I_{sid}")
            return sid, rank_series, score_series, None

        # Compact segment ids to 0..n_segs-1 (saves memory if codes are sparse)
        raw_ids = seg_codes_raw[valid].astype(np.int64, copy=False)
        uniq, inv = np.unique(raw_ids, return_inverse=True)
        n_segs = int(uniq.size)
        seg_codes = np.full(seg_codes_raw.shape[0], -1, dtype=np.int32)
        seg_codes[np.flatnonzero(valid)] = inv.astype(np.int32, copy=False)

        expr_agg = pseudo_bulk_kwargs.get("expr_agg", "sum")
        mode = str(pseudo_bulk_kwargs.get("mode", "moran")).lower()
        normalize_total = bool(pseudo_bulk_kwargs.get("normalize_total", True))
        log_transform = bool(pseudo_bulk_kwargs.get("log_transform", True))
        min_genes_in_segment = int(pseudo_bulk_kwargs.get("min_genes_in_segment", 1))
        size_residualize = _normalize_size_residualize(pseudo_bulk_kwargs.get("size_residualize", None))

        collapse_make_binary = bool(pseudo_bulk_kwargs.get("collapse_make_binary", False))
        collapse_topk_per_segment = pseudo_bulk_kwargs.get("collapse_topk_per_segment", None)
        collapse_row_normalize = bool(pseudo_bulk_kwargs.get("collapse_row_normalize", False))
        sq_autocorr_kwargs = pseudo_bulk_kwargs.get("sq_autocorr_kwargs", {}) or {}
        try:
            weights_transform = sq_autocorr_kwargs.get("transformation", None)
        except Exception:
            weights_transform = None

        X_bulk = _aggregate_X_to_segments(adata.X, seg_codes, n_segs, expr_agg=str(expr_agg))
        seg_sizes = np.bincount(seg_codes[valid], minlength=n_segs).astype(np.float64, copy=False)

        if min_genes_in_segment > 1:
            nnz_per_gene = np.asarray(X_bulk.getnnz(axis=0)).reshape(-1)
            keep = nnz_per_gene >= min_genes_in_segment
            if not np.any(keep):
                rank_series = pd.Series(np.zeros(0, dtype=np.int32), index=pd.Index([], dtype=str), name=f"rank_{sid}")
                score_series = pd.Series(np.zeros(0, dtype=np.float64), index=pd.Index([], dtype=str), name=f"I_{sid}")
                return sid, rank_series, score_series, None
            X_bulk = X_bulk[:, keep]
            var_names = np.asarray(adata.var_names).astype(str)[keep]
        else:
            var_names = np.asarray(adata.var_names).astype(str)

        raw_row_sums = np.asarray(X_bulk.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)

        if normalize_total:
            row_sums = raw_row_sums
            nz = row_sums > 0
            scale = np.ones_like(row_sums, dtype=np.float64)
            scale[nz] = 1e4 / row_sums[nz]
            from scipy.sparse import csr_matrix  # type: ignore

            D = csr_matrix((scale.astype(np.float32), (np.arange(n_segs), np.arange(n_segs))), shape=(n_segs, n_segs))
            X_bulk = (D @ X_bulk).tocsr()
            X_bulk.eliminate_zeros()

        if log_transform:
            X_bulk.data = np.log1p(X_bulk.data)

        residualize_covariate = None
        if size_residualize == "log_n_spots":
            residualize_covariate = np.log1p(seg_sizes)
        elif size_residualize == "log_total_counts":
            residualize_covariate = np.log1p(raw_row_sums)

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

        moran = _compute_moranI_fallback(
            X_bulk,
            W,
            var_names,
            residualize_covariate=residualize_covariate,
        )
        score_series, rank_series = _extract_autocorr_metric_series(
            moran,
            mode=mode,
            thresh=float(moran_thresh),
        )
        metric_col, _ = _autocorr_metric_config(mode)
        rank_series.name = f"rank_{sid}"
        score_series.name = f"{metric_col}_{sid}"
        return sid, rank_series, score_series, None
    except Exception as e:
        return sid, None, None, f"{sid}: {type(e).__name__}: {e}"
    finally:
        gc.collect()


def _moran_rank_fast_worker(
    task: Tuple[str, str, float, str, Dict[str, Any], int]
) -> Tuple[str, Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    global _GLOBAL_A_EDGES
    if _GLOBAL_A_EDGES is None:
        return task[0], None, None, f"{task[0]}: Global edges are not set"
    if _GLOBAL_ADATA is None:
        return task[0], None, None, f"{task[0]}: Global adata is not set"
    threads_per_process = task[5]
    with _scoped_thread_limits(int(threads_per_process)):
        return _moran_rank_fast_thread(_GLOBAL_ADATA, task, _GLOBAL_A_EDGES)


def _moran_rank_thread(
    adata: sc.AnnData,
    task: Tuple[str, str, float, str, Dict[str, Any], int],
) -> Tuple[str, Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    sid, npz_path, moran_thresh, segment_key, pseudo_bulk_kwargs, _threads_per_process = task
    try:
        if str(npz_path) == _NATIVE_IDENTITY_PATH:
            A = adata.obsp.get("spatial_connectivities", None)
            if A is None:
                raise RuntimeError("native_identity scale requires adata.obsp['spatial_connectivities']")
            try:
                from scipy.sparse import issparse  # type: ignore
            except Exception as e:
                raise RuntimeError(f"scipy is required for native_identity MoranI: {e}") from e
            if not issparse(A):
                raise ValueError("adata.obsp['spatial_connectivities'] must be sparse for native_identity")
            A = A.maximum(A.T)
            A_coo = A.tocoo()
            mask = A_coo.row < A_coo.col
            edges = (
                A_coo.row[mask].astype(np.int64, copy=False),
                A_coo.col[mask].astype(np.int64, copy=False),
                A_coo.data[mask].astype(np.float32, copy=False),
            )
            return _moran_rank_fast_native_thread(adata, task, edges)

        with np.load(npz_path, allow_pickle=False) as z:
            seg_codes = z["seg_codes"].astype(np.int32, copy=False)

        if seg_codes.shape[0] != adata.n_obs:
            raise ValueError(
                f"{sid}: seg_codes length ({seg_codes.shape[0]}) != adata.n_obs ({adata.n_obs})"
            )

        max_code = int(seg_codes.max(initial=-1))
        if max_code < 0:
            return sid, None, None, f"{sid}: no valid segments"

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
                    return sid, None, None, f"{sid}: empty MoranI (uns keys: {available})"
                if mode != "moran":
                    del new_adata
                    return sid, None, None, f"{sid}: fallback autocorr only supports mode='moran' (got mode='{mode}')"
                moran = _compute_moranI_fallback(new_adata.X, W, new_adata.var_names)
        del new_adata

        score_series, rank_series = _extract_autocorr_metric_series(
            moran,
            mode=mode,
            thresh=float(moran_thresh),
        )
        metric_col, _ = _autocorr_metric_config(mode)
        rank_series.name = f"rank_{sid}"
        score_series.name = f"{metric_col}_{sid}"
        return sid, rank_series, score_series, None
    except Exception as e:
        return sid, None, None, f"{sid}: {type(e).__name__}: {e}"
    finally:
        gc.collect()


def multiscale_moran_ranks(
    adata: sc.AnnData,
    segments_index_csv: str,
    out_csv: Optional[str] = None,
    out_score_csv: Optional[str] = None,
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
    return_scores: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
    - Setting `engine="fast"` bypasses `perform_pseudo_bulk_analysis`/`squidpy.gr.spatial_autocorr` and computes Moran's I
      directly on a segment-collapsed graph (usually much faster and less memory, but should match the default
      `segment_graph_strategy="collapsed"` semantics).
    - `pseudo_bulk_kwargs["size_residualize"]` is available only for `engine="fast"` and currently supports
      `"log_n_spots"` and `"log_total_counts"`. It residualizes each gene against the chosen segment-size covariate
      before Moran ranking, which is useful as a lightweight sensitivity analysis for size-driven SVG ranks.
    - If `out_score_csv` is provided, the raw autocorrelation metric table is also written with columns like
      `I_<scale_id>` for Moran's I or `C_<scale_id>` for Geary's C.
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
        is_native = False
        try:
            native_value = getattr(row, "native_identity")
            if isinstance(native_value, str):
                is_native = native_value.strip().lower() in {"1", "true", "yes", "y", "on"}
            else:
                is_native = bool(native_value)
        except Exception:
            is_native = False
        try:
            is_native = is_native or str(getattr(row, "mode")).strip().lower() == "native_identity"
        except Exception:
            pass
        raw = getattr(row, "path")
        if is_native or str(raw) == _NATIVE_IDENTITY_PATH:
            tasks.append((sid, _NATIVE_IDENTITY_PATH, float(moran_thresh), segment_key, dict(pseudo_bulk_kwargs), int(threads_per_process)))
            continue
        p = _resolve_segment_path(raw, base_dir)
        if validate_paths and not p.exists():
            raise FileNotFoundError(
                f"Segment npz not found for scale_id='{sid}': resolved='{p}' (raw='{raw}', base_dir='{base_dir}')"
            )
        tasks.append((sid, str(p), float(moran_thresh), segment_key, dict(pseudo_bulk_kwargs), int(threads_per_process)))

    if not tasks:
        rank_df = pd.DataFrame()
        score_df = pd.DataFrame()
        rank_df.to_csv(out_csv)
        if out_score_csv is not None:
            score_df.to_csv(out_score_csv)
        if return_scores:
            return rank_df, score_df
        return rank_df

    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)
    n_jobs = max(1, int(n_jobs))

    series_by_sid: Dict[str, pd.Series] = {}
    score_series_by_sid: Dict[str, pd.Series] = {}

    backend = str(backend).lower()
    if backend not in {"threads", "process", "sequential"}:
        raise ValueError("backend must be one of: 'threads', 'process', 'sequential'")
    if (
        backend == "process"
        and _running_inside_ipykernel()
        and not _truthy_env("SPIX_ALLOW_NOTEBOOK_FORK")
    ):
        warnings.warn(
            "backend='process' was requested inside a Jupyter/ipykernel session. "
            "Forking a live notebook kernel can leave large ipykernel worker copies "
            "and kill the kernel during repeated multiscale runs. Falling back to "
            "backend='threads' for notebook safety. Set SPIX_ALLOW_NOTEBOOK_FORK=1 "
            "to force the old behavior.",
            RuntimeWarning,
        )
        backend = "threads"

    engine = str(engine).lower()
    if engine not in {"squidpy", "fast"}:
        raise ValueError("engine must be one of: 'squidpy', 'fast'")
    size_residualize = _normalize_size_residualize(pseudo_bulk_kwargs.get("size_residualize", None))
    if size_residualize is not None:
        pseudo_bulk_kwargs["size_residualize"] = size_residualize
        if engine != "fast":
            raise ValueError("pseudo_bulk_kwargs['size_residualize'] currently requires engine='fast'")
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
        # Squidpy's generic spatial graph can be directed. Moran ranking uses an
        # undirected segment-contact graph, so preserve either-direction contacts
        # before taking the upper triangle.
        A = A.maximum(A.T)
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
                    sid, rank_series, score_series, err = _moran_rank_fast_thread(adata, task, edges)
                else:
                    sid, rank_series, score_series, err = _moran_rank_thread(adata, task)
                if not quiet:
                    print(f"[{i}/{len(tasks)}] MoranI ranks for {sid}", flush=True)
                if err is not None:
                    warnings.warn(err)
                    continue
                if rank_series is not None:
                    series_by_sid[sid] = rank_series
                if score_series is not None:
                    score_series_by_sid[sid] = score_series

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
                    sid, rank_series, score_series, err = fut.result()
                    done_n += 1
                    if err is not None:
                        warnings.warn(err)
                        continue
                    if rank_series is not None:
                        series_by_sid[sid] = rank_series
                    if score_series is not None:
                        score_series_by_sid[sid] = score_series
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
                    out_score_csv=out_score_csv,
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
                    return_scores=return_scores,
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
            try:
                with ctx.Pool(processes=n_jobs, maxtasksperchild=maxtasksperchild) as pool:
                    for done, (sid, rank_series, score_series, err) in enumerate(
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
                        if rank_series is not None:
                            series_by_sid[sid] = rank_series
                        if score_series is not None:
                            score_series_by_sid[sid] = score_series
                        if not quiet:
                            print(f"[{done}/{len(tasks)}] done: {sid}", flush=True)
            finally:
                _GLOBAL_ADATA = None
                _GLOBAL_A_EDGES = None
                gc.collect()

    ordered_cols = [series_by_sid[sid] for sid in index_order if sid in series_by_sid]
    rank_df = pd.concat(ordered_cols, axis=1, sort=False) if ordered_cols else pd.DataFrame()
    ordered_score_cols = [score_series_by_sid[sid] for sid in index_order if sid in score_series_by_sid]
    score_df = pd.concat(ordered_score_cols, axis=1, sort=False) if ordered_score_cols else pd.DataFrame()
    if rank_df.empty and tasks:
        warnings.warn(
            "multiscale_moran_ranks produced an empty table (no scale returned non-empty MoranI). "
            "Try `moran_thresh=-1`, set `quiet=False` to see warnings, or run a single scale sequentially to debug."
        )
    rank_df.to_csv(out_csv)
    if out_score_csv is not None:
        score_df.to_csv(out_score_csv)
    if return_scores:
        return rank_df, score_df
    return rank_df


def compare_rank_tables(
    baseline_rank_df: pd.DataFrame,
    adjusted_rank_df: pd.DataFrame,
    *,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Summarize how much SVG ranks shift between two rank tables, per shared scale column.

    Parameters
    ----------
    baseline_rank_df
        Reference rank table, typically the unadjusted `multiscale_moran_ranks` output.
    adjusted_rank_df
        Comparison rank table, typically produced with `size_residualize`.
    top_n
        Top-ranked genes used for overlap summaries.
    """

    if int(top_n) < 1:
        raise ValueError("top_n must be >= 1")

    shared_cols = [c for c in baseline_rank_df.columns if c in adjusted_rank_df.columns]
    rows: List[Dict[str, Any]] = []
    for col in shared_cols:
        base = pd.to_numeric(baseline_rank_df[col], errors="coerce")
        adj = pd.to_numeric(adjusted_rank_df[col], errors="coerce")
        common = base.notna() & adj.notna()
        base_valid = base.dropna()
        adj_valid = adj.dropna()
        base_top = set(base_valid.nsmallest(int(top_n)).index.astype(str))
        adj_top = set(adj_valid.nsmallest(int(top_n)).index.astype(str))
        overlap = len(base_top & adj_top)
        delta = (adj[common] - base[common]).astype(np.float64, copy=False)
        rows.append(
            {
                "scale_id": str(col)[len("rank_") :] if str(col).startswith("rank_") else str(col),
                "rank_column": str(col),
                "n_common_genes": int(common.sum()),
                "spearman_rho": float(base[common].corr(adj[common], method="spearman")) if int(common.sum()) >= 2 else np.nan,
                "median_abs_rank_delta": float(np.median(np.abs(delta))) if delta.size else np.nan,
                "mean_rank_delta": float(delta.mean()) if delta.size else np.nan,
                "max_abs_rank_delta": float(np.max(np.abs(delta))) if delta.size else np.nan,
                "top_n": int(top_n),
                "top_n_overlap": int(overlap),
                "top_n_overlap_fraction": float(overlap / max(1, min(len(base_top), len(adj_top), int(top_n)))),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["scale_id", "rank_column"]).reset_index(drop=True)
    return out


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


def _format_observed_count_label(value: float, *, suffix: str = "K") -> str:
    if not np.isfinite(float(value)):
        return "NA"
    n = int(round(float(value)))
    if n >= 999_500:
        label = f"{n / 1_000_000.0:.1f}".rstrip("0").rstrip(".")
        return f"{label}M"
    if n >= 1000:
        return f"{int(round(n / 1000.0))}{suffix}"
    return str(n)


def _coerce_bool_series_value(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(value)
    except Exception:
        return False


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicate column labels by taking the first non-null value."""
    if df.columns.is_unique:
        return df
    pieces = []
    seen = set()
    columns = pd.Index(df.columns)
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        positions = np.flatnonzero(columns == col)
        block = df.iloc[:, positions]
        if block.shape[1] == 1:
            series = block.iloc[:, 0]
        else:
            series = block.bfill(axis=1).iloc[:, 0]
        series = series.copy()
        series.name = col
        pieces.append(series)
    return pd.concat(pieces, axis=1)


def make_multiscale_resolution_label_map(
    segments_index: str | pd.DataFrame,
    *,
    prefix: str = "rank_",
    col_prefix: str = "res_",
    observed_col: str = "observed_obs_n_segments",
    native_label: str | None = None,
    count_suffix: str = "K",
    include_scale_ids: bool = True,
    include_rank_columns: bool = True,
    include_resolution_columns: bool = True,
    atol: float = 1e-4,
) -> dict[str, str]:
    """
    Build display labels for multiscale rank/trajectory columns from ``segments_index.csv``.

    Labels are based on the actual observed segment count, so float native
    resolutions such as ``pitch_um`` do not need brittle manual string matching.
    """
    index_df = pd.read_csv(segments_index) if isinstance(segments_index, (str, os.PathLike)) else segments_index.copy()
    required = {"scale_id", "resolution"}
    if not required.issubset(index_df.columns):
        raise KeyError("segments_index must contain columns: 'scale_id', 'resolution'")

    if observed_col not in index_df.columns:
        raise KeyError(f"segments_index must contain '{observed_col}' to label by observed segment count")

    label_map: dict[str, str] = {}
    res_vals = pd.to_numeric(index_df["resolution"], errors="coerce")
    obs_vals = pd.to_numeric(index_df[observed_col], errors="coerce")

    for i, row in index_df.reset_index(drop=True).iterrows():
        res = float(res_vals.iloc[i])
        if not np.isfinite(res):
            continue
        sid = str(row["scale_id"])
        is_native = _coerce_bool_series_value(row.get("native_identity", False)) or str(row.get("mode", "")).strip().lower() == "native_identity"
        obs = float(obs_vals.iloc[i])
        label = str(native_label) if (is_native and native_label is not None) else _format_observed_count_label(obs, suffix=count_suffix)

        if include_scale_ids:
            label_map[sid] = label
        if include_rank_columns:
            label_map[f"{prefix}{sid}"] = label
        if include_resolution_columns:
            res_key = format(res, "g")
            label_map[f"{col_prefix}{res_key}"] = label
            label_map[res_key] = label
            for existing_res in res_vals.dropna().to_numpy(dtype=float):
                if np.isclose(res, float(existing_res), rtol=0, atol=float(atol)):
                    label_map[f"{col_prefix}{format(float(existing_res), 'g')}"] = label
                    label_map[format(float(existing_res), "g")] = label
    return label_map


def label_multiscale_resolution_columns(
    df: pd.DataFrame,
    segments_index: str | pd.DataFrame,
    *,
    prefix: str = "rank_",
    col_prefix: str = "res_",
    observed_col: str = "observed_obs_n_segments",
    native_label: str | None = None,
    count_suffix: str = "K",
    inplace: bool = False,
) -> pd.DataFrame:
    """Rename rank/trajectory columns using labels derived from ``segments_index``."""
    label_map = make_multiscale_resolution_label_map(
        segments_index,
        prefix=prefix,
        col_prefix=col_prefix,
        observed_col=observed_col,
        native_label=native_label,
        count_suffix=count_suffix,
    )
    target = df if inplace else df.copy()
    target.columns = [label_map.get(str(c), label_map.get(str(c).replace(prefix, "", 1), str(c))) for c in target.columns]
    return target


def plot_multiscale_rank_trajectory(
    rank_trajectory: pd.DataFrame,
    genes: List[str],
    *,
    segments_index: str | pd.DataFrame | None = None,
    scale_groups: str | pd.DataFrame | None = None,
    show_scale_group_bar: bool | None = None,
    scale_group_col: str = "target_group",
    scale_group_palette: Optional[Dict[str, str]] = None,
    scale_group_bar_y: float = -0.095,
    scale_group_bar_height: float = 0.045,
    scale_group_label_y: float | None = None,
    scale_group_label_inside: bool = True,
    scale_group_xtick_pad: Optional[float] = 24,
    label_columns: bool = True,
    segment_label_mode: str = "observed_count",
    native_label: str | None = "Native",
    max_xtick_labels: int | None = None,
    xtick_label_every: int | None = None,
    xlabel: str = "Resolution",
    ylabel: str = "Rank",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 7),
    dpi: int = 300,
    color_palette: str = "tab20",
    linewidth: float = 2.5,
    markersize: float = 8,
    font_scale: float = 1.3,
    legend: bool = True,
    show_legend: bool | None = None,
    legend_ncol: int = 1,
    legend_outside: bool = True,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
    grid: bool = True,
    rotate_xticks: float = 45,
    missing: str = "warn",
):
    """Plot publication-style multiscale rank trajectories.

    If ``segments_index`` is provided, columns are relabeled from
    ``observed_obs_n_segments`` by default using integer count labels such as
    ``406K`` and ``475``. Set ``segment_label_mode="resolution"`` to label the
    x-axis by scale/resolution instead. Lower ranks are plotted higher by
    inverting the y-axis.  Pass ``scale_groups`` to draw a fine/mid/coarse band
    below the x-axis; set ``show_scale_group_bar=False`` to suppress it.
    """
    if show_legend is not None:
        legend = bool(show_legend)
    if show_scale_group_bar is None:
        show_scale_group_bar = scale_groups is not None
    if rank_trajectory is None or rank_trajectory.empty:
        raise ValueError("rank_trajectory must be a non-empty DataFrame.")

    plot_df = rank_trajectory.copy()
    original_columns = [str(c) for c in plot_df.columns]
    if bool(label_columns) and segments_index is not None:
        mode = str(segment_label_mode).lower()
        if mode in {"resolution", "scale", "scale_um", "um"}:
            index_df = pd.read_csv(segments_index) if isinstance(segments_index, (str, os.PathLike)) else segments_index.copy()
            required = {"scale_id", "resolution"}
            if not required.issubset(index_df.columns):
                raise KeyError("segments_index must contain columns: 'scale_id', 'resolution'")
            label_map: dict[str, str] = {}
            for _, row in index_df.iterrows():
                sid = str(row["scale_id"])
                res = pd.to_numeric(pd.Series([row.get("resolution", np.nan)]), errors="coerce").iloc[0]
                is_native = _coerce_bool_series_value(row.get("native_identity", False)) or str(row.get("mode", "")).strip().lower() == "native_identity"
                label = str(native_label) if is_native and native_label is not None else (f"{float(res):g}" if np.isfinite(res) else sid)
                label_map[sid] = label
                label_map[f"rank_{sid}"] = label
                label_map[f"res_{format(float(res), 'g')}"] = label if np.isfinite(res) else sid
                label_map[format(float(res), "g")] = label if np.isfinite(res) else sid
            plot_df.columns = [label_map.get(str(c), label_map.get(str(c).replace("rank_", "", 1), str(c))) for c in plot_df.columns]
        elif mode in {"observed_count", "count", "segments", "n_segments"}:
            plot_df = label_multiscale_resolution_columns(plot_df, segments_index, native_label=native_label)
        else:
            raise ValueError("segment_label_mode must be 'observed_count' or 'resolution'.")

    genes_use = [str(g) for g in genes if str(g) in plot_df.index]
    missing_genes = [str(g) for g in genes if str(g) not in plot_df.index]
    if missing_genes:
        msg = f"{len(missing_genes)} requested genes are missing from rank_trajectory: {missing_genes[:10]}"
        if str(missing).lower() in {"raise", "error"}:
            raise KeyError(msg)
        if str(missing).lower() not in {"ignore", "none", "skip"}:
            warnings.warn(msg)
    if not genes_use:
        raise ValueError("None of the requested genes were found in rank_trajectory.index.")

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    try:
        import seaborn as sns  # type: ignore
        colors = sns.color_palette(color_palette, n_colors=len(genes_use))
    except Exception:
        cmap = plt.get_cmap(color_palette if isinstance(color_palette, str) else "tab20")
        colors = [cmap(i / max(1, len(genes_use) - 1)) for i in range(len(genes_use))]

    local_rc = {
        "font.size": 14 * float(font_scale),
        "axes.labelsize": 16 * float(font_scale),
        "axes.titlesize": 18 * float(font_scale),
        "xtick.labelsize": 14 * float(font_scale),
        "ytick.labelsize": 14 * float(font_scale),
        "legend.fontsize": 12 * float(font_scale),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    }

    with mpl.rc_context(local_rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
        x = np.arange(plot_df.shape[1])

        for idx, gene in enumerate(genes_use):
            y = pd.to_numeric(plot_df.loc[gene], errors="coerce").to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                "-o",
                linewidth=float(linewidth),
                markersize=float(markersize),
                alpha=0.85,
                label=gene,
                color=colors[idx],
                markeredgewidth=1.2,
                markeredgecolor="white",
            )

        ax.invert_yaxis()
        labels = [str(c) for c in plot_df.columns]
        if xtick_label_every is not None:
            every = max(1, int(xtick_label_every))
            tick_idx = list(range(0, len(labels), every))
            if (len(labels) - 1) not in tick_idx:
                tick_idx.append(len(labels) - 1)
        elif max_xtick_labels is not None and len(labels) > int(max_xtick_labels):
            max_labels = max(2, int(max_xtick_labels))
            step = int(np.ceil(len(labels) / float(max_labels)))
            tick_idx = list(range(0, len(labels), step))
            if (len(labels) - 1) not in tick_idx:
                tick_idx.append(len(labels) - 1)
        else:
            tick_idx = list(range(len(labels)))
        ax.set_xticks(np.asarray(tick_idx, dtype=int))
        ax.set_xticklabels([labels[i] for i in tick_idx], rotation=float(rotate_xticks), ha="right")
        if bool(show_scale_group_bar) and scale_group_xtick_pad is not None:
            ax.tick_params(axis="x", pad=float(scale_group_xtick_pad))
        ax.set_xlabel(str(xlabel), fontweight="bold")
        if bool(show_scale_group_bar):
            ax.xaxis.set_label_coords(0.5, -0.34)
        ax.set_ylabel(str(ylabel), fontweight="bold")
        if title is not None:
            ax.set_title(str(title), fontweight="bold", pad=20)

        if bool(show_scale_group_bar):
            if scale_groups is None:
                raise ValueError("scale_groups must be provided when show_scale_group_bar=True.")
            group_df = pd.read_csv(scale_groups) if isinstance(scale_groups, (str, os.PathLike)) else scale_groups.copy()
            if "scale_id" not in group_df.columns or scale_group_col not in group_df.columns:
                raise KeyError(f"scale_groups must contain 'scale_id' and {scale_group_col!r}.")
            group_lookup = {
                str(row["scale_id"]): str(row[scale_group_col])
                for _, row in group_df.dropna(subset=["scale_id", scale_group_col]).iterrows()
            }
            group_lookup.update({f"rank_{k}": v for k, v in list(group_lookup.items())})
            if "resolution" in group_df.columns:
                for _, row in group_df.dropna(subset=["resolution", scale_group_col]).iterrows():
                    res = pd.to_numeric(pd.Series([row["resolution"]]), errors="coerce").iloc[0]
                    if np.isfinite(res):
                        group_lookup[f"res_{format(float(res), 'g')}"] = str(row[scale_group_col])
                        group_lookup[format(float(res), "g")] = str(row[scale_group_col])

            label_to_group: dict[str, str] = {}
            if segments_index is not None:
                try:
                    observed_label_map = make_multiscale_resolution_label_map(
                        segments_index,
                        native_label=native_label,
                    )
                    for key, label in observed_label_map.items():
                        if str(key) in group_lookup:
                            label_to_group[str(label)] = group_lookup[str(key)]
                except Exception:
                    observed_label_map = {}
                try:
                    index_df_for_resolution = pd.read_csv(segments_index) if isinstance(segments_index, (str, os.PathLike)) else segments_index.copy()
                    if {"scale_id", "resolution"}.issubset(index_df_for_resolution.columns):
                        for _, row in index_df_for_resolution.iterrows():
                            sid = str(row["scale_id"])
                            if sid not in group_lookup:
                                continue
                            res = pd.to_numeric(pd.Series([row.get("resolution", np.nan)]), errors="coerce").iloc[0]
                            is_native = _coerce_bool_series_value(row.get("native_identity", False)) or str(row.get("mode", "")).strip().lower() == "native_identity"
                            label = str(native_label) if is_native and native_label is not None else (f"{float(res):g}" if np.isfinite(res) else sid)
                            label_to_group[str(label)] = group_lookup[sid]
                except Exception:
                    pass
            if scale_group_palette is None:
                scale_group_palette = {
                    "fine_hclust": "#4C78A8",
                    "mid_hclust": "#F58518",
                    "coarse_hclust": "#54A24B",
                    "high_resolution": "#4C78A8",
                    "mid_resolution": "#F58518",
                    "low_resolution": "#54A24B",
                }
            fallback = plt.get_cmap("tab10")

            def _scale_id_from_column(col: str) -> str:
                c = str(col)
                if c in group_lookup:
                    return c
                if c.startswith("rank_") and c[5:] in group_lookup:
                    return c[5:]
                if c in label_to_group:
                    return c
                return c

            group_labels = []
            for c in original_columns:
                key = _scale_id_from_column(c)
                group_labels.append(group_lookup.get(key, label_to_group.get(str(c))))
            present_groups = [g for g in group_labels if g is not None]
            if not present_groups:
                group_values = group_df[scale_group_col].dropna().astype(str).tolist()
                if len(group_values) == len(original_columns):
                    group_labels = group_values
                    present_groups = group_values
                else:
                    warnings.warn(
                        "No scale-group labels matched trajectory columns. "
                        "Check that scale_groups['scale_id'] matches rank_trajectory columns "
                        "or pass scale_groups with the same row order as rank_trajectory columns.",
                        stacklevel=2,
                    )
            dynamic_colors: dict[str, Any] = {}
            for i, group in enumerate(dict.fromkeys(present_groups)):
                dynamic_colors[group] = scale_group_palette.get(group, fallback(i % 10))

            trans = ax.get_xaxis_transform()
            runs: list[tuple[int, int, str]] = []
            start: Optional[int] = None
            current: Optional[str] = None
            for i, group in enumerate(group_labels + [None]):
                if group != current:
                    if current is not None and start is not None:
                        runs.append((start, i - 1, current))
                    start = i if group is not None else None
                    current = group
            for start_i, end_i, group in runs:
                color = dynamic_colors.get(group, "#BDBDBD")
                ax.add_patch(
                    Rectangle(
                        (start_i - 0.5, float(scale_group_bar_y)),
                        (end_i - start_i + 1),
                        float(scale_group_bar_height),
                        transform=trans,
                        clip_on=False,
                        linewidth=0,
                        facecolor=color,
                        alpha=0.95,
                    )
                )
                center = (start_i + end_i) / 2.0
                label_y = (
                    float(scale_group_bar_y) + float(scale_group_bar_height) / 2.0
                    if bool(scale_group_label_inside)
                    else (float(scale_group_label_y) if scale_group_label_y is not None else float(scale_group_bar_y) - 0.045)
                )
                ax.text(
                    center,
                    label_y,
                    str(group).replace("_hclust", "").replace("_resolution", ""),
                    transform=trans,
                    ha="center",
                    va="center" if bool(scale_group_label_inside) else "top",
                    fontsize=8 * float(font_scale),
                    color="white" if bool(scale_group_label_inside) else color,
                    fontweight="bold",
                    clip_on=False,
                )

        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
            ax.set_axisbelow(True)

        if legend:
            if legend_outside:
                leg = ax.legend(
                    bbox_to_anchor=legend_bbox_to_anchor,
                    loc=legend_loc,
                    ncol=int(legend_ncol),
                    frameon=True,
                    fancybox=False,
                    shadow=False,
                    edgecolor="black",
                    framealpha=1,
                )
                bottom = 0.38 if bool(show_scale_group_bar) else 0.22
                fig.subplots_adjust(right=0.72, bottom=bottom)
            else:
                leg = ax.legend(ncol=int(legend_ncol), frameon=True, edgecolor="black", framealpha=1)
                if bool(show_scale_group_bar):
                    fig.subplots_adjust(bottom=0.38)
                else:
                    fig.tight_layout()
            leg.get_frame().set_linewidth(1.5)
        else:
            if bool(show_scale_group_bar):
                fig.subplots_adjust(bottom=0.38)
            else:
                fig.tight_layout()

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    return fig, ax


def plot_traj_publication(
    rank_trajectory: pd.DataFrame,
    genes: List[str],
    xlabel: str = "Resolution",
    title: str | None = None,
    **kwargs,
):
    """Backward-compatible alias for ``plot_multiscale_rank_trajectory``."""
    return plot_multiscale_rank_trajectory(
        rank_trajectory,
        genes,
        xlabel=xlabel,
        title=title,
        **kwargs,
    )


def plot_multiscale_score_trajectory(
    score_trajectory: pd.DataFrame,
    genes: List[str],
    *,
    segments_index: str | pd.DataFrame | None = None,
    label_columns: bool = True,
    segment_label_mode: str = "observed_count",
    native_label: str | None = "Native",
    max_xtick_labels: int | None = None,
    xtick_label_every: int | None = None,
    prefix: str = "I_",
    col_prefix: str = "res_",
    xlabel: str = "Resolution",
    ylabel: str = "Moran's I",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 7),
    dpi: int = 300,
    color_palette: str = "tab20",
    linewidth: float = 2.5,
    markersize: float = 8,
    font_scale: float = 1.3,
    legend: bool = True,
    show_legend: bool | None = None,
    legend_ncol: int = 1,
    legend_outside: bool = True,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
    grid: bool = True,
    rotate_xticks: float = 45,
    ylim: tuple[float, float] | None = None,
    zero_line: bool = True,
    missing: str = "warn",
):
    """Plot publication-style multiscale Moran/score trajectories.

    This is the score/effect-size companion to
    :func:`plot_multiscale_rank_trajectory`. It does not invert the y-axis:
    higher score is plotted higher. If ``segments_index`` is provided, columns
    are relabeled from ``observed_obs_n_segments`` using integer count labels.
    Set ``segment_label_mode="resolution"`` to label by scale/resolution
    instead.
    """
    if show_legend is not None:
        legend = bool(show_legend)
    if score_trajectory is None or score_trajectory.empty:
        raise ValueError("score_trajectory must be a non-empty DataFrame.")

    plot_df = score_trajectory.copy()
    if bool(label_columns) and segments_index is not None:
        mode = str(segment_label_mode).lower()
        if mode in {"resolution", "scale", "scale_um", "um"}:
            index_df = pd.read_csv(segments_index) if isinstance(segments_index, (str, os.PathLike)) else segments_index.copy()
            required = {"scale_id", "resolution"}
            if not required.issubset(index_df.columns):
                raise KeyError("segments_index must contain columns: 'scale_id', 'resolution'")
            label_map: dict[str, str] = {}
            for _, row in index_df.iterrows():
                sid = str(row["scale_id"])
                res = pd.to_numeric(pd.Series([row.get("resolution", np.nan)]), errors="coerce").iloc[0]
                is_native = _coerce_bool_series_value(row.get("native_identity", False)) or str(row.get("mode", "")).strip().lower() == "native_identity"
                label = str(native_label) if is_native and native_label is not None else (f"{float(res):g}" if np.isfinite(res) else sid)
                label_map[sid] = label
                label_map[f"{prefix}{sid}"] = label
                label_map[f"{col_prefix}{format(float(res), 'g')}"] = label if np.isfinite(res) else sid
                label_map[format(float(res), "g")] = label if np.isfinite(res) else sid
            plot_df.columns = [label_map.get(str(c), label_map.get(str(c).replace(prefix, "", 1), str(c))) for c in plot_df.columns]
        elif mode in {"observed_count", "count", "segments", "n_segments"}:
            plot_df = label_multiscale_resolution_columns(
                plot_df,
                segments_index,
                prefix=prefix,
                col_prefix=col_prefix,
                native_label=native_label,
            )
        else:
            raise ValueError("segment_label_mode must be 'observed_count' or 'resolution'.")

    genes_use = [str(g) for g in genes if str(g) in plot_df.index]
    missing_genes = [str(g) for g in genes if str(g) not in plot_df.index]
    if missing_genes:
        msg = f"{len(missing_genes)} requested genes are missing from score_trajectory: {missing_genes[:10]}"
        if str(missing).lower() in {"raise", "error"}:
            raise KeyError(msg)
        if str(missing).lower() not in {"ignore", "none", "skip"}:
            warnings.warn(msg)
    if not genes_use:
        raise ValueError("None of the requested genes were found in score_trajectory.index.")

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore
        colors = sns.color_palette(color_palette, n_colors=len(genes_use))
    except Exception:
        cmap = plt.get_cmap(color_palette if isinstance(color_palette, str) else "tab20")
        colors = [cmap(i / max(1, len(genes_use) - 1)) for i in range(len(genes_use))]

    local_rc = {
        "font.size": 14 * float(font_scale),
        "axes.labelsize": 16 * float(font_scale),
        "axes.titlesize": 18 * float(font_scale),
        "xtick.labelsize": 14 * float(font_scale),
        "ytick.labelsize": 14 * float(font_scale),
        "legend.fontsize": 12 * float(font_scale),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    }

    with mpl.rc_context(local_rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
        x = np.arange(plot_df.shape[1])

        for idx, gene in enumerate(genes_use):
            y = pd.to_numeric(plot_df.loc[gene], errors="coerce").to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                "-o",
                linewidth=float(linewidth),
                markersize=float(markersize),
                alpha=0.85,
                label=gene,
                color=colors[idx],
                markeredgewidth=1.2,
                markeredgecolor="white",
            )

        if zero_line:
            ax.axhline(0, color="black", linewidth=1.0, alpha=0.35, zorder=0)
        if ylim is not None:
            ax.set_ylim(*ylim)

        labels = [str(c) for c in plot_df.columns]
        if xtick_label_every is not None:
            every = max(1, int(xtick_label_every))
            tick_idx = list(range(0, len(labels), every))
            if (len(labels) - 1) not in tick_idx:
                tick_idx.append(len(labels) - 1)
        elif max_xtick_labels is not None and len(labels) > int(max_xtick_labels):
            max_labels = max(2, int(max_xtick_labels))
            step = int(np.ceil(len(labels) / float(max_labels)))
            tick_idx = list(range(0, len(labels), step))
            if (len(labels) - 1) not in tick_idx:
                tick_idx.append(len(labels) - 1)
        else:
            tick_idx = list(range(len(labels)))
        ax.set_xticks(np.asarray(tick_idx, dtype=int))
        ax.set_xticklabels([labels[i] for i in tick_idx], rotation=float(rotate_xticks), ha="right")
        ax.set_xlabel(str(xlabel), fontweight="bold")
        ax.set_ylabel(str(ylabel), fontweight="bold")
        if title is not None:
            ax.set_title(str(title), fontweight="bold", pad=20)

        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
            ax.set_axisbelow(True)

        if legend:
            if legend_outside:
                leg = ax.legend(
                    bbox_to_anchor=legend_bbox_to_anchor,
                    loc=legend_loc,
                    ncol=int(legend_ncol),
                    frameon=True,
                    fancybox=False,
                    shadow=False,
                    edgecolor="black",
                    framealpha=1,
                )
                fig.subplots_adjust(right=0.72, bottom=0.22)
            else:
                leg = ax.legend(ncol=int(legend_ncol), frameon=True, edgecolor="black", framealpha=1)
                fig.tight_layout()
            leg.get_frame().set_linewidth(1.5)
        else:
            fig.tight_layout()

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    return fig, ax


def plot_traj_publication_score(
    score_trajectory: pd.DataFrame,
    genes: List[str],
    xlabel: str = "Resolution",
    title: str | None = None,
    **kwargs,
):
    """Backward-compatible score-trajectory alias for publication plots."""
    return plot_multiscale_score_trajectory(
        score_trajectory,
        genes,
        xlabel=xlabel,
        title=title,
        **kwargs,
    )


def _resolve_region_cols_for_table(
    table: pd.DataFrame,
    region_cols: Dict[str, List[Any]],
    *,
    atol: float = 1e-4,
) -> Dict[str, List[str]]:
    def _numeric_token(value: Any) -> float | None:
        text = str(value).strip()
        for prefix in ("rank_", "res_", "I_", "C_"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        try:
            return float(text)
        except Exception:
            return None

    col_lookup = {str(c): str(c) for c in table.columns}
    numeric_cols: list[tuple[str, float]] = []
    for col in table.columns:
        val = _numeric_token(col)
        if val is not None and np.isfinite(val):
            numeric_cols.append((str(col), float(val)))

    resolved: Dict[str, List[str]] = {}
    for region, cols in region_cols.items():
        out_cols: list[str] = []
        for col in cols:
            col_str = str(col)
            if col_str in col_lookup:
                out_cols.append(col_lookup[col_str])
                continue
            for prefix in ("res_", "rank_", "I_", "C_"):
                prefixed = f"{prefix}{col_str}"
                if prefixed in col_lookup:
                    out_cols.append(col_lookup[prefixed])
                    break
            else:
                val = _numeric_token(col)
                if val is None:
                    continue
                matches = [name for name, col_val in numeric_cols if np.isclose(col_val, val, rtol=0, atol=float(atol))]
                out_cols.extend(matches)
        seen = set()
        resolved[str(region)] = [c for c in out_cols if not (c in seen or seen.add(c))]
    return resolved


def ranks_to_resolution_trajectory(
    rank_df: pd.DataFrame,
    segments_index_csv: str,
    fillna_worst: bool = True,
    prefix: str = "rank_",
    col_prefix: str = "res_",
    label_by_observed: bool = False,
    observed_col: str = "observed_obs_n_segments",
    native_label: str | None = None,
    count_suffix: str = "K",
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
    label_map = (
        make_multiscale_resolution_label_map(
            index_df,
            prefix=prefix,
            col_prefix=col_prefix,
            observed_col=observed_col,
            native_label=native_label,
            count_suffix=count_suffix,
        )
        if bool(label_by_observed)
        else {}
    )
    for res in sorted(tmp["resolution"].unique()):
        cols = tmp.loc[tmp["resolution"] == res, "col"].tolist()
        if not cols:
            continue
        default_label = f"{col_prefix}{format(float(res), 'g')}"
        label = label_map.get(default_label, default_label)
        res_rank[label] = rank_df[cols].mean(axis=1)
    return res_rank


def scores_to_resolution_trajectory(
    score_df: pd.DataFrame,
    segments_index_csv: str,
    fillna: Optional[float] = None,
    prefix: str = "I_",
    col_prefix: str = "res_",
    label_by_observed: bool = False,
    observed_col: str = "observed_obs_n_segments",
    native_label: str | None = None,
    count_suffix: str = "K",
) -> pd.DataFrame:
    """
    Convert per-scale autocorrelation score table to a per-resolution trajectory table.

    This mirrors :func:`ranks_to_resolution_trajectory`, but operates on raw score
    columns such as ``I_<scale_id>`` from ``multiscale_moran_ranks(...,
    return_scores=True)``. For multiple compactness candidates at the same
    resolution, scores are averaged per gene.
    """
    if score_df is None or len(score_df) == 0:
        return pd.DataFrame(index=getattr(score_df, "index", None))

    work = score_df.copy()
    if fillna is not None:
        work = work.fillna(float(fillna))

    index_df = pd.read_csv(segments_index_csv)
    if "scale_id" not in index_df.columns or "resolution" not in index_df.columns:
        raise KeyError("segments_index.csv must contain columns: 'scale_id', 'resolution'")

    res_vals = pd.to_numeric(index_df["resolution"], errors="coerce")
    if res_vals.isna().any():
        raise ValueError("segments_index.csv 'resolution' must be numeric")

    scale_ids = [str(s) for s in index_df["scale_id"].tolist()]
    col_names = [f"{prefix}{sid}" for sid in scale_ids]
    existing = [c for c in col_names if c in work.columns]
    if not existing:
        return pd.DataFrame(index=work.index)

    tmp = pd.DataFrame({"scale_id": scale_ids, "resolution": res_vals.to_numpy()})
    tmp["col"] = col_names
    tmp = tmp[tmp["col"].isin(work.columns)]

    label_map = (
        make_multiscale_resolution_label_map(
            index_df,
            prefix=prefix,
            col_prefix=col_prefix,
            observed_col=observed_col,
            native_label=native_label,
            count_suffix=count_suffix,
        )
        if bool(label_by_observed)
        else {}
    )

    res_score = pd.DataFrame(index=work.index)
    for res in sorted(tmp["resolution"].unique()):
        cols = tmp.loc[tmp["resolution"] == res, "col"].tolist()
        if not cols:
            continue
        default_label = f"{col_prefix}{format(float(res), 'g')}"
        label = label_map.get(default_label, default_label)
        res_score[label] = work[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return res_score


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


def scores_to_compactness_trajectory(
    score_df: pd.DataFrame,
    segments_index_csv: str,
    fillna: Optional[float] = None,
    prefix: str = "I_",
    col_prefix: str = "comp_",
) -> pd.DataFrame:
    """
    Convert per-scale autocorrelation score table to a compactness trajectory table.

    Use this for score tables such as ``I_<scale_id>`` from
    ``multiscale_moran_ranks(..., return_scores=True)``. This is the score
    counterpart to ``ranks_to_compactness_trajectory``.
    """
    if score_df is None or len(score_df) == 0:
        return pd.DataFrame(index=getattr(score_df, "index", None))

    work = score_df.copy()
    if fillna is not None:
        work = work.fillna(float(fillna))

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
    tmp = tmp[tmp["col"].isin(work.columns)]

    comp_score = pd.DataFrame(index=work.index)
    for comp in sorted(tmp["compactness"].unique()):
        cols = tmp.loc[tmp["compactness"] == comp, "col"].tolist()
        if not cols:
            continue
        label = f"{col_prefix}{format(float(comp), 'g')}"
        comp_score[label] = work[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return comp_score


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
    region_cols: Dict[str, List[Any]],
    fillna_worst: bool = True,
    resolve_region_cols: bool = True,
    region_col_atol: float = 1e-4,
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

    if bool(resolve_region_cols):
        region_cols = _resolve_region_cols_for_table(
            res_rank,
            region_cols,
            atol=float(region_col_atol),
        )
    else:
        region_cols = {
            str(region): [str(c) for c in cols if str(c) in res_rank.columns]
            for region, cols in region_cols.items()
        }
    missing = [region for region, cols in region_cols.items() if len(cols) == 0]
    if missing:
        raise ValueError(
            f"Missing columns for groups: {missing}. "
            f"Available columns: {list(map(str, res_rank.columns))}"
        )

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


def classify_scale_specific_scores_quantile(
    res_score: pd.DataFrame,
    *,
    region_cols: Dict[str, List[Any]],
    normalize_columns: bool = True,
    fillna_floor: bool = True,
    resolve_region_cols: bool = True,
    region_col_atol: float = 1e-4,
    q_gap_abs: float = 0.50,
    q_gap_rel: float = 0.50,
    q_best_mean: float = 0.95,
    q_best_std: float = 0.50,
    q_spec: float = 0.50,
) -> pd.DataFrame:
    """Classify scale-specific genes from resolution-score trajectories where higher is better.

    This is intended for score tables such as Moran's I trajectories. By default,
    each resolution column is min-max normalized to ``[0, 1]`` before classification
    so values remain comparable across scales.

    Parameters
    ----------
    res_score
        Gene x resolution-score table where larger values indicate stronger signal.
    region_cols
        Required mapping from region name -> resolution columns.
    normalize_columns
        If True, min-max normalize each resolution column independently before scoring.
    fillna_floor
        If True, fill missing values with the weakest normalized score (0 after normalization,
        otherwise the column minimum).
    q_gap_abs
        Quantile for ``gap_abs``; larger is stricter because larger region separation is better.
    q_gap_rel
        Quantile for ``gap_rel``; larger is stricter because larger relative separation is better.
    q_best_mean
        Quantile for ``best_mean``; larger is stricter because better regions have larger scores.
    q_best_std
        Quantile for ``best_std``; smaller is stricter because stable regions vary less.
    q_spec
        Quantile for ``specificity_score``; larger is stricter because more specific profiles score higher.
    """
    if res_score is None or len(res_score) == 0:
        return pd.DataFrame(index=getattr(res_score, "index", None))

    if region_cols is None:
        raise ValueError("region_cols must be provided explicitly.")

    if bool(resolve_region_cols):
        region_cols = _resolve_region_cols_for_table(
            res_score,
            region_cols,
            atol=float(region_col_atol),
        )
    else:
        region_cols = {
            str(region): [str(c) for c in cols if str(c) in res_score.columns]
            for region, cols in region_cols.items()
        }
    missing = [region for region, cols in region_cols.items() if len(cols) == 0]
    if missing:
        raise ValueError(
            f"Missing columns for groups: {missing}. "
            f"Available columns: {list(map(str, res_score.columns))}"
        )

    score_cols = [c for cols in region_cols.values() for c in cols]
    score_mat = res_score.loc[:, score_cols].apply(pd.to_numeric, errors="coerce")

    if normalize_columns:
        col_min = score_mat.min(axis=0, skipna=True)
        col_max = score_mat.max(axis=0, skipna=True)
        denom = (col_max - col_min).replace(0, np.nan)
        score_norm = score_mat.subtract(col_min, axis=1).divide(denom, axis=1)
    else:
        score_norm = score_mat.copy()

    if fillna_floor:
        if normalize_columns:
            score_norm = score_norm.fillna(0.0)
        else:
            col_floor = score_norm.min(axis=0, skipna=True)
            for c in score_norm.columns:
                floor = float(col_floor.get(c, np.nan))
                if not np.isfinite(floor):
                    floor = 0.0
                score_norm[c] = score_norm[c].fillna(floor)

    out = res_score.copy()

    for region, cols in region_cols.items():
        out[f"mean_{region}"] = score_norm[cols].mean(axis=1)
        out[f"std_{region}"] = score_norm[cols].std(axis=1, ddof=0)

    mean_tbl = pd.DataFrame({region: out[f"mean_{region}"] for region in region_cols})
    std_tbl = pd.DataFrame({region: out[f"std_{region}"] for region in region_cols})

    mean_arr = mean_tbl.to_numpy(dtype=np.float64, copy=False)
    sort_idx = np.argsort(-mean_arr, axis=1)
    best_idx = sort_idx[:, 0]
    second_idx = sort_idx[:, 1] if mean_arr.shape[1] > 1 else np.zeros(mean_arr.shape[0], dtype=np.int64)

    mean_cols = np.asarray(mean_tbl.columns, dtype=object)
    out["best_region"] = mean_cols[best_idx]
    out["best_mean"] = mean_arr[np.arange(len(out)), best_idx]
    out["second_best_mean"] = mean_arr[np.arange(len(out)), second_idx]
    out["gap_abs"] = out["best_mean"] - out["second_best_mean"]
    out["gap_rel"] = out["gap_abs"] / (np.abs(out["best_mean"]) + 1e-8)

    std_arr = std_tbl.to_numpy(dtype=np.float64, copy=False)
    out["best_std"] = std_arr[np.arange(len(out)), best_idx]
    out["specificity_score"] = out["gap_abs"] / (out["best_std"] + 1e-3)

    sp = score_norm.to_numpy(dtype=np.float64, copy=False)
    others_mean = (sp.sum(axis=1, keepdims=True) - sp) / max(1, sp.shape[1] - 1)
    res_score_gain = sp - others_mean
    best_res_idx = np.argmax(res_score_gain, axis=1)
    if res_score_gain.shape[1] > 1:
        second_res = np.partition(res_score_gain, kth=res_score_gain.shape[1] - 2, axis=1)[:, -2]
    else:
        second_res = np.zeros(res_score_gain.shape[0], dtype=np.float64)

    out["best_resolution"] = np.asarray(score_cols, dtype=object)[best_res_idx]
    out["best_resolution_score"] = res_score_gain[np.arange(res_score_gain.shape[0]), best_res_idx]
    out["best_resolution_margin"] = out["best_resolution_score"] - second_res

    min_gap_abs = float(out["gap_abs"].quantile(float(q_gap_abs)))
    min_gap_rel = float(out["gap_rel"].quantile(float(q_gap_rel)))
    min_best_mean = float(out["best_mean"].quantile(float(q_best_mean)))
    max_best_std = float(out["best_std"].quantile(float(q_best_std)))
    min_spec_score = float(out["specificity_score"].quantile(float(q_spec)))

    is_specific = (
        (out["gap_abs"] >= min_gap_abs) &
        (out["gap_rel"] >= min_gap_rel) &
        (out["best_mean"] >= min_best_mean) &
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
        "min_best_mean": min_best_mean,
        "max_best_std": max_best_std,
        "min_spec_score": min_spec_score,
        "region_cols": {k: list(v) for k, v in region_cols.items()},
        "normalize_columns": bool(normalize_columns),
        "fillna_floor": bool(fillna_floor),
    }
    return out


def classify_scale_specific_svg_rank(
    res_rank: pd.DataFrame,
    *,
    region_cols: Dict[str, List[Any]],
    fillna_worst: bool = True,
    resolve_region_cols: bool = True,
    region_col_atol: float = 1e-4,
    q_gap_abs: float = 0.50,
    q_gap_rel: float = 0.50,
    q_best_mean: float = 0.05,
    q_best_std: float = 0.50,
    q_spec: float = 0.50,
) -> pd.DataFrame:
    """
    Rank-based scale-specific SVG selection.

    This is a clear alias around ``classify_scale_specific_genes_quantile``:
    lower within-resolution ranks are better, and scale specificity is called
    from rank trajectory shape only. Moran's I scores are not used.
    """
    out = classify_scale_specific_genes_quantile(
        res_rank,
        region_cols=region_cols,
        fillna_worst=fillna_worst,
        resolve_region_cols=resolve_region_cols,
        region_col_atol=region_col_atol,
        q_gap_abs=q_gap_abs,
        q_gap_rel=q_gap_rel,
        q_best_mean=q_best_mean,
        q_best_std=q_best_std,
        q_spec=q_spec,
    )
    if not out.empty and "category" in out.columns:
        out["is_svg_rank"] = out["category"].astype(str) != "mixed"
    return out


def _scale_group_contains(group_spec: Any, scale_values: pd.Series) -> pd.Series:
    """Return a boolean mask for scale values belonging to one group spec."""
    numeric_scales = pd.to_numeric(scale_values, errors="coerce")
    if isinstance(group_spec, slice):
        start = -np.inf if group_spec.start is None else float(group_spec.start)
        stop = np.inf if group_spec.stop is None else float(group_spec.stop)
        return numeric_scales.ge(start) & numeric_scales.le(stop)

    if isinstance(group_spec, tuple) and len(group_spec) == 2:
        lo, hi = float(group_spec[0]), float(group_spec[1])
        return numeric_scales.ge(lo) & numeric_scales.le(hi)

    if np.isscalar(group_spec) or isinstance(group_spec, str):
        values = [group_spec]
    else:
        values = list(group_spec)
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric_values.notna().all():
        mask = pd.Series(False, index=scale_values.index)
        scale_arr = numeric_scales.to_numpy(dtype=float)
        for v in numeric_values.to_numpy(dtype=float):
            mask |= np.isclose(scale_arr, float(v), rtol=0, atol=1e-4)
        return mask
    return scale_values.astype(str).isin([str(v) for v in values])


def _assign_scale_groups(
    df: pd.DataFrame,
    *,
    scale_col: str,
    scale_groups: Dict[str, Any],
    out_col: str,
) -> pd.Series:
    labels = pd.Series(pd.NA, index=df.index, dtype="object")
    for group_name, group_spec in scale_groups.items():
        mask = _scale_group_contains(group_spec, df[scale_col])
        labels.loc[mask] = str(group_name)
    if labels.isna().any():
        missing = sorted(df.loc[labels.isna(), scale_col].astype(str).unique().tolist())
        raise ValueError(
            f"Some scales were not assigned to any scale group: {missing[:20]}. "
            "Pass scale_groups that cover all scales, or provide group_col."
        )
    return labels.rename(out_col)


def call_scale_specific_svgs(
    per_scale: pd.DataFrame,
    *,
    sample_col: Optional[str] = "sample",
    gene_col: str = "gene",
    scale_col: str = "scale",
    group_col: Optional[str] = "target_band",
    scale_groups: Optional[Dict[str, Any]] = None,
    group_order: Optional[List[str]] = None,
    rank_col: str = "moran_rank",
    n_genes_col: str = "n_genes_in_scale",
    moran_col: Optional[str] = "moran_I",
    qval_col: Optional[str] = "qval_bh",
    target_rank_pct_max: float = 0.10,
    min_gap_to_second_best: float = 0.10,
    min_target_good_fraction: float = 0.50,
    good_rank_pct_max: Optional[float] = None,
    qval_cutoff: float = 0.05,
    require_positive_moran: bool = True,
    min_groups: int = 3,
    return_all: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Call scale-preferential SVG candidates from a long per-scale Moran rank table.

    This is the simple notebook-facing SPIX rule:

    1. Convert each scale-level Moran rank to a within-scale rank percentile.
    2. Summarize median rank percentile within each scale group.
    3. Assign each gene to the group with the best median rank percentile.
    4. Keep the candidate only when the best group is strong, separated from the
       second-best group, and supported by enough individual target scales.

    The function intentionally does not use a custom score, Fisher test, or
    quantile-derived threshold. Lower rank percentile is always better.
    """
    if per_scale is None or per_scale.empty:
        empty = pd.DataFrame()
        return (empty, empty) if return_all else empty

    df = per_scale.copy()
    if gene_col not in df.columns:
        raise KeyError(f"Missing gene column: {gene_col}")
    if scale_col not in df.columns:
        raise KeyError(f"Missing scale column: {scale_col}")

    internal_sample_col = sample_col
    if sample_col is None or sample_col not in df.columns:
        internal_sample_col = "__sample__"
        df[internal_sample_col] = "sample"

    if rank_col not in df.columns:
        if moran_col is None or moran_col not in df.columns:
            raise KeyError(f"Missing {rank_col}; cannot compute ranks because {moran_col!r} is unavailable.")
        df[rank_col] = (
            df.groupby([internal_sample_col, scale_col], observed=True)[moran_col]
            .rank(method="min", ascending=False)
            .astype(int)
        )

    if n_genes_col not in df.columns:
        df[n_genes_col] = df.groupby([internal_sample_col, scale_col], observed=True)[gene_col].transform("size")

    if group_col is not None and group_col in df.columns:
        working_group_col = group_col
    else:
        if scale_groups is None:
            raise ValueError("Provide group_col in per_scale or pass scale_groups.")
        working_group_col = "__scale_group__"
        df[working_group_col] = _assign_scale_groups(
            df,
            scale_col=scale_col,
            scale_groups=scale_groups,
            out_col=working_group_col,
        )

    if group_order is None:
        if scale_groups is not None:
            group_order = [str(k) for k in scale_groups.keys()]
        else:
            group_order = [str(x) for x in pd.Series(df[working_group_col]).dropna().drop_duplicates().tolist()]
    else:
        group_order = [str(x) for x in group_order]

    good_rank_pct_max = float(target_rank_pct_max if good_rank_pct_max is None else good_rank_pct_max)
    df["rank_percentile"] = pd.to_numeric(df[rank_col], errors="coerce") / pd.to_numeric(df[n_genes_col], errors="coerce")
    df["_good_rank"] = df["rank_percentile"].le(good_rank_pct_max)
    if require_positive_moran and moran_col is not None and moran_col in df.columns:
        df["_good_rank"] &= pd.to_numeric(df[moran_col], errors="coerce").gt(0)
    if qval_col is not None and qval_col in df.columns:
        df["_good_rank"] &= pd.to_numeric(df[qval_col], errors="coerce").lt(float(qval_cutoff))

    rows = []
    for (sample_value, gene_value), sdf in df.groupby([internal_sample_col, gene_col], observed=True):
        band_stats: Dict[str, Dict[str, float]] = {}
        for group_name in group_order:
            gdf = sdf.loc[sdf[working_group_col].astype(str).eq(group_name)]
            if gdf.empty:
                continue
            stats = {
                "median_rank_percentile": float(gdf["rank_percentile"].median()),
                "best_rank_percentile": float(gdf["rank_percentile"].min()),
                "good_fraction": float(gdf["_good_rank"].mean()),
                "n_scales": int(len(gdf)),
            }
            if moran_col is not None and moran_col in gdf.columns:
                stats["median_moran"] = float(pd.to_numeric(gdf[moran_col], errors="coerce").median())
            band_stats[str(group_name)] = stats

        if len(band_stats) < int(min_groups):
            continue
        medians = {group_name: band_stats[group_name]["median_rank_percentile"] for group_name in group_order if group_name in band_stats}
        ordered = sorted(medians.items(), key=lambda kv: kv[1])
        if len(ordered) < 2:
            continue
        target_group, target_median = ordered[0]
        second_group, second_median = ordered[1]
        gap = float(second_median - target_median)
        target_good_fraction = float(band_stats[target_group]["good_fraction"])
        is_scale_specific = (
            target_median <= float(target_rank_pct_max)
            and gap >= float(min_gap_to_second_best)
            and target_good_fraction >= float(min_target_good_fraction)
        )

        row = {
            "sample": sample_value,
            "gene": gene_value,
            "target_group": target_group,
            "second_best_group": second_group,
            "target_median_rank_percentile": float(target_median),
            "second_best_median_rank_percentile": float(second_median),
            "gap_to_second_best": gap,
            "target_good_fraction": target_good_fraction,
            "scale_specific_svg": bool(is_scale_specific),
            "scale_preferential_svg_candidate": bool(is_scale_specific),
        }
        for group_name in group_order:
            if group_name not in band_stats:
                continue
            row[f"{group_name}_median_rank_percentile"] = band_stats[group_name]["median_rank_percentile"]
            row[f"{group_name}_best_rank_percentile"] = band_stats[group_name]["best_rank_percentile"]
            row[f"{group_name}_good_fraction"] = band_stats[group_name]["good_fraction"]
            row[f"{group_name}_n_scales"] = band_stats[group_name]["n_scales"]
            if "median_moran" in band_stats[group_name]:
                row[f"{group_name}_median_moran"] = band_stats[group_name]["median_moran"]
        rows.append(row)

    all_genes = pd.DataFrame(rows)
    if all_genes.empty:
        return (all_genes, all_genes) if return_all else all_genes

    sort_cols = ["sample", "target_group", "target_median_rank_percentile", "gap_to_second_best", "target_good_fraction"]
    all_genes = all_genes.sort_values(sort_cols, ascending=[True, True, True, False, False]).reset_index(drop=True)
    calls = all_genes.loc[all_genes["scale_specific_svg"]].copy().reset_index(drop=True)

    attrs = {
        "rule": {
            "target_rank_pct_max": float(target_rank_pct_max),
            "min_gap_to_second_best": float(min_gap_to_second_best),
            "min_target_good_fraction": float(min_target_good_fraction),
            "good_rank_pct_max": float(good_rank_pct_max),
            "qval_cutoff": float(qval_cutoff),
            "require_positive_moran": bool(require_positive_moran),
            "group_order": list(group_order),
        }
    }
    all_genes.attrs.update(attrs)
    calls.attrs.update(attrs)
    return (calls, all_genes) if return_all else calls


def _read_segments_index_like(segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]]) -> Optional[pd.DataFrame]:
    if segments_index is None:
        return None
    if isinstance(segments_index, pd.DataFrame):
        return segments_index.copy()
    return pd.read_csv(segments_index)


def _is_native_index_row(row: pd.Series) -> bool:
    for col in ("native_identity",):
        if col in row.index and _coerce_bool_series_value(row[col]):
            return True
    if "mode" in row.index and str(row["mode"]).strip().lower() == "native_identity":
        return True
    if "path" in row.index and str(row["path"]) == _NATIVE_IDENTITY_PATH:
        return True
    return False


def _infer_native_mask_from_scale_table(
    table: pd.DataFrame,
    *,
    scale_id_col: str = "scale_id",
    native_col: str = "native_identity",
) -> pd.Series:
    """Infer native identity robustly from scale metadata.

    Prefer explicit non-boolean metadata such as scale_id/mode/path. A stale
    table can occasionally carry native_identity=True for every scale after
    notebook merges; for multi-scale tables that pattern is not credible and is
    ignored unless there are no better hints and only one row exists.
    """
    if table is None or table.empty:
        return pd.Series(False, index=getattr(table, "index", None), dtype=bool)

    out = pd.Series(False, index=table.index, dtype=bool)
    if scale_id_col in table.columns:
        sid = table[scale_id_col].astype(str).str.strip().str.lower()
        out |= sid.isin({"native", "native_identity", "__native_identity__"})
        out |= sid.str.contains("native_identity", regex=False, na=False)
    if "mode" in table.columns:
        out |= table["mode"].astype(str).str.strip().str.lower().eq("native_identity")
    if "path" in table.columns:
        out |= table["path"].astype(str).eq(_NATIVE_IDENTITY_PATH)
    if out.any():
        return out

    if native_col in table.columns:
        native = table[native_col].map(_coerce_bool_series_value)
        if len(native) > 1 and bool(native.all()):
            return pd.Series(False, index=table.index, dtype=bool)
        return native.astype(bool)
    return out


def _find_scale_column(table: pd.DataFrame, sid: str, prefixes: Tuple[str, ...]) -> Optional[str]:
    sid = str(sid)
    candidates = [sid]
    candidates.extend([f"{prefix}{sid}" for prefix in prefixes])
    sid_no_prefix = sid[1:] if sid.startswith("r") else sid
    sid_no_res = sid[4:] if sid.startswith("res_") else sid_no_prefix
    candidates.extend([sid_no_prefix, f"r{sid_no_prefix}", sid_no_res, f"r{sid_no_res}", f"res_{sid_no_res}"])
    for prefix in prefixes:
        candidates.extend(
            [
                f"{prefix}{sid_no_prefix}",
                f"{prefix}r{sid_no_prefix}",
                f"{prefix}{sid_no_res}",
                f"{prefix}r{sid_no_res}",
                f"{prefix}res_{sid_no_res}",
            ]
        )
    candidates = list(dict.fromkeys(candidates))
    for col in candidates:
        if col in table.columns:
            return col
    return None


def infer_scale_groups_by_hclustering(
    rank_df: pd.DataFrame,
    segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]] = None,
    *,
    include_native: bool = True,
    top_rank_pct: float = 0.10,
    n_clusters: int = 3,
    distance_threshold: Optional[float] = None,
    linkage_method: str = "average",
    rank_prefixes: Tuple[str, ...] = ("rank_",),
    return_diagnostics: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Infer scale groups by hierarchical clustering of top-SVG overlap.

    Each scale is represented by the set of genes in the top ``top_rank_pct``
    of that scale's Moran rank list. Scale-to-scale similarity is the Jaccard
    overlap between those top-gene sets, and hierarchical clustering is run on
    ``1 - Jaccard``. Group names are assigned only after clustering by sorting
    clusters from fine to coarse resolution.
    """
    if rank_df is None or rank_df.empty:
        empty = pd.DataFrame()
        return (empty, {"jaccard": empty, "distance": empty, "linkage": None}) if return_diagnostics else empty
    if not (0 < float(top_rank_pct) <= 1):
        raise ValueError("top_rank_pct must be in (0, 1].")
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1.")

    rank = rank_df.copy()
    rank.index = rank.index.astype(str)
    index_df = _read_segments_index_like(segments_index)
    if index_df is None:
        scale_ids = []
        resolutions = []
        for col in rank.columns:
            sid = str(col)
            for prefix in rank_prefixes:
                sid = sid.replace(prefix, "", 1) if sid.startswith(prefix) else sid
            scale_ids.append(sid)
            try:
                resolutions.append(float(sid[1:] if sid.startswith("r") else sid))
            except Exception:
                resolutions.append(sid)
        index_df = pd.DataFrame({"scale_id": scale_ids, "resolution": resolutions})
    else:
        if "scale_id" not in index_df.columns:
            raise KeyError("segments_index must contain 'scale_id'.")
        if "resolution" not in index_df.columns:
            raise KeyError("segments_index must contain 'resolution'.")

    rows = []
    top_sets: Dict[str, set] = {}
    n_total_genes = int(rank.shape[0])
    for _, row in index_df.reset_index(drop=True).iterrows():
        sid = str(row["scale_id"])
        is_native = _is_native_index_row(row)
        if is_native and not include_native:
            continue
        rank_col_found = _find_scale_column(rank, sid, rank_prefixes)
        if rank_col_found is None:
            continue
        scale_value = row["resolution"]
        try:
            scale_value = float(scale_value)
        except Exception:
            pass
        ranks = pd.to_numeric(rank[rank_col_found], errors="coerce")
        rank_pct = ranks / float(n_total_genes)
        top_mask = rank_pct.le(float(top_rank_pct))
        top_genes = set(rank.index[top_mask.fillna(False)])
        rows.append(
            {
                "scale_id": sid,
                "scale": scale_value,
                "rank_col": rank_col_found,
                "native_identity": bool(is_native),
                "n_genes": n_total_genes,
                "n_ranked_genes": int(ranks.notna().sum()),
                "n_top_genes": int(len(top_genes)),
                "top_rank_pct": float(top_rank_pct),
            }
        )
        top_sets[sid] = top_genes

    group_table = pd.DataFrame(rows)
    if group_table.empty:
        empty = pd.DataFrame()
        return (group_table, {"jaccard": empty, "distance": empty, "linkage": None}) if return_diagnostics else group_table

    scale_ids = group_table["scale_id"].astype(str).tolist()
    n_scales = len(scale_ids)
    jaccard = pd.DataFrame(np.eye(n_scales), index=scale_ids, columns=scale_ids, dtype=float)
    for i, sid_i in enumerate(scale_ids):
        set_i = top_sets[sid_i]
        for j in range(i + 1, n_scales):
            sid_j = scale_ids[j]
            set_j = top_sets[sid_j]
            union = len(set_i | set_j)
            sim = 0.0 if union == 0 else len(set_i & set_j) / float(union)
            jaccard.iat[i, j] = sim
            jaccard.iat[j, i] = sim
    distance = 1.0 - jaccard
    np.fill_diagonal(distance.values, 0.0)

    linkage_matrix = None
    if n_scales == 1:
        labels = np.ones(n_scales, dtype=int)
    else:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(distance.to_numpy(dtype=float), checks=False)
        linkage_matrix = linkage(condensed, method=linkage_method)
        if distance_threshold is not None:
            labels = fcluster(linkage_matrix, t=float(distance_threshold), criterion="distance")
        else:
            labels = fcluster(linkage_matrix, t=min(int(n_clusters), n_scales), criterion="maxclust")

    group_table["hclust_cluster"] = labels.astype(int)
    numeric_scale = pd.to_numeric(group_table["scale"], errors="coerce")
    cluster_order = (
        group_table.assign(_numeric_scale=numeric_scale)
        .groupby("hclust_cluster", observed=True)["_numeric_scale"]
        .median()
        .sort_values(kind="mergesort")
        .index.tolist()
    )
    if len(cluster_order) == 3:
        ordered_names = ["fine_hclust", "mid_hclust", "coarse_hclust"]
    elif len(cluster_order) == 2:
        ordered_names = ["fine_hclust", "coarse_hclust"]
    elif len(cluster_order) == 1:
        ordered_names = ["scale_hclust_01"]
    else:
        ordered_names = [f"scale_hclust_{i:02d}" for i in range(1, len(cluster_order) + 1)]
    name_map = {int(cluster_id): ordered_names[i] for i, cluster_id in enumerate(cluster_order)}
    group_table["target_group"] = group_table["hclust_cluster"].map(name_map).astype(str)
    group_table["group_order"] = group_table["target_group"].map({name: i for i, name in enumerate(ordered_names)})
    group_table = group_table.sort_values(["group_order", "scale"], kind="mergesort").reset_index(drop=True)
    group_table = group_table.drop(columns=["group_order"])
    group_table.attrs["group_order"] = ordered_names
    group_table.attrs["method"] = {
        "method": "hclust_top_gene_jaccard",
        "top_rank_pct": float(top_rank_pct),
        "n_clusters": int(n_clusters),
        "distance_threshold": None if distance_threshold is None else float(distance_threshold),
        "linkage_method": str(linkage_method),
        "include_native": bool(include_native),
    }

    if not return_diagnostics:
        return group_table
    linkage_df = None
    if linkage_matrix is not None:
        linkage_df = pd.DataFrame(linkage_matrix, columns=["left", "right", "distance", "n_scales"])
    diagnostics = {
        "jaccard": jaccard,
        "distance": distance,
        "linkage": linkage_df,
        "top_sets": top_sets,
    }
    return group_table, diagnostics


def _scale_group_display_label(value: Any) -> str:
    label = str(value)
    replacements = {
        "fine_hclust": "Fine",
        "high_hclust": "High",
        "mid_hclust": "Mid",
        "coarse_hclust": "Coarse",
        "low_hclust": "Low",
    }
    return replacements.get(label, label.replace("_hclust", "").replace("_", " ").title())


def _scale_overlap_axis_label(
    row: pd.Series,
    *,
    scale_id_col: str,
    scale_col: str,
    native_col: str,
    observed_col: str,
    native_label: str,
    segment_label_mode: str,
    count_suffix: str,
) -> str:
    if native_col in row.index and bool(row[native_col]):
        return str(native_label)

    mode = str(segment_label_mode).lower()
    if mode in {"scale_id", "id"}:
        return str(row[scale_id_col])

    if mode in {"observed_count", "count", "segments", "n_segments", "seg", "n_seg", "n seg"}:
        if observed_col in row.index:
            observed = pd.to_numeric(pd.Series([row[observed_col]]), errors="coerce").iloc[0]
            if pd.notna(observed):
                return _format_observed_count_label(float(observed), suffix=count_suffix)

    raw = row[scale_col] if scale_col in row.index else row[scale_id_col]
    numeric = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        value = f"{float(numeric):g}"
    else:
        value = str(raw)
        if value.startswith("res_"):
            value = value.replace("res_", "", 1)
        elif value.startswith("r") and value[1:].replace(".", "", 1).isdigit():
            value = value[1:]

    if mode in {"res", "resolution", "scale"}:
        return f"res_{value}"
    return str(value)


def _merge_scale_group_segment_metadata(
    groups: pd.DataFrame,
    segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]],
    *,
    scale_id_col: str,
    observed_col: str,
    native_col: str,
) -> pd.DataFrame:
    groups = _coalesce_duplicate_columns(groups.copy())
    if segments_index is None:
        return groups
    index_df = _read_segments_index_like(segments_index)
    if index_df is not None:
        index_df = _coalesce_duplicate_columns(index_df)
    if index_df is None or index_df.empty or "scale_id" not in index_df.columns:
        return groups

    meta_cols = ["scale_id"]
    resolution_meta_cols = (
        "resolution",
        "resolution_um",
        "scale_um",
        "pitch_um",
        "bin_um",
        "bin_size_um",
        "requested_resolution",
        "requested_resolution_um",
        "spatial_resolution_um",
        "segmentation_resolution_um",
    )
    for col in (
        observed_col,
        "observed_tile_n_segments",
        "requested_n_segments",
        *resolution_meta_cols,
        native_col,
        "mode",
        "path",
    ):
        if col in index_df.columns and col not in meta_cols:
            meta_cols.append(col)
    meta = index_df[meta_cols].copy()
    meta["scale_id"] = meta["scale_id"].astype(str)
    out = groups.copy()
    out[scale_id_col] = out[scale_id_col].astype(str)
    out = out.merge(meta, left_on=scale_id_col, right_on="scale_id", how="left", suffixes=("", "_segments_index"))
    out = _coalesce_duplicate_columns(out)
    if "scale_id_segments_index" in out.columns:
        out = out.drop(columns=["scale_id_segments_index"])

    if f"{observed_col}_segments_index" in out.columns:
        if observed_col in out.columns:
            out[observed_col] = out[observed_col].where(out[observed_col].notna(), out[f"{observed_col}_segments_index"])
        else:
            out[observed_col] = out[f"{observed_col}_segments_index"]
    if observed_col not in out.columns:
        for fallback in ("observed_tile_n_segments", "requested_n_segments"):
            if fallback in out.columns:
                out[observed_col] = out[fallback]
                break
    if native_col not in out.columns and f"{native_col}_segments_index" in out.columns:
        out[native_col] = out[f"{native_col}_segments_index"]
    elif f"{native_col}_segments_index" in out.columns:
        current_raw = out[native_col]
        from_index_raw = out[f"{native_col}_segments_index"]
        current = current_raw.map(_coerce_bool_series_value)
        from_index = from_index_raw.map(_coerce_bool_series_value)
        # segments_index is the authoritative source for native identity.
        # This avoids stale scale_group tables with native_identity=True for
        # every row turning every scale into a Stereo-seq fine tier.
        out[native_col] = from_index.where(from_index_raw.notna(), current)
    out[native_col] = _infer_native_mask_from_scale_table(
        out,
        scale_id_col=scale_id_col,
        native_col=native_col,
    )
    return out


def _resolve_scale_overlap_cmap(cmap: Any):
    if str(cmap).lower() not in {"spix_overlap", "overlap", "svg_overlap", "jaccard_overlap"}:
        return cmap
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "spix_overlap",
        ["#f2f8e8", "#d6edc4", "#a9d99a", "#73c3a6", "#3f8fa1", "#123f66"],
        N=256,
    )


def apply_resolution_tier(
    scale_groups: pd.DataFrame,
    segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]] = None,
    *,
    high_resolutions: Tuple[float, ...] = (2.0, 8.0, 16.0),
    include_native_in_high: bool = True,
    source_group_col: str = "target_group",
    out_col: str = "stereoseq_resolution_tier",
    scale_id_col: str = "scale_id",
    scale_col: str = "scale",
    native_col: str = "native_identity",
    fine_group: str = "fine_hclust",
    exclusive_high: bool = False,
    exclusive_fallback_group: Optional[str] = None,
) -> pd.DataFrame:
    """Add an interpretation tier for selected high-resolution scales.

    Native or very fine physical resolutions can appear as singleton hclust
    groups because their top-SVG lists are ultra-fine and partly outlier-like.
    For interpretation, this helper keeps the original hclust columns and
    creates ``out_col`` where native plus selected fine resolutions are assigned
    to the fine/high tier. Other scales keep their original hclust group labels.
    """
    if scale_groups is None or scale_groups.empty:
        raise ValueError("scale_groups must be a non-empty DataFrame.")
    if scale_id_col not in scale_groups.columns:
        raise KeyError(f"scale_groups must contain {scale_id_col!r}.")

    work = scale_groups.copy()
    if source_group_col not in work.columns:
        raise KeyError(f"scale_groups must contain {source_group_col!r}.")
    work[out_col] = work[source_group_col].astype(str)

    work = _merge_scale_group_segment_metadata(
        work,
        segments_index,
        scale_id_col=scale_id_col,
        observed_col="observed_obs_n_segments",
        native_col=native_col,
    )
    if scale_col not in work.columns:
        if "resolution" in work.columns:
            work[scale_col] = work["resolution"]
        else:
            work[scale_col] = work[scale_id_col]

    resolution_candidate_cols = [
        scale_col,
        "resolution",
        "resolution_um",
        "scale_um",
        "bin_um",
        "bin_size_um",
        "requested_resolution",
        "requested_resolution_um",
        "spatial_resolution_um",
        "segmentation_resolution_um",
    ]
    resolution_arrays = []
    seen_resolution_cols = set()
    for col in resolution_candidate_cols:
        if col in work.columns and col not in seen_resolution_cols:
            seen_resolution_cols.add(col)
            values = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
            finite_unique = np.unique(values[np.isfinite(values)])
            if len(work) > 1 and finite_unique.size <= 1:
                continue
            resolution_arrays.append(values)
    scale_id_numeric = (
        work[scale_id_col]
        .astype(str)
        .str.replace("^res_", "", regex=True)
        .str.replace("^r", "", regex=True)
    )
    scale_id_numeric = pd.to_numeric(scale_id_numeric, errors="coerce")
    resolution_arrays.append(scale_id_numeric.to_numpy(dtype=float))
    high_mask = pd.Series(False, index=work.index)
    for resolution in high_resolutions:
        for values in resolution_arrays:
            high_mask |= np.isclose(values, float(resolution), rtol=0, atol=1e-4)
    if include_native_in_high and native_col in work.columns:
        high_mask |= _infer_native_mask_from_scale_table(
            work,
            scale_id_col=scale_id_col,
            native_col=native_col,
        )
    if bool(exclusive_high):
        fallback_group = exclusive_fallback_group
        if fallback_group is None:
            source_values = work.loc[~high_mask, source_group_col].astype(str).tolist()
            fallback_candidates = [g for g in source_values if g != str(fine_group)]
            if fallback_candidates:
                fallback_group = fallback_candidates[0]
        if fallback_group is None:
            raise ValueError(
                "exclusive_high=True could not infer a fallback group for non-selected "
                f"{fine_group!r} scales. Pass exclusive_fallback_group explicitly."
            )
        demote_mask = (~high_mask) & work[out_col].astype(str).eq(str(fine_group))
        work.loc[demote_mask, out_col] = str(fallback_group)
    work.loc[high_mask, out_col] = str(fine_group)

    group_order = []
    for group in work[out_col].astype(str).tolist():
        if group not in group_order:
            group_order.append(group)
    work.attrs["group_order"] = group_order
    work.attrs["resolution_tier"] = {
        "high_resolutions": tuple(float(x) for x in high_resolutions),
        "include_native_in_high": bool(include_native_in_high),
        "source_group_col": str(source_group_col),
        "out_col": str(out_col),
        "fine_group": str(fine_group),
        "exclusive_high": bool(exclusive_high),
        "exclusive_fallback_group": None if exclusive_fallback_group is None else str(exclusive_fallback_group),
    }
    return work


def apply_stereoseq_resolution_tier(
    scale_groups: pd.DataFrame,
    segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]] = None,
    *,
    high_resolutions: Tuple[float, ...] = (2.0, 8.0, 16.0),
    include_native_in_high: bool = True,
    source_group_col: str = "target_group",
    out_col: str = "stereoseq_resolution_tier",
    scale_id_col: str = "scale_id",
    scale_col: str = "scale",
    native_col: str = "native_identity",
    fine_group: str = "fine_hclust",
    exclusive_high: bool = False,
    exclusive_fallback_group: Optional[str] = None,
) -> pd.DataFrame:
    """Backward-compatible alias for :func:`apply_resolution_tier`."""
    out = apply_resolution_tier(
        scale_groups,
        segments_index,
        high_resolutions=high_resolutions,
        include_native_in_high=include_native_in_high,
        source_group_col=source_group_col,
        out_col=out_col,
        scale_id_col=scale_id_col,
        scale_col=scale_col,
        native_col=native_col,
        fine_group=fine_group,
        exclusive_high=exclusive_high,
        exclusive_fallback_group=exclusive_fallback_group,
    )
    out.attrs["stereoseq_resolution_tier"] = out.attrs.get("resolution_tier", {}).copy()
    return out


def plot_scale_svg_overlap_heatmap(
    jaccard: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
    scale_groups: Optional[pd.DataFrame] = None,
    *,
    rank_df: Optional[pd.DataFrame] = None,
    segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]] = None,
    include_native: bool = True,
    top_rank_pct: float = 0.10,
    n_clusters: int = 3,
    distance_threshold: Optional[float] = None,
    linkage_method: str = "average",
    resolution_tier: bool = False,
    resolution_tier_high_resolutions: Optional[Tuple[float, ...]] = None,
    resolution_tier_include_native: bool = True,
    resolution_tier_fine_group: str = "fine_hclust",
    resolution_tier_exclusive: bool = False,
    resolution_tier_fallback_group: Optional[str] = None,
    resolution_tier_mode: Optional[str] = None,
    resolution_tier_label: Optional[str] = None,
    resolution_tier_color: Optional[str] = None,
    resolution_tier_other_color: Optional[str] = None,
    resolution_tier_boundary_col: Optional[str] = None,
    stereoseq_resolution_tier: bool = False,
    stereoseq_high_resolutions: Tuple[float, ...] = (2.0, 8.0, 16.0),
    stereoseq_tier_mode: str = "scale_bar",
    stereoseq_tier_label: Optional[str] = None,
    stereoseq_tier_color: str = "#2ca02c",
    stereoseq_tier_other_color: str = "#f0f0f0",
    scale_id_col: str = "scale_id",
    scale_col: str = "scale",
    native_col: str = "native_identity",
    group_col: str = "target_group",
    scale_bar_group_col: Optional[str] = None,
    boundary_group_col: Optional[str] = None,
    scale_label: Optional[str] = None,
    segment_label_mode: str = "observed_count",
    observed_col: str = "observed_obs_n_segments",
    native_label: str = "Native",
    count_suffix: str = "K",
    native_position: str = "first",
    title: Optional[str] = None,
    similarity_label: str = "Jaccard similarity",
    group_legend_title: str = "H-clust group",
    group_palette: Optional[Dict[str, Any]] = None,
    cmap: str = "spix_overlap",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    group_strip_size: float = 0.035,
    colorbar_width: float = 0.035,
    legend_width: Optional[float] = None,
    colorbar_gap: float = 0.075,
    legend_gap: float = 0.11,
    legend_style: str = "inline",
    colorbar_label_side: str = "right",
    colorbar_tick_side: str = "right",
    show_xlabel: bool = False,
    show_ylabel: bool = False,
    base_fontsize: float = 16,
    tick_fontsize: Optional[float] = None,
    min_tick_fontsize: float = 10,
    title_fontsize: Optional[float] = None,
    title_pad: float = 0.030,
    title_y: Optional[float] = None,
    axis_label_fontsize: Optional[float] = None,
    colorbar_fontsize: Optional[float] = None,
    legend_fontsize: Optional[float] = None,
    legend_title_fontsize: Optional[float] = None,
    inline_legend_fontsize: Optional[float] = None,
    inline_legend_min_run: int = 2,
    inline_legend_color: str = "white",
    inline_legend_bbox: bool = False,
    inline_legend_y: float = 0.0,
    auto_shrink_tick_font: bool = True,
    xtick_rotation: float = 90,
    cell_grid: bool = True,
    grid_color: str = "white",
    grid_linewidth: float = 0.8,
    grid_linestyle: str = ":",
    group_boundary_lines: bool = True,
    group_boundary_color: str = "black",
    group_boundary_linewidth: float = 1.2,
    scale_boundary_lines: bool = False,
    scale_boundary_color: str = "black",
    scale_boundary_linewidth: float = 0.45,
    save: Optional[Union[str, os.PathLike]] = None,
    return_data: bool = False,
):
    """Plot scale-to-scale SVG overlap with H-clustering group annotations.

    Parameters
    ----------
    jaccard
        Square similarity matrix indexed and columned by ``scale_id``. A
        diagnostics dictionary from ``infer_scale_groups_by_hclustering`` or
        ``call_scale_specific_svgs_from_multiscale(..., return_diagnostics=True)``
        can also be passed.
    scale_groups
        Table containing ``scale_id``, ``scale``, ``native_identity``, and
        ``target_group``. When omitted with ``rank_df`` present, groups are
        inferred by hierarchical clustering.
    segment_label_mode
        Tick-label mode. The default, ``"observed_count"``, matches trajectory
        plotting and uses ``observed_obs_n_segments`` from ``segments_index``.
        Use ``"resolution"`` for resolution labels or ``"scale_id"`` for raw
        scale ids.
    base_fontsize
        Main font size used for tick labels when space allows. Tick labels are
        automatically shrunk to avoid overlap unless ``auto_shrink_tick_font``
        is False.
    legend_style
        ``"inline"`` labels scale groups directly on the top strip, which is
        the default paper-style display. Use ``"right"`` for a separate legend
        or ``"none"`` to hide group labels.
    rank_df
        Optional Moran-rank table used to compute the overlap matrix when
        ``jaccard`` is not provided.
    resolution_tier
        When True, display interpretation tiers by assigning native plus
        ``resolution_tier_high_resolutions`` to the fine/high group in the
        annotation strips. The similarity matrix itself is unchanged. The older
        ``stereoseq_resolution_tier`` name is retained as an alias.
    stereoseq_tier_mode
        ``"scale_bar"`` colors the annotation strips by the Stereo-seq tier
        while keeping black boundary lines at the original H-clustering
        transitions. ``"secondary_strip"`` adds a separate thin Stereo-seq
        strip, and ``"replace"`` uses the Stereo-seq tier for both strip colors
        and black boundary lines.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes dictionary. With ``return_data=True``, the
        ordered matrix and scale-group table are returned as a third value.
    """
    if isinstance(jaccard, dict):
        jaccard = jaccard.get("jaccard", pd.DataFrame())

    scale_group_attrs: Dict[str, Any] = {}
    if isinstance(scale_groups, pd.DataFrame):
        scale_group_attrs = dict(getattr(scale_groups, "attrs", {}) or {})
    carried_tier_config = scale_group_attrs.get("resolution_tier_config", {}) or {}
    carried_tier_enabled = bool(carried_tier_config.get("enabled", False))
    resolution_tier_enabled = bool(resolution_tier) or bool(stereoseq_resolution_tier)
    use_carried_resolution_tier = (
        carried_tier_enabled
        and isinstance(scale_groups, pd.DataFrame)
        and "resolution_tier" in scale_groups.columns
        and not resolution_tier_enabled
    )
    tier_high_resolutions = (
        tuple(float(x) for x in resolution_tier_high_resolutions)
        if resolution_tier_high_resolutions is not None
        else tuple(
            float(x)
            for x in carried_tier_config.get("high_resolutions", stereoseq_high_resolutions)
        )
    )
    if carried_tier_enabled:
        resolution_tier_include_native = bool(carried_tier_config.get("include_native", resolution_tier_include_native))
        resolution_tier_fine_group = str(carried_tier_config.get("fine_group", resolution_tier_fine_group))
        resolution_tier_exclusive = bool(carried_tier_config.get("exclusive", resolution_tier_exclusive))
        if resolution_tier_fallback_group is None:
            resolution_tier_fallback_group = carried_tier_config.get("fallback_group", None)
    tier_mode_value = str(resolution_tier_mode if resolution_tier_mode is not None else stereoseq_tier_mode).lower()
    tier_label_value = resolution_tier_label if resolution_tier_label is not None else stereoseq_tier_label
    tier_color_value = resolution_tier_color if resolution_tier_color is not None else stereoseq_tier_color
    tier_other_color_value = (
        resolution_tier_other_color
        if resolution_tier_other_color is not None
        else stereoseq_tier_other_color
    )

    inferred_boundary_groups = None
    if jaccard is None:
        if rank_df is None:
            raise ValueError("Provide either jaccard or rank_df.")
        clustering_index = segments_index
        if clustering_index is None and scale_groups is not None and not scale_groups.empty:
            clustering_index = scale_groups.copy()
            if "resolution" not in clustering_index.columns and scale_col in clustering_index.columns:
                clustering_index["resolution"] = clustering_index[scale_col]
        inferred_groups, diagnostics = infer_scale_groups_by_hclustering(
            rank_df,
            segments_index=clustering_index,
            include_native=include_native,
            top_rank_pct=top_rank_pct,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage_method=linkage_method,
            return_diagnostics=True,
        )
        if scale_groups is None:
            inferred_boundary_groups = inferred_groups
            scale_groups = inferred_groups
        elif not any(col in scale_groups.columns for col in ("hclust_target_group", "hclust_cluster")):
            inferred_boundary_groups = inferred_groups
        jaccard = diagnostics.get("jaccard", pd.DataFrame())
    elif rank_df is not None and resolution_tier_enabled and boundary_group_col is None:
        clustering_index = segments_index
        if clustering_index is None and scale_groups is not None and not scale_groups.empty:
            clustering_index = scale_groups.copy()
            if "resolution" not in clustering_index.columns and scale_col in clustering_index.columns:
                clustering_index["resolution"] = clustering_index[scale_col]
        inferred_boundary_groups = infer_scale_groups_by_hclustering(
            rank_df,
            segments_index=clustering_index,
            include_native=include_native,
            top_rank_pct=top_rank_pct,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage_method=linkage_method,
            return_diagnostics=False,
        )

    mat = pd.DataFrame(jaccard).copy()
    if mat.empty:
        raise ValueError(
            "jaccard must be a non-empty square DataFrame. If you passed rank_df, "
            "check that scale_groups['scale_id'] matches rank_df columns such as "
            "'rank_res_10', 'rank_10', 'res_10', or '10'."
        )
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    if scale_groups is None or scale_groups.empty:
        scale_groups = pd.DataFrame({scale_id_col: mat.index.astype(str), scale_col: mat.index.astype(str)})

    missing_cols = [c for c in [scale_id_col, group_col] if c not in scale_groups.columns]
    if scale_bar_group_col is not None and scale_bar_group_col not in scale_groups.columns:
        missing_cols.append(scale_bar_group_col)
    if boundary_group_col is not None and boundary_group_col not in scale_groups.columns:
        missing_cols.append(boundary_group_col)
    if resolution_tier_boundary_col is not None and resolution_tier_boundary_col not in scale_groups.columns:
        missing_cols.append(resolution_tier_boundary_col)
    if missing_cols:
        raise KeyError(f"scale_groups is missing required columns: {missing_cols}")

    groups = scale_groups.copy()
    tier_mode = tier_mode_value
    replace_with_resolution_tier = resolution_tier_enabled and tier_mode in {
        "replace",
        "main",
        "primary",
        "overwrite",
    }
    scale_bar_with_resolution_tier = resolution_tier_enabled and tier_mode in {
        "scale_bar",
        "bar",
        "strip",
    }
    show_resolution_tier_secondary = resolution_tier_enabled and tier_mode in {
        "secondary_strip",
        "secondary",
        "extra",
        "aux",
    }
    groups[scale_id_col] = groups[scale_id_col].astype(str)
    groups = _merge_scale_group_segment_metadata(
        groups,
        segments_index,
        scale_id_col=scale_id_col,
        observed_col=observed_col,
        native_col=native_col,
    )
    groups = groups.loc[groups[scale_id_col].isin(mat.index) & groups[scale_id_col].isin(mat.columns)].copy()
    if groups.empty:
        raise ValueError("No scale_groups rows match the jaccard matrix index/columns.")
    if scale_col not in groups.columns:
        groups[scale_col] = groups[scale_id_col]
    if native_col not in groups.columns:
        groups[native_col] = False
    if inferred_boundary_groups is not None and not inferred_boundary_groups.empty:
        boundary_meta_cols = [scale_id_col]
        inferred_boundary = inferred_boundary_groups.copy()
        if scale_id_col not in inferred_boundary.columns and "scale_id" in inferred_boundary.columns:
            inferred_boundary[scale_id_col] = inferred_boundary["scale_id"].astype(str)
        if "hclust_cluster" in inferred_boundary.columns:
            boundary_meta_cols.append("hclust_cluster")
        if "target_group" in inferred_boundary.columns:
            inferred_boundary["hclust_target_group"] = inferred_boundary["target_group"].astype(str)
            boundary_meta_cols.append("hclust_target_group")
        boundary_meta_cols = [c for c in boundary_meta_cols if c in inferred_boundary.columns]
        if len(boundary_meta_cols) > 1:
            boundary_meta = inferred_boundary[boundary_meta_cols].copy()
            boundary_meta[scale_id_col] = boundary_meta[scale_id_col].astype(str)
            groups = groups.merge(boundary_meta.drop_duplicates(scale_id_col), on=scale_id_col, how="left", suffixes=("", "_inferred"))
            for col in ("hclust_cluster", "hclust_target_group"):
                inferred_col = f"{col}_inferred"
                if inferred_col in groups.columns:
                    groups[col] = groups[inferred_col].where(groups[inferred_col].notna(), groups.get(col))
                    groups = groups.drop(columns=[inferred_col])

    groups["_scale_numeric"] = pd.to_numeric(groups[scale_col], errors="coerce")
    groups["_native_sort"] = groups[native_col].astype(bool).map({True: 0, False: 1})
    sort_cols = ["_scale_numeric", scale_id_col]
    if str(native_position).lower() in {"first", "left"}:
        sort_cols = ["_native_sort", "_scale_numeric", scale_id_col]
    elif str(native_position).lower() in {"last", "right"}:
        groups["_native_sort"] = groups[native_col].astype(bool).map({False: 0, True: 1})
        sort_cols = ["_native_sort", "_scale_numeric", scale_id_col]
    groups = groups.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    order = groups[scale_id_col].astype(str).tolist()
    mat = mat.loc[order, order].astype(float)
    label_mode = str(scale_label) if scale_label is not None else str(segment_label_mode)
    if label_mode.lower() in {"seg", "n_seg", "n seg"}:
        label_mode = "observed_count"
    labels = [
        _scale_overlap_axis_label(
            row,
            scale_id_col=scale_id_col,
            scale_col=scale_col,
            native_col=native_col,
            observed_col=observed_col,
            native_label=native_label,
            segment_label_mode=label_mode,
            count_suffix=count_suffix,
        )
        for _, row in groups.iterrows()
    ]

    boundary_col = boundary_group_col
    if boundary_col is None:
        if resolution_tier_boundary_col is not None:
            boundary_col = resolution_tier_boundary_col
        elif "hclust_target_group" in groups.columns:
            boundary_col = "hclust_target_group"
        elif "hclust_cluster" in groups.columns:
            boundary_col = "hclust_cluster"
        else:
            boundary_col = group_col
    hclust_group_values = groups[boundary_col].astype(str).tolist()
    strip_groups = groups.copy()
    if use_carried_resolution_tier:
        groups["resolution_tier"] = groups["resolution_tier"].astype(str)
        groups["stereoseq_resolution_tier"] = groups["resolution_tier"]
    elif replace_with_resolution_tier or scale_bar_with_resolution_tier or show_resolution_tier_secondary:
        strip_groups = apply_resolution_tier(
            groups.drop(columns=["_scale_numeric", "_native_sort"], errors="ignore"),
            None,
            high_resolutions=tier_high_resolutions,
            include_native_in_high=resolution_tier_include_native,
            source_group_col=group_col,
            out_col="__resolution_tier__",
            scale_id_col=scale_id_col,
            scale_col=scale_col,
            native_col=native_col,
            fine_group=resolution_tier_fine_group,
            exclusive_high=resolution_tier_exclusive,
            exclusive_fallback_group=resolution_tier_fallback_group,
        )
        tier_map = (
            strip_groups[[scale_id_col, "__resolution_tier__"]]
            .drop_duplicates(scale_id_col)
            .set_index(scale_id_col)["__resolution_tier__"]
            .astype(str)
            .to_dict()
        )
        groups["resolution_tier"] = groups[scale_id_col].map(tier_map)
        groups["stereoseq_resolution_tier"] = groups["resolution_tier"]
    if scale_bar_group_col is not None:
        group_values = groups[scale_bar_group_col].astype(str).tolist()
    elif use_carried_resolution_tier or replace_with_resolution_tier or scale_bar_with_resolution_tier:
        group_values = groups["resolution_tier"].fillna(groups[group_col]).astype(str).tolist()
    else:
        group_values = hclust_group_values
    boundary_group_values = hclust_group_values

    group_order = []
    for value in group_values:
        if value not in group_order:
            group_order.append(value)
    default_palette = {
        "fine_hclust": "#2ca02c",
        "high_hclust": "#2ca02c",
        "mid_hclust": "#ff7f0e",
        "coarse_hclust": "#1f77b4",
        "low_hclust": "#1f77b4",
        "scale_hclust_01": "#4c78a8",
    }
    if group_palette:
        default_palette.update(group_palette)

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fallback = plt.get_cmap("tab10")
    group_colors = {
        group: default_palette.get(group, fallback(i % 10))
        for i, group in enumerate(group_order)
    }
    group_codes = np.array([group_order.index(g) for g in group_values], dtype=float)
    group_cmap = ListedColormap([group_colors[g] for g in group_order])
    stereoseq_fine_mask = None
    if show_resolution_tier_secondary:
        stereoseq_fine_mask = np.array(
            [groups["resolution_tier"].iloc[i] == resolution_tier_fine_group for i in range(len(groups))],
            dtype=float,
        )

    n = len(order)
    if figsize is None:
        size_factor = max(float(base_fontsize) / 12.0, 1.0)
        figsize = (max(9.2, 0.40 * n * size_factor + 4.6), max(7.0, 0.40 * n * size_factor + 2.0))
    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    fig_w, fig_h = float(figsize[0]), float(figsize[1])
    heat_left = 0.08
    heat_bottom = 0.10
    cbar_w = max(float(colorbar_width), 0.020)
    cbar_gap = max(float(colorbar_gap), 0.045 + 0.003 * float(base_fontsize))
    legend_mode = str(legend_style).lower()
    show_right_legend = legend_mode in {"right", "boxed", "legend"}
    legend_gap_eff = max(float(legend_gap), 0.035) if show_right_legend else 0.035
    if legend_width is None:
        legend_w = 0.17 if show_right_legend else 0.0
    else:
        legend_w = max(float(legend_width), 0.0)
    available_w = max(0.20, 1.0 - heat_left - legend_w - cbar_w - cbar_gap - legend_gap_eff - 0.025)
    available_h = 0.78
    heat_w = min(available_w, available_h * fig_h / fig_w)
    heat_h = heat_w * fig_w / fig_h
    strip_h = max(0.010, min(0.035, heat_h * max(float(group_strip_size), 0.005)))
    strip_w = max(0.010, min(0.035, heat_w * max(float(group_strip_size), 0.005)))
    strip_gap = 0.012
    tier_strip_h = strip_h * 0.55 if show_resolution_tier_secondary else 0.0
    tier_strip_w = strip_w * 0.55 if show_resolution_tier_secondary else 0.0
    tier_gap = 0.006 if show_resolution_tier_secondary else 0.0

    ax_heat = fig.add_axes([heat_left, heat_bottom, heat_w, heat_h])
    ax_top = fig.add_axes([heat_left, heat_bottom + heat_h + strip_gap, heat_w, strip_h])
    ax_left = fig.add_axes([heat_left - strip_gap - strip_w, heat_bottom, strip_w, heat_h])
    ax_top_tier = fig.add_axes([heat_left, heat_bottom + heat_h + strip_gap + strip_h + tier_gap, heat_w, max(tier_strip_h, 0.001)])
    ax_left_tier = fig.add_axes([heat_left - strip_gap - strip_w - tier_gap - tier_strip_w, heat_bottom, max(tier_strip_w, 0.001), heat_h])
    corner_left = heat_left - strip_gap - strip_w - tier_gap - tier_strip_w
    corner_w = strip_w + tier_gap + tier_strip_w
    corner_h = strip_h + tier_gap + tier_strip_h
    ax_corner = fig.add_axes([corner_left, heat_bottom + heat_h + strip_gap, corner_w, corner_h])
    ax_cbar = fig.add_axes([heat_left + heat_w + cbar_gap, heat_bottom, cbar_w, heat_h])
    if show_right_legend:
        ax_legend = fig.add_axes([heat_left + heat_w + cbar_gap + cbar_w + legend_gap_eff, heat_bottom + 0.30 * heat_h, legend_w, 0.36 * heat_h])
    else:
        ax_legend = fig.add_axes([0.0, 0.0, 0.001, 0.001])

    ax_corner.axis("off")
    ax_top.imshow(group_codes.reshape(1, -1), aspect="auto", cmap=group_cmap, vmin=-0.5, vmax=max(len(group_order) - 0.5, 0.5))
    ax_left.imshow(group_codes.reshape(-1, 1), aspect="auto", cmap=group_cmap, vmin=-0.5, vmax=max(len(group_order) - 0.5, 0.5))
    for strip_ax in (ax_top, ax_left):
        strip_ax.set_xticks([])
        strip_ax.set_yticks([])
        for spine in strip_ax.spines.values():
            spine.set_visible(False)
    tier_cmap = ListedColormap([tier_other_color_value, tier_color_value])
    if show_resolution_tier_secondary and stereoseq_fine_mask is not None:
        ax_top_tier.imshow(stereoseq_fine_mask.reshape(1, -1), aspect="auto", cmap=tier_cmap, vmin=-0.5, vmax=1.5)
        ax_left_tier.imshow(stereoseq_fine_mask.reshape(-1, 1), aspect="auto", cmap=tier_cmap, vmin=-0.5, vmax=1.5)
        for tier_ax in (ax_top_tier, ax_left_tier):
            tier_ax.set_xticks([])
            tier_ax.set_yticks([])
            for spine in tier_ax.spines.values():
                spine.set_visible(False)
        run_start = 0
        while run_start < n:
            is_fine_run = bool(stereoseq_fine_mask[run_start])
            run_end = run_start + 1
            while run_end < n and bool(stereoseq_fine_mask[run_end]) == is_fine_run:
                run_end += 1
            if tier_label_value and is_fine_run and run_end - run_start >= int(inline_legend_min_run):
                ax_top_tier.text(
                    (run_start + run_end - 1) / 2.0,
                    0.0,
                    str(tier_label_value),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=max((inline_legend_fontsize or base_fontsize - 8.0), 7.0),
                    fontweight="bold",
                    clip_on=True,
                )
            run_start = run_end
    else:
        ax_top_tier.axis("off")
        ax_left_tier.axis("off")
    ax_legend.axis("off")
    if legend_mode in {"inline", "strip", "top"}:
        strip_height_pt = strip_h * fig_h * 72.0
        default_inline_size = min(max(float(base_fontsize) - 7.0, 8.0), max(strip_height_pt * 0.42, 6.0))
        inline_size = float(inline_legend_fontsize) if inline_legend_fontsize is not None else default_inline_size
        run_start = 0
        while run_start < n:
            run_group = group_values[run_start]
            run_end = run_start + 1
            while run_end < n and group_values[run_end] == run_group:
                run_end += 1
            if run_end - run_start >= int(inline_legend_min_run):
                x_mid = (run_start + run_end - 1) / 2.0
                bbox = (
                    {"facecolor": "black", "edgecolor": "none", "alpha": 0.22, "boxstyle": "round,pad=0.18"}
                    if inline_legend_bbox
                    else None
                )
                ax_top.text(
                    x_mid,
                    float(inline_legend_y),
                    _scale_group_display_label(run_group),
                    ha="center",
                    va="center",
                    color=inline_legend_color,
                    fontsize=inline_size,
                    fontweight="bold",
                    bbox=bbox,
                    clip_on=True,
                )
            run_start = run_end

    im = ax_heat.imshow(mat.to_numpy(float), cmap=_resolve_scale_overlap_cmap(cmap), vmin=0, vmax=1, aspect="equal")
    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_yticks(np.arange(n))
    if cell_grid:
        ax_heat.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax_heat.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax_heat.grid(which="minor", color=grid_color, linestyle=grid_linestyle, linewidth=float(grid_linewidth), alpha=0.8)
        ax_heat.tick_params(which="minor", bottom=False, left=False)
    if scale_boundary_lines:
        for boundary in np.arange(0.5, n - 0.5, 1.0):
            ax_heat.axvline(boundary, color=scale_boundary_color, lw=float(scale_boundary_linewidth), alpha=0.55)
            ax_heat.axhline(boundary, color=scale_boundary_color, lw=float(scale_boundary_linewidth), alpha=0.55)
    if group_boundary_lines:
        for boundary_idx in range(1, n):
            if boundary_group_values[boundary_idx] != boundary_group_values[boundary_idx - 1]:
                boundary = boundary_idx - 0.5
                ax_heat.axvline(boundary, color=group_boundary_color, lw=float(group_boundary_linewidth))
                ax_heat.axhline(boundary, color=group_boundary_color, lw=float(group_boundary_linewidth))
    heat_w_inches = heat_w * fig_w
    heat_h_inches = heat_h * fig_h
    col_width_pt = heat_w_inches * 72.0 / max(n, 1)
    row_height_pt = heat_h_inches * 72.0 / max(n, 1)
    if tick_fontsize is None:
        tick_size = float(base_fontsize)
        if auto_shrink_tick_font:
            max_tick = min(tick_size, 0.86 * col_width_pt, 0.78 * row_height_pt)
            tick_size = max(float(min_tick_fontsize), max_tick)
    else:
        tick_size = float(tick_fontsize)
    title_size = float(title_fontsize) if title_fontsize is not None else max(float(base_fontsize) + 3.0, 15.0)
    axis_size = float(axis_label_fontsize) if axis_label_fontsize is not None else max(float(base_fontsize), 12.0)
    cbar_size = float(colorbar_fontsize) if colorbar_fontsize is not None else max(float(base_fontsize), 12.0)
    legend_size = float(legend_fontsize) if legend_fontsize is not None else max(float(base_fontsize), 12.0)
    legend_title_size = float(legend_title_fontsize) if legend_title_fontsize is not None else max(float(base_fontsize) + 1.0, 13.0)
    ax_heat.set_xticklabels(labels, rotation=float(xtick_rotation), ha="center", fontsize=tick_size)
    ax_heat.set_yticklabels(labels, fontsize=tick_size)
    ax_heat.yaxis.tick_right()
    ax_heat.yaxis.set_label_position("right")
    ax_heat.tick_params(length=0)
    ax_heat.set_xlabel("scale" if show_xlabel else "", fontsize=axis_size)
    ax_heat.set_ylabel("scale" if show_ylabel else "", fontsize=axis_size)

    fig_title = title if title is not None else "Top-SVG overlap across scales"
    annotation_top = heat_bottom + heat_h + strip_gap + strip_h + tier_gap + tier_strip_h
    title_pad_eff = float(title_pad)
    if show_resolution_tier_secondary:
        title_pad_eff = max(title_pad_eff, 1.25 * title_size / (72.0 * fig_h))
    title_pos = float(title_y) if title_y is not None else min(0.975, annotation_top + title_pad_eff)
    fig.suptitle(fig_title, y=title_pos, fontsize=title_size, fontweight="bold")
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(similarity_label, labelpad=10, fontsize=cbar_size)
    if str(colorbar_label_side).lower() == "left":
        cbar.ax.yaxis.set_label_position("left")
    else:
        cbar.ax.yaxis.set_label_position("right")
    if str(colorbar_tick_side).lower() == "right":
        cbar.ax.yaxis.tick_right()
    else:
        cbar.ax.yaxis.tick_left()
    cbar.ax.tick_params(labelsize=max(cbar_size - 1.0, float(min_tick_fontsize)))

    handles = [
        Patch(facecolor=group_colors[group], edgecolor="none", label=_scale_group_display_label(group))
        for group in group_order
    ]
    if show_right_legend:
        ax_legend.legend(
            handles=handles,
            title=group_legend_title,
            frameon=False,
            loc="center",
            fontsize=legend_size,
            title_fontsize=legend_title_size,
            handlelength=1.6,
            handleheight=0.7,
            borderaxespad=0.0,
            labelspacing=0.45,
        )
    if save is not None:
        fig.savefig(Path(save), dpi=int(dpi), bbox_inches="tight")

    axes = {
        "heatmap": ax_heat,
        "top_group": ax_top,
        "left_group": ax_left,
        "top_stereoseq_tier": ax_top_tier,
        "left_stereoseq_tier": ax_left_tier,
        "corner": ax_corner,
        "colorbar": ax_cbar,
        "legend": ax_legend,
    }
    if return_data:
        data = {"jaccard": mat, "scale_groups": groups.drop(columns=["_scale_numeric", "_native_sort"])}
        return fig, axes, data
    return fig, axes


plot_hclust_scale_overlap_heatmap = plot_scale_svg_overlap_heatmap


def multiscale_moran_to_long_table(
    rank_df: pd.DataFrame,
    moran_df: Optional[pd.DataFrame] = None,
    segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]] = None,
    *,
    sample: Optional[str] = None,
    include_native: bool = True,
    native_group: str = "native",
    scale_col: str = "scale",
    group_col: str = "target_group",
    scale_groups: Optional[Dict[str, Any]] = None,
    scale_group_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Convert ``multiscale_moran_ranks`` wide outputs to a long SVG table.

    Parameters
    ----------
    rank_df
        Gene x scale rank table returned by :func:`multiscale_moran_ranks`.
        Columns usually look like ``rank_<scale_id>``.
    moran_df
        Optional Gene x scale Moran score table returned with
        ``return_scores=True``. Columns usually look like ``I_<scale_id>``.
    segments_index
        DataFrame or ``segments_index.csv`` from
        :func:`SPIX.sp.precompute_multiscale_segments`. When provided, scale ids
        are mapped to their numeric ``resolution`` values and native identity
        rows can be excluded.
    include_native
        Include native-identity scale rows in the long output. The default is
        True so the native resolution remains visible in downstream rank
        trajectories and scale-group inference.
    scale_groups
        Optional mapping used to add ``group_col`` immediately. Example:
        ``{"high": (20, 90), "mid": (100, 180), "low": (190, 300)}``.
    scale_group_table
        Optional output of :func:`infer_scale_groups_by_hclustering`. When
        provided, ``scale_id`` is used to assign ``group_col``.
    """
    if rank_df is None or rank_df.empty:
        return pd.DataFrame()

    rank = rank_df.copy()
    rank.index = rank.index.astype(str)
    score = None
    if moran_df is not None and not moran_df.empty:
        score = moran_df.copy()
        score.index = score.index.astype(str)

    index_df = _read_segments_index_like(segments_index)
    rows = []

    if index_df is not None:
        if "scale_id" not in index_df.columns:
            raise KeyError("segments_index must contain 'scale_id'.")
        if "resolution" not in index_df.columns:
            raise KeyError("segments_index must contain 'resolution'.")
        iterator = index_df.reset_index(drop=True).iterrows()
    else:
        index_df = pd.DataFrame(
            {
                "scale_id": [str(c).replace("rank_", "", 1) for c in rank.columns],
                "resolution": [str(c).replace("rank_", "", 1) for c in rank.columns],
            }
        )
        iterator = index_df.iterrows()

    for _, row in iterator:
        sid = str(row["scale_id"])
        is_native = _is_native_index_row(row)
        if is_native and not include_native:
            continue

        rank_col_found = _find_scale_column(rank, sid, ("rank_",))
        if rank_col_found is None:
            continue
        moran_col_found = None if score is None else _find_scale_column(score, sid, ("I_", "moran_", "score_"))

        scale_value = row["resolution"]
        try:
            scale_value = float(scale_value)
        except Exception:
            pass
        target_group = native_group if is_native else pd.NA

        r = pd.to_numeric(rank[rank_col_found], errors="coerce")
        n_genes = int(rank.shape[0])
        tmp = pd.DataFrame(
            {
                "gene": rank.index,
                scale_col: scale_value,
                "scale_id": sid,
                "moran_rank": r.to_numpy(),
                "n_genes_in_scale": n_genes,
                "native_identity": bool(is_native),
                group_col: target_group,
            }
        )
        if sample is not None:
            tmp["sample"] = str(sample)
        if score is not None and moran_col_found is not None:
            tmp["moran_I"] = pd.to_numeric(score.reindex(rank.index)[moran_col_found], errors="coerce").to_numpy()
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.loc[out["moran_rank"].notna()].copy()
    if scale_group_table is not None:
        if "scale_id" not in scale_group_table.columns or group_col not in scale_group_table.columns:
            raise KeyError(f"scale_group_table must contain 'scale_id' and {group_col!r}.")
        group_map = (
            scale_group_table[["scale_id", group_col]]
            .dropna(subset=["scale_id", group_col])
            .assign(scale_id=lambda x: x["scale_id"].astype(str))
            .drop_duplicates("scale_id")
            .set_index("scale_id")[group_col]
            .astype(str)
            .to_dict()
        )
        out[group_col] = out["scale_id"].astype(str).map(group_map)
        if out[group_col].isna().any():
            missing = sorted(out.loc[out[group_col].isna(), "scale_id"].astype(str).unique().tolist())
            raise ValueError(f"Some scale_id values were not assigned by scale_group_table: {missing[:20]}")
    if scale_groups is not None:
        non_native = ~out["native_identity"].astype(bool)
        out.loc[non_native, group_col] = _assign_scale_groups(
            out.loc[non_native],
            scale_col=scale_col,
            scale_groups=scale_groups,
            out_col=group_col,
        ).to_numpy()
    return out


def _impute_multiscale_missing_as_worst(
    rank_df: pd.DataFrame,
    moran_df: Optional[pd.DataFrame],
    segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Fill unavailable gene-scale values with conservative worst values."""
    rank = rank_df.copy()
    rank.index = rank.index.astype(str)
    score = None
    if moran_df is not None and not moran_df.empty:
        score = moran_df.copy()
        score.index = score.index.astype(str)

    index_df = _read_segments_index_like(segments_index)
    if index_df is not None and "scale_id" in index_df.columns:
        scale_ids = index_df["scale_id"].astype(str).tolist()
    else:
        scale_ids = [str(c).replace("rank_", "", 1) for c in rank.columns if str(c).startswith("rank_")]

    rows = []
    worst_rank = float(max(len(rank), 1))
    for sid in scale_ids:
        rank_col = _find_scale_column(rank, sid, ("rank_",))
        if rank_col is None:
            continue
        r = pd.to_numeric(rank[rank_col], errors="coerce")
        n_missing_rank = int(r.isna().sum())
        if n_missing_rank:
            rank[rank_col] = r.fillna(worst_rank)

        n_missing_moran = 0
        moran_floor = np.nan
        if score is not None:
            moran_col = _find_scale_column(score, sid, ("I_", "moran_", "score_"))
            if moran_col is not None:
                m = pd.to_numeric(score[moran_col], errors="coerce")
                n_missing_moran = int(m.isna().sum())
                finite = m[np.isfinite(m)]
                moran_floor = float(finite.min()) if len(finite) else 0.0
                if n_missing_moran:
                    score[moran_col] = m.fillna(moran_floor)
        rows.append(
            {
                "scale_id": sid,
                "n_missing_rank_filled": n_missing_rank,
                "rank_fill_value": worst_rank if n_missing_rank else np.nan,
                "n_missing_moran_filled": n_missing_moran,
                "moran_fill_value": moran_floor if n_missing_moran else np.nan,
            }
        )
    return rank, score, pd.DataFrame(rows)


def call_scale_specific_svgs_from_multiscale(
    rank_df: pd.DataFrame,
    moran_df: Optional[pd.DataFrame] = None,
    segments_index: Optional[Union[str, os.PathLike, pd.DataFrame]] = None,
    *,
    sample: Optional[str] = None,
    scale_groups: Optional[Dict[str, Any]] = None,
    scale_group_table: Optional[pd.DataFrame] = None,
    scale_group_table_group_col: str = "target_group",
    group_order: Optional[List[str]] = None,
    group_method: str = "hclust",
    n_scale_clusters: int = 3,
    top_rank_pct_for_clustering: float = 0.10,
    hclust_distance_threshold: Optional[float] = None,
    hclust_linkage_method: str = "average",
    resolution_tier: bool = False,
    resolution_tier_high_resolutions: Optional[Tuple[float, ...]] = None,
    resolution_tier_include_native: bool = True,
    resolution_tier_fine_group: str = "fine_hclust",
    resolution_tier_exclusive: bool = False,
    resolution_tier_fallback_group: Optional[str] = None,
    stereoseq_resolution_tier: bool = False,
    stereoseq_high_resolutions: Tuple[float, ...] = (2.0, 8.0, 16.0),
    include_native: bool = True,
    return_long: bool = False,
    return_all: bool = False,
    return_group_table: bool = False,
    return_diagnostics: bool = False,
    missing_as_worst: bool = True,
    return_missing_imputation: bool = False,
    **call_kwargs: Any,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    """Call scale-preferential SVG candidates after ``multiscale_moran_ranks``.

    This is the intended notebook continuation after:

    ``rank_df, moran_df = SPIX.an.multiscale_moran_ranks(..., return_scores=True)``

    By default, scale groups are inferred by hierarchical clustering of
    scale-to-scale top-SVG Jaccard overlap. Native identity rows are included
    in the clustering/call by default when they are present in
    ``segments_index``.
    Set ``include_native=False`` to cluster only SPIX segment scales.

    When ``missing_as_worst=True`` (default), unavailable gene-scale entries
    are retained as conservative worst observations before scale grouping and
    scale-preferential SVG candidate calling: missing ranks are filled with the number of
    genes, and missing Moran values are filled with the lowest observed Moran's
    I for that scale. This prevents scale specificity from being inflated by
    dropping scales where a gene was not evaluated or did not pass filtering.

    Set ``resolution_tier=True`` when native identity plus selected fine
    physical resolutions should be treated as the fine/high-resolution tier
    even if top-SVG hclustering makes native a singleton group. The older
    ``stereoseq_resolution_tier`` option is retained as a backward-compatible
    alias.
    """
    missing_imputation = pd.DataFrame()
    if missing_as_worst:
        rank_df, moran_df, missing_imputation = _impute_multiscale_missing_as_worst(rank_df, moran_df, segments_index)
    if scale_groups is not None and scale_group_table is not None:
        raise ValueError("Pass either scale_groups or scale_group_table, not both.")

    resolved_scale_group_table = None
    diagnostics = None
    scale_group_method = "manual" if scale_groups is not None else "hclust_top_gene_jaccard"
    if scale_group_table is not None:
        if "scale_id" not in scale_group_table.columns:
            raise KeyError("scale_group_table must contain 'scale_id'.")
        if scale_group_table_group_col not in scale_group_table.columns:
            raise KeyError(f"scale_group_table must contain {scale_group_table_group_col!r}.")
        resolved_scale_group_table = scale_group_table.copy()
        if scale_group_table_group_col != "target_group":
            resolved_scale_group_table["target_group"] = resolved_scale_group_table[scale_group_table_group_col]
        resolved_scale_group_table["scale_id"] = resolved_scale_group_table["scale_id"].astype(str)
        resolved_scale_group_table["target_group"] = resolved_scale_group_table["target_group"].astype(str)
        scale_group_method = "scale_group_table"
        if group_order is None:
            group_order = list(resolved_scale_group_table.attrs.get("group_order", []))
            if not group_order:
                group_order = resolved_scale_group_table["target_group"].dropna().drop_duplicates().astype(str).tolist()
    elif scale_groups is None:
        if str(group_method).lower() not in {"hclust", "hierarchical", "hierarchical_clustering"}:
            raise ValueError("When scale_groups is not provided, group_method must be 'hclust'.")
        inferred = infer_scale_groups_by_hclustering(
            rank_df,
            segments_index=segments_index,
            include_native=include_native,
            top_rank_pct=top_rank_pct_for_clustering,
            n_clusters=n_scale_clusters,
            distance_threshold=hclust_distance_threshold,
            linkage_method=hclust_linkage_method,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            resolved_scale_group_table, diagnostics = inferred  # type: ignore[misc]
        else:
            resolved_scale_group_table = inferred  # type: ignore[assignment]
        if group_order is None:
            group_order = list(resolved_scale_group_table.attrs.get("group_order", []))
            if not group_order and not resolved_scale_group_table.empty:
                group_order = (
                    resolved_scale_group_table[["target_group", "scale"]]
                    .drop_duplicates("target_group")
                    .sort_values("scale", kind="mergesort")["target_group"]
                    .astype(str)
                    .tolist()
                )
    resolution_tier_enabled = bool(resolution_tier) or bool(stereoseq_resolution_tier)
    tier_high_resolutions = (
        tuple(float(x) for x in resolution_tier_high_resolutions)
        if resolution_tier_high_resolutions is not None
        else tuple(float(x) for x in stereoseq_high_resolutions)
    )
    if resolution_tier_enabled:
        if resolved_scale_group_table is None:
            raise ValueError("resolution_tier=True requires hclust grouping or scale_group_table, not scale_groups dict alone.")
        if "target_group" in resolved_scale_group_table.columns and "hclust_target_group" not in resolved_scale_group_table.columns:
            resolved_scale_group_table["hclust_target_group"] = resolved_scale_group_table["target_group"].astype(str)
        resolved_scale_group_table = apply_resolution_tier(
            resolved_scale_group_table,
            segments_index,
            high_resolutions=tier_high_resolutions,
            include_native_in_high=resolution_tier_include_native,
            source_group_col="target_group",
            out_col="target_group",
            fine_group=resolution_tier_fine_group,
            exclusive_high=resolution_tier_exclusive,
            exclusive_fallback_group=resolution_tier_fallback_group,
        )
        resolved_scale_group_table["resolution_tier"] = resolved_scale_group_table["target_group"].astype(str)
        resolved_scale_group_table["scale_group_display"] = resolved_scale_group_table["target_group"].astype(str)
        resolved_scale_group_table.attrs["scale_group_display_col"] = "target_group"
        resolved_scale_group_table.attrs["resolution_tier_config"] = {
            "enabled": True,
            "high_resolutions": tier_high_resolutions,
            "include_native": bool(resolution_tier_include_native),
            "fine_group": str(resolution_tier_fine_group),
            "exclusive": bool(resolution_tier_exclusive),
            "fallback_group": None if resolution_tier_fallback_group is None else str(resolution_tier_fallback_group),
        }
        group_order = list(resolved_scale_group_table.attrs.get("group_order", []))
        if not group_order:
            group_order = resolved_scale_group_table["target_group"].dropna().drop_duplicates().astype(str).tolist()
        scale_group_method = f"{scale_group_method}_resolution_tier"
    if group_order is None:
        group_order = [str(k) for k in scale_groups.keys()] if scale_groups is not None else None

    per_scale = multiscale_moran_to_long_table(
        rank_df,
        moran_df=moran_df,
        segments_index=segments_index,
        sample=sample,
        include_native=include_native,
        scale_groups=scale_groups,
        scale_group_table=resolved_scale_group_table,
        group_col="target_group",
    )
    per_scale.attrs["scale_group_table"] = resolved_scale_group_table
    per_scale.attrs["scale_group_method"] = scale_group_method
    if missing_as_worst:
        per_scale.attrs["missing_as_worst"] = True
    if per_scale.empty:
        empty = pd.DataFrame()
        parts: List[pd.DataFrame] = [empty]
        if return_all:
            parts.append(empty)
        if return_long:
            parts.append(per_scale)
        if return_group_table:
            parts.append(resolved_scale_group_table if resolved_scale_group_table is not None else empty)
        if return_diagnostics:
            parts.append(diagnostics if diagnostics is not None else {})
        if return_missing_imputation:
            parts.append(missing_imputation)
        return tuple(parts) if len(parts) > 1 else empty

    calls_result = call_scale_specific_svgs(
        per_scale,
        sample_col="sample" if "sample" in per_scale.columns else None,
        group_col="target_group",
        group_order=group_order,
        qval_col=None,
        return_all=return_all,
        **call_kwargs,
    )

    if return_all:
        calls, all_genes = calls_result  # type: ignore[misc]
        calls.attrs["scale_group_table"] = resolved_scale_group_table
        all_genes.attrs["scale_group_table"] = resolved_scale_group_table
        calls.attrs["scale_group_method"] = scale_group_method
        all_genes.attrs["scale_group_method"] = scale_group_method
        if return_diagnostics:
            calls.attrs["scale_group_diagnostics"] = diagnostics
            all_genes.attrs["scale_group_diagnostics"] = diagnostics
        if missing_as_worst:
            calls.attrs["missing_as_worst"] = True
            all_genes.attrs["missing_as_worst"] = True
        parts = [calls, all_genes]
        if return_long:
            parts.append(per_scale)
        if return_group_table:
            parts.append(resolved_scale_group_table if resolved_scale_group_table is not None else pd.DataFrame())
        if return_diagnostics:
            parts.append(diagnostics if diagnostics is not None else {})
        if return_missing_imputation:
            parts.append(missing_imputation)
        return tuple(parts)  # type: ignore[return-value]
    if isinstance(calls_result, pd.DataFrame):
        calls_result.attrs["scale_group_table"] = resolved_scale_group_table
        calls_result.attrs["scale_group_method"] = scale_group_method
        if return_diagnostics:
            calls_result.attrs["scale_group_diagnostics"] = diagnostics
        if missing_as_worst:
            calls_result.attrs["missing_as_worst"] = True
    if return_long:
        parts = [calls_result, per_scale]
        if return_group_table:
            parts.append(resolved_scale_group_table if resolved_scale_group_table is not None else pd.DataFrame())
        if return_diagnostics:
            parts.append(diagnostics if diagnostics is not None else {})
        if return_missing_imputation:
            parts.append(missing_imputation)
        return tuple(parts)  # type: ignore[return-value]
    if return_group_table or return_diagnostics:
        parts = [calls_result]
        if return_group_table:
            parts.append(resolved_scale_group_table if resolved_scale_group_table is not None else pd.DataFrame())
        if return_diagnostics:
            parts.append(diagnostics if diagnostics is not None else {})
        if return_missing_imputation:
            parts.append(missing_imputation)
        return tuple(parts)  # type: ignore[return-value]
    if return_missing_imputation:
        return calls_result, missing_imputation  # type: ignore[return-value]
    return calls_result


def summarize_scale_specific_svg_gene_union(
    calls: pd.DataFrame,
    *,
    group_col: str = "target_group",
    gene_col: str = "gene",
    sample_col: str = "sample",
) -> pd.DataFrame:
    """Summarize scale-specific SVG calls as one row per scale group and gene."""
    if calls is None or calls.empty:
        return pd.DataFrame()
    required = {group_col, gene_col}
    missing = required.difference(calls.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    work = calls.copy()
    if sample_col not in work.columns:
        work[sample_col] = "sample"

    agg_kwargs = {
        "samples": (sample_col, lambda x: ";".join(sorted(map(str, set(x))))),
        "n_samples": (sample_col, lambda x: len(set(x))),
    }
    metric_cols = ["target_median_rank_percentile", "gap_to_second_best", "target_good_fraction"]
    for col in metric_cols:
        if col in work.columns:
            agg_kwargs[f"best_{col}"] = (col, "min" if "rank" in col else "max")
            agg_kwargs[f"median_{col}"] = (col, "median")

    out = work.groupby([group_col, gene_col], observed=True).agg(**agg_kwargs).reset_index()
    if "best_target_median_rank_percentile" in out.columns:
        out = out.sort_values(
            [group_col, "best_target_median_rank_percentile", "n_samples"],
            ascending=[True, True, False],
        )
    else:
        out = out.sort_values([group_col, gene_col])
    return out.reset_index(drop=True)


def plot_scale_specific_svg_trajectories(
    per_scale: pd.DataFrame,
    calls: Optional[pd.DataFrame] = None,
    genes: Optional[List[str]] = None,
    *,
    sample: Any = None,
    sample_col: Optional[str] = "sample",
    gene_col: str = "gene",
    scale_col: str = "scale",
    rank_col: str = "moran_rank",
    n_genes_col: str = "n_genes_in_scale",
    moran_col: Optional[str] = "moran_I",
    metric: str = "rank_percentile",
    top_n: int = 10,
    group_col: str = "target_group",
    color_by_group: bool = True,
    figsize: Tuple[float, float] = (8, 4.8),
    dpi: int = 150,
):
    """Plot rank-percentile or Moran trajectories for selected scale-specific SVGs."""
    if per_scale is None or per_scale.empty:
        raise ValueError("per_scale must be a non-empty DataFrame.")
    if gene_col not in per_scale.columns or scale_col not in per_scale.columns:
        raise KeyError(f"per_scale must contain {gene_col!r} and {scale_col!r}.")

    work = per_scale.copy()
    internal_sample_col = sample_col
    if sample_col is None or sample_col not in work.columns:
        internal_sample_col = "__sample__"
        work[internal_sample_col] = "sample"
    if sample is not None:
        work = work.loc[work[internal_sample_col].astype(str).eq(str(sample))].copy()

    if rank_col not in work.columns:
        if moran_col is None or moran_col not in work.columns:
            raise KeyError(f"Missing {rank_col}; cannot compute rank percentile without {moran_col!r}.")
        work[rank_col] = (
            work.groupby([internal_sample_col, scale_col], observed=True)[moran_col]
            .rank(method="min", ascending=False)
            .astype(int)
        )
    if n_genes_col not in work.columns:
        work[n_genes_col] = work.groupby([internal_sample_col, scale_col], observed=True)[gene_col].transform("size")
    work["rank_percentile"] = pd.to_numeric(work[rank_col], errors="coerce") / pd.to_numeric(work[n_genes_col], errors="coerce")

    if genes is None:
        if calls is None or calls.empty:
            raise ValueError("Provide genes or non-empty calls.")
        call_work = calls.copy()
        if sample is not None and "sample" in call_work.columns:
            call_work = call_work.loc[call_work["sample"].astype(str).eq(str(sample))]
        sort_cols = [c for c in ["target_median_rank_percentile", "gap_to_second_best"] if c in call_work.columns]
        if sort_cols:
            call_work = call_work.sort_values(sort_cols, ascending=[True, False][: len(sort_cols)])
        genes = [str(g) for g in call_work[gene_col].head(int(top_n)).tolist()]
    else:
        genes = [str(g) for g in genes]

    metric_key = str(metric).lower()
    value_col = "rank_percentile" if metric_key in {"rank", "rank_percentile", "rank_pct"} else str(moran_col)
    if value_col not in work.columns:
        raise KeyError(f"Cannot plot metric {metric!r}; missing column {value_col!r}.")

    plot_df = work.loc[work[gene_col].astype(str).isin(genes)].copy()
    if plot_df.empty:
        raise ValueError("None of the requested genes were found in per_scale.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
    group_map: Dict[str, str] = {}
    if calls is not None and not calls.empty and group_col in calls.columns:
        cwork = calls.copy()
        if sample is not None and "sample" in cwork.columns:
            cwork = cwork.loc[cwork["sample"].astype(str).eq(str(sample))]
        group_map = dict(zip(cwork[gene_col].astype(str), cwork[group_col].astype(str)))

    default_colors = plt.get_cmap("tab10")
    groups = sorted(set(group_map.values()))
    group_colors = {g: default_colors(i % 10) for i, g in enumerate(groups)}

    for i, gene in enumerate(genes):
        gdf = plot_df.loc[plot_df[gene_col].astype(str).eq(gene)].copy()
        if gdf.empty:
            continue
        gdf["_scale_numeric"] = pd.to_numeric(gdf[scale_col], errors="coerce")
        gdf = gdf.sort_values("_scale_numeric")
        color = group_colors.get(group_map.get(gene, ""), default_colors(i % 10)) if color_by_group else default_colors(i % 10)
        ax.plot(
            gdf["_scale_numeric"],
            pd.to_numeric(gdf[value_col], errors="coerce"),
            marker="o",
            lw=1.8,
            ms=4,
            label=gene,
            color=color,
        )

    if value_col == "rank_percentile":
        ax.invert_yaxis()
        ax.axhline(0.10, color="black", ls="--", lw=0.8, alpha=0.7)
        ax.set_ylabel("Moran rank percentile")
    else:
        ax.axhline(0.0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylabel(value_col)
    ax.set_xlabel("Scale")
    ax.set_title("Scale-specific SVG trajectories", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    fig.tight_layout()
    return fig, ax


def classify_scale_specific_svg_score_primary(
    res_score: pd.DataFrame,
    res_rank: Optional[pd.DataFrame] = None,
    *,
    region_cols: Dict[str, List[Any]],
    resolve_region_cols: bool = True,
    region_col_atol: float = 1e-4,
    fillna_score_floor: bool = True,
    fillna_worst_rank: bool = True,
    min_score_mean: float = 0.25,
    min_score_gap: float = 0.15,
    min_score_min: float = 0.0,
    max_score_std: Optional[float] = 0.20,
    min_score_pct: float = 0.75,
    min_rank_pref: Optional[float] = 0.70,
) -> pd.DataFrame:
    """
    Select scale-specific SVGs with Moran's I as the primary effect size.

    This is stricter than ``classify_scale_specific_scores_quantile`` for
    coarse scales: it uses raw Moran's I means/gaps and can require the weakest
    resolution within the best region to remain above ``min_score_min``. That
    avoids selecting genes that spike to Moran's I ~= 1 at only one coarse scale
    while staying near zero elsewhere.
    """
    if res_score is None or len(res_score) == 0:
        return pd.DataFrame(index=getattr(res_score, "index", None))
    if region_cols is None:
        raise ValueError("region_cols must be provided explicitly.")

    if bool(resolve_region_cols):
        score_region_cols = _resolve_region_cols_for_table(
            res_score,
            region_cols,
            atol=float(region_col_atol),
        )
    else:
        score_region_cols = {
            str(region): [str(c) for c in cols if str(c) in res_score.columns]
            for region, cols in region_cols.items()
        }
    missing_score = [region for region, cols in score_region_cols.items() if len(cols) == 0]
    if missing_score:
        raise ValueError(
            f"Missing score columns for groups: {missing_score}. "
            f"Available columns: {list(map(str, res_score.columns))}"
        )

    score_cols = [c for cols in score_region_cols.values() for c in cols]
    score_raw = res_score.loc[:, score_cols].apply(pd.to_numeric, errors="coerce")
    if fillna_score_floor:
        col_floor = score_raw.min(axis=0, skipna=True)
        for c in score_raw.columns:
            floor = float(col_floor.get(c, np.nan))
            if not np.isfinite(floor):
                floor = 0.0
            score_raw[c] = score_raw[c].fillna(floor)
    score_pct = _score_percentile_table(score_raw, fillna_floor=fillna_score_floor)

    common_index = score_raw.index
    rank_region_cols = None
    rank_pref = None
    if res_rank is not None:
        common_index = common_index.intersection(res_rank.index)
        if len(common_index) == 0:
            raise ValueError("res_score and res_rank have no overlapping gene index.")
        score_raw = score_raw.loc[common_index]
        score_pct = score_pct.loc[common_index]
        res_score = res_score.loc[common_index]
        res_rank = res_rank.loc[common_index]

        if bool(resolve_region_cols):
            rank_region_cols = _resolve_region_cols_for_table(
                res_rank,
                region_cols,
                atol=float(region_col_atol),
            )
        else:
            rank_region_cols = {
                str(region): [str(c) for c in cols if str(c) in res_rank.columns]
                for region, cols in region_cols.items()
            }
        missing_rank = [region for region, cols in rank_region_cols.items() if len(cols) == 0]
        if missing_rank:
            raise ValueError(
                f"Missing rank columns for groups: {missing_rank}. "
                f"Available columns: {list(map(str, res_rank.columns))}"
            )
        rank_cols = [c for cols in rank_region_cols.values() for c in cols]
        rank_pref = _rank_preference_table(res_rank.loc[:, rank_cols], fillna_worst=fillna_worst_rank)

    out = res_score.loc[common_index].copy()

    for region, cols in score_region_cols.items():
        out[f"score_mean_{region}"] = score_raw[cols].mean(axis=1)
        out[f"score_min_{region}"] = score_raw[cols].min(axis=1)
        out[f"score_std_{region}"] = score_raw[cols].std(axis=1, ddof=0)
        out[f"score_pct_mean_{region}"] = score_pct[cols].mean(axis=1)

    mean_tbl = pd.DataFrame({region: out[f"score_mean_{region}"] for region in score_region_cols})
    min_tbl = pd.DataFrame({region: out[f"score_min_{region}"] for region in score_region_cols})
    std_tbl = pd.DataFrame({region: out[f"score_std_{region}"] for region in score_region_cols})
    pct_tbl = pd.DataFrame({region: out[f"score_pct_mean_{region}"] for region in score_region_cols})

    mean_arr = mean_tbl.to_numpy(dtype=np.float64, copy=False)
    sort_idx = np.argsort(-mean_arr, axis=1)
    best_idx = sort_idx[:, 0]
    second_idx = sort_idx[:, 1] if mean_arr.shape[1] > 1 else np.zeros(mean_arr.shape[0], dtype=np.int64)
    region_names = np.asarray(mean_tbl.columns, dtype=object)

    out["best_region"] = region_names[best_idx]
    out["score_best_mean"] = mean_arr[np.arange(len(out)), best_idx]
    out["score_second_mean"] = mean_arr[np.arange(len(out)), second_idx]
    out["score_gap"] = out["score_best_mean"] - out["score_second_mean"]
    out["score_gap_rel"] = out["score_gap"] / (np.abs(out["score_best_mean"]) + 1e-8)
    out["score_best_min"] = min_tbl.to_numpy(dtype=np.float64, copy=False)[np.arange(len(out)), best_idx]
    out["score_best_std"] = std_tbl.to_numpy(dtype=np.float64, copy=False)[np.arange(len(out)), best_idx]
    out["score_best_pct"] = pct_tbl.to_numpy(dtype=np.float64, copy=False)[np.arange(len(out)), best_idx]

    score_supported = (
        (out["score_best_mean"] >= float(min_score_mean)) &
        (out["score_gap"] >= float(min_score_gap)) &
        (out["score_best_min"] >= float(min_score_min)) &
        (out["score_best_pct"] >= float(min_score_pct))
    )
    if max_score_std is not None:
        score_supported &= out["score_best_std"] <= float(max_score_std)

    if rank_pref is not None and rank_region_cols is not None and min_rank_pref is not None:
        for region, cols in rank_region_cols.items():
            out[f"rank_pref_mean_{region}"] = rank_pref[cols].mean(axis=1)
        rank_tbl = pd.DataFrame({region: out[f"rank_pref_mean_{region}"] for region in score_region_cols})
        best_region_index = out["best_region"].map({r: i for i, r in enumerate(rank_tbl.columns)}).to_numpy()
        out["rank_pref_best_region"] = rank_tbl.to_numpy(dtype=np.float64, copy=False)[np.arange(len(out)), best_region_index]
        rank_supported = out["rank_pref_best_region"] >= float(min_rank_pref)
    else:
        out["rank_pref_best_region"] = np.nan
        rank_supported = pd.Series(True, index=out.index)

    out["score_supported"] = score_supported
    out["rank_supported"] = rank_supported
    out["is_svg_score_primary"] = out["score_supported"] & out["rank_supported"]
    out["score_primary_score"] = (
        out["score_best_mean"]
        + 0.50 * out["score_gap"]
        + 0.15 * out["score_best_pct"]
        - 0.25 * out["score_best_std"]
    )
    out["category"] = np.where(out["is_svg_score_primary"], out["best_region"], "mixed")
    out.attrs["score_primary_thresholds"] = {
        "min_score_mean": float(min_score_mean),
        "min_score_gap": float(min_score_gap),
        "min_score_min": float(min_score_min),
        "max_score_std": None if max_score_std is None else float(max_score_std),
        "min_score_pct": float(min_score_pct),
        "min_rank_pref": None if min_rank_pref is None else float(min_rank_pref),
        "score_region_cols": {k: list(v) for k, v in score_region_cols.items()},
        "rank_region_cols": None if rank_region_cols is None else {k: list(v) for k, v in rank_region_cols.items()},
    }

    return out.sort_values(
        ["is_svg_score_primary", "category", "score_primary_score", "score_gap", "score_best_mean"],
        ascending=[False, True, False, False, False],
    )


def classify_scale_specific_svg_moran(
    res_score: pd.DataFrame,
    *,
    region_cols: Dict[str, List[Any]],
    resolve_region_cols: bool = True,
    region_col_atol: float = 1e-4,
    fillna_score_floor: bool = True,
    min_score_mean: float = 0.25,
    min_score_gap: float = 0.15,
    min_score_min: float = 0.0,
    max_score_std: Optional[float] = 0.20,
    min_score_pct: float = 0.75,
) -> pd.DataFrame:
    """
    Moran-score-based scale-specific SVG selection.

    This uses Moran's I trajectory only. It intentionally does not use rank
    information, so it is suitable as the score-based counterpart to
    ``classify_scale_specific_svg_rank``.
    """
    return classify_scale_specific_svg_score_primary(
        res_score,
        res_rank=None,
        region_cols=region_cols,
        resolve_region_cols=resolve_region_cols,
        region_col_atol=region_col_atol,
        fillna_score_floor=fillna_score_floor,
        min_score_mean=min_score_mean,
        min_score_gap=min_score_gap,
        min_score_min=min_score_min,
        max_score_std=max_score_std,
        min_score_pct=min_score_pct,
        min_rank_pref=None,
    )


def _rank_preference_table(rank_mat: pd.DataFrame, fillna_worst: bool = True) -> pd.DataFrame:
    """Convert ranks where lower is better to preference scores in [0, 1]."""
    work = rank_mat.apply(pd.to_numeric, errors="coerce")
    if fillna_worst:
        work = fill_missing_ranks_with_worst(work)
    col_max = work.max(axis=0, skipna=True)
    denom = col_max - 1.0
    denom = denom.where(np.isfinite(denom) & (denom > 0), 1.0)
    pref = 1.0 - work.subtract(1.0).divide(denom, axis=1)
    pref = pref.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.0)
    return pref.fillna(0.0)


def _score_percentile_table(score_mat: pd.DataFrame, fillna_floor: bool = True) -> pd.DataFrame:
    """Convert per-resolution scores to within-column percentiles where higher is better."""
    work = score_mat.apply(pd.to_numeric, errors="coerce")
    pct = work.rank(axis=0, method="average", pct=True, ascending=True)
    if fillna_floor:
        pct = pct.fillna(0.0)
    return pct.clip(lower=0.0, upper=1.0)


def classify_scale_specific_svg_hybrid(
    res_rank: pd.DataFrame,
    res_score: Optional[pd.DataFrame] = None,
    *,
    region_cols: Dict[str, List[Any]],
    fillna_worst: bool = True,
    fillna_score_floor: bool = True,
    resolve_region_cols: bool = True,
    region_col_atol: float = 1e-4,
    min_rank_pref: float = 0.85,
    min_rank_gap: float = 0.15,
    min_score_pct: float = 0.70,
    min_moran_score: float = 0.0,
    max_rank_std: Optional[float] = None,
    score_weight: float = 0.35,
) -> pd.DataFrame:
    """
    Select scale-specific SVGs using rank as the primary signal and Moran score as support.

    The main selection statistic is the within-resolution rank percentile
    preference, so genes are compared relative to other genes at the same scale.
    Moran's I is then used as an effect-size/support filter to avoid calling a
    gene scale-specific only because it has a good rank among uniformly weak
    scores.

    Parameters
    ----------
    res_rank
        Gene x resolution rank table where lower rank is better.
    res_score
        Optional gene x resolution Moran score table where higher score is
        better. If provided, columns should correspond to ``res_rank`` columns
        or be resolvable from the same ``region_cols`` specification.
    region_cols
        Mapping from category name (for example ``"high"``, ``"mid"``,
        ``"low"``) to resolution columns or numeric resolution values.
    min_rank_pref
        Minimum mean rank preference in the best region. ``0.85`` roughly means
        the gene is in the top 15% within the best region's resolutions.
    min_rank_gap
        Minimum difference between the best and second-best region rank
        preference.
    min_score_pct
        Minimum mean within-resolution Moran score percentile in the best
        region. Ignored if ``res_score`` is None.
    min_moran_score
        Minimum mean raw Moran score in the best region. Ignored if
        ``res_score`` is None.
    max_rank_std
        Optional maximum rank preference standard deviation within the best
        region. Leave as None for no stability cutoff.
    score_weight
        Contribution of Moran score percentile to the sortable
        ``hybrid_score``. Rank preference remains the primary selector.

    Returns
    -------
    pd.DataFrame
        One row per gene with region-level rank/score summaries, a ``category``
        column, and a boolean ``is_svg_hybrid`` column. Sorts selected genes
        first by category and hybrid score.
    """
    if res_rank is None or len(res_rank) == 0:
        return pd.DataFrame(index=getattr(res_rank, "index", None))
    if region_cols is None:
        raise ValueError("region_cols must be provided explicitly.")

    if bool(resolve_region_cols):
        rank_region_cols = _resolve_region_cols_for_table(
            res_rank,
            region_cols,
            atol=float(region_col_atol),
        )
    else:
        rank_region_cols = {
            str(region): [str(c) for c in cols if str(c) in res_rank.columns]
            for region, cols in region_cols.items()
        }
    missing_rank = [region for region, cols in rank_region_cols.items() if len(cols) == 0]
    if missing_rank:
        raise ValueError(
            f"Missing rank columns for groups: {missing_rank}. "
            f"Available columns: {list(map(str, res_rank.columns))}"
        )

    rank_cols = [c for cols in rank_region_cols.values() for c in cols]
    rank_pref = _rank_preference_table(res_rank.loc[:, rank_cols], fillna_worst=fillna_worst)

    score_region_cols = None
    score_pct = None
    score_raw = None
    if res_score is not None:
        common_index = res_rank.index.intersection(res_score.index)
        if len(common_index) == 0:
            raise ValueError("res_rank and res_score have no overlapping gene index.")
        res_rank = res_rank.loc[common_index]
        rank_pref = rank_pref.loc[common_index]
        res_score = res_score.loc[common_index]

        if bool(resolve_region_cols):
            score_region_cols = _resolve_region_cols_for_table(
                res_score,
                region_cols,
                atol=float(region_col_atol),
            )
        else:
            score_region_cols = {
                str(region): [str(c) for c in cols if str(c) in res_score.columns]
                for region, cols in region_cols.items()
            }
        missing_score = [region for region, cols in score_region_cols.items() if len(cols) == 0]
        if missing_score:
            raise ValueError(
                f"Missing score columns for groups: {missing_score}. "
                f"Available columns: {list(map(str, res_score.columns))}"
            )

        score_cols = [c for cols in score_region_cols.values() for c in cols]
        score_raw = res_score.loc[:, score_cols].apply(pd.to_numeric, errors="coerce")
        score_pct = _score_percentile_table(score_raw, fillna_floor=fillna_score_floor)

    out = pd.DataFrame(index=res_rank.index.copy())

    for region, cols in rank_region_cols.items():
        out[f"rank_pref_mean_{region}"] = rank_pref[cols].mean(axis=1)
        out[f"rank_pref_std_{region}"] = rank_pref[cols].std(axis=1, ddof=0)

    rank_mean_tbl = pd.DataFrame(
        {region: out[f"rank_pref_mean_{region}"] for region in rank_region_cols},
        index=out.index,
    )
    rank_std_tbl = pd.DataFrame(
        {region: out[f"rank_pref_std_{region}"] for region in rank_region_cols},
        index=out.index,
    )
    rank_mean_arr = rank_mean_tbl.to_numpy(dtype=np.float64, copy=False)
    sort_idx = np.argsort(-rank_mean_arr, axis=1)
    best_idx = sort_idx[:, 0]
    second_idx = sort_idx[:, 1] if rank_mean_arr.shape[1] > 1 else np.zeros(rank_mean_arr.shape[0], dtype=np.int64)
    region_names = np.asarray(rank_mean_tbl.columns, dtype=object)

    out["best_region"] = region_names[best_idx]
    out["rank_pref_best"] = rank_mean_arr[np.arange(len(out)), best_idx]
    out["rank_pref_second"] = rank_mean_arr[np.arange(len(out)), second_idx]
    out["rank_pref_gap"] = out["rank_pref_best"] - out["rank_pref_second"]
    out["rank_pref_gap_rel"] = out["rank_pref_gap"] / (np.abs(out["rank_pref_best"]) + 1e-8)
    out["rank_pref_best_std"] = rank_std_tbl.to_numpy(dtype=np.float64, copy=False)[np.arange(len(out)), best_idx]

    if score_pct is not None and score_raw is not None and score_region_cols is not None:
        for region, cols in score_region_cols.items():
            out[f"moran_pct_mean_{region}"] = score_pct[cols].mean(axis=1)
            out[f"moran_score_mean_{region}"] = score_raw[cols].mean(axis=1)

        score_pct_tbl = pd.DataFrame(
            {region: out[f"moran_pct_mean_{region}"] for region in rank_region_cols},
            index=out.index,
        )
        score_raw_tbl = pd.DataFrame(
            {region: out[f"moran_score_mean_{region}"] for region in rank_region_cols},
            index=out.index,
        )
        best_region_index = out["best_region"].map({r: i for i, r in enumerate(score_pct_tbl.columns)}).to_numpy()
        out["moran_pct_best_region"] = score_pct_tbl.to_numpy(dtype=np.float64, copy=False)[np.arange(len(out)), best_region_index]
        out["moran_score_best_region"] = score_raw_tbl.to_numpy(dtype=np.float64, copy=False)[np.arange(len(out)), best_region_index]
        score_supported = (
            (out["moran_pct_best_region"] >= float(min_score_pct)) &
            (out["moran_score_best_region"] > float(min_moran_score))
        )
    else:
        out["moran_pct_best_region"] = np.nan
        out["moran_score_best_region"] = np.nan
        score_supported = pd.Series(True, index=out.index)

    rank_supported = (
        (out["rank_pref_best"] >= float(min_rank_pref)) &
        (out["rank_pref_gap"] >= float(min_rank_gap))
    )
    if max_rank_std is not None:
        rank_supported &= out["rank_pref_best_std"] <= float(max_rank_std)

    out["rank_supported"] = rank_supported
    out["score_supported"] = score_supported
    out["is_svg_hybrid"] = out["rank_supported"] & out["score_supported"]

    sw = float(np.clip(score_weight, 0.0, 1.0))
    score_component = out["moran_pct_best_region"].fillna(0.0)
    if score_pct is None:
        score_component = out["rank_pref_best"]
    out["hybrid_score"] = (
        (1.0 - sw) * out["rank_pref_best"]
        + sw * score_component
        + 0.25 * out["rank_pref_gap"]
    )
    out["category"] = np.where(out["is_svg_hybrid"], out["best_region"], "mixed")
    out.attrs["hybrid_thresholds"] = {
        "min_rank_pref": float(min_rank_pref),
        "min_rank_gap": float(min_rank_gap),
        "min_score_pct": float(min_score_pct),
        "min_moran_score": float(min_moran_score),
        "max_rank_std": None if max_rank_std is None else float(max_rank_std),
        "score_weight": sw,
        "rank_region_cols": {k: list(v) for k, v in rank_region_cols.items()},
        "score_region_cols": None if score_region_cols is None else {k: list(v) for k, v in score_region_cols.items()},
    }

    return out.sort_values(
        ["is_svg_hybrid", "category", "hybrid_score", "rank_pref_gap", "rank_pref_best"],
        ascending=[False, True, False, False, False],
    )


def compare_segmentation_moran_stability(
    score_trajectory: pd.DataFrame,
    *,
    spix_cols: List[str],
    grid_col: str,
    high_threshold: float = 0.30,
    min_spix_median: float = 0.30,
    min_delta_median: float = 0.20,
    min_spix_frac_high: float = 0.60,
    max_spix_iqr: float = 0.20,
    max_spike_score: float = 0.35,
    max_spix_max: float = 0.98,
    min_spix_min: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compare SPIX compactness variants against a GRID baseline using stable Moran's I.

    This is intended for same-resolution segmentation comparisons. It avoids
    selecting single-compactness spikes by using SPIX median/IQR/fraction-above
    metrics across neighboring compactness values.
    """
    if score_trajectory is None or score_trajectory.empty:
        raise ValueError("score_trajectory must be a non-empty DataFrame.")
    if not spix_cols:
        raise ValueError("spix_cols must contain at least one column.")

    missing = [c for c in list(spix_cols) + [grid_col] if c not in score_trajectory.columns]
    if missing:
        raise KeyError(f"Missing columns in score_trajectory: {missing}")

    spix = score_trajectory.loc[:, spix_cols].apply(pd.to_numeric, errors="coerce")
    grid = pd.to_numeric(score_trajectory.loc[:, grid_col], errors="coerce")

    out = pd.DataFrame(index=score_trajectory.index.copy())
    out["SPIX_mean"] = spix.mean(axis=1)
    out["SPIX_median"] = spix.median(axis=1)
    out["SPIX_min"] = spix.min(axis=1)
    out["SPIX_max"] = spix.max(axis=1)
    out["SPIX_std"] = spix.std(axis=1, ddof=0)
    out["SPIX_q25"] = spix.quantile(0.25, axis=1)
    out["SPIX_q75"] = spix.quantile(0.75, axis=1)
    out["SPIX_iqr"] = out["SPIX_q75"] - out["SPIX_q25"]
    out["GRID_I"] = grid
    out["delta_mean"] = out["SPIX_mean"] - out["GRID_I"]
    out["delta_median"] = out["SPIX_median"] - out["GRID_I"]
    out[f"SPIX_n_ge_{float(high_threshold):g}"] = (spix >= float(high_threshold)).sum(axis=1)
    out[f"SPIX_frac_ge_{float(high_threshold):g}"] = (spix >= float(high_threshold)).mean(axis=1)
    out["spike_score"] = out["SPIX_max"] - out["SPIX_median"]

    is_stable = (
        (out["SPIX_median"] >= float(min_spix_median)) &
        (out["delta_median"] >= float(min_delta_median)) &
        (out[f"SPIX_frac_ge_{float(high_threshold):g}"] >= float(min_spix_frac_high)) &
        (out["SPIX_iqr"] <= float(max_spix_iqr)) &
        (out["spike_score"] <= float(max_spike_score)) &
        (out["SPIX_max"] < float(max_spix_max))
    )
    if min_spix_min is not None:
        is_stable &= out["SPIX_min"] >= float(min_spix_min)

    out["is_stable_spix_moran"] = is_stable
    out["stable_moran_score"] = (
        out["delta_median"]
        + 0.50 * out["SPIX_median"]
        + 0.20 * out[f"SPIX_frac_ge_{float(high_threshold):g}"]
        - 0.50 * out["SPIX_iqr"]
        - 0.50 * out["spike_score"]
    )
    out.attrs["stability_thresholds"] = {
        "spix_cols": list(spix_cols),
        "grid_col": str(grid_col),
        "high_threshold": float(high_threshold),
        "min_spix_median": float(min_spix_median),
        "min_delta_median": float(min_delta_median),
        "min_spix_frac_high": float(min_spix_frac_high),
        "max_spix_iqr": float(max_spix_iqr),
        "max_spike_score": float(max_spike_score),
        "max_spix_max": float(max_spix_max),
        "min_spix_min": None if min_spix_min is None else float(min_spix_min),
    }
    return out.sort_values(
        ["is_stable_spix_moran", "stable_moran_score", "delta_median", "SPIX_median"],
        ascending=[False, False, False, False],
    )


def summarize_segmentation_rank_robustness(
    rank_trajectory: pd.DataFrame,
    *,
    spix_cols: List[str],
    grid_col: str,
    top_ks: Tuple[int, ...] = (50, 100, 200),
    allow_missing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize rank robustness across SPIX compactness variants against GRID.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(gene_rank_summary, topk_summary)``. Lower ranks are assumed better.
    """
    if rank_trajectory is None or rank_trajectory.empty:
        raise ValueError("rank_trajectory must be a non-empty DataFrame.")
    if not spix_cols:
        raise ValueError("spix_cols must contain at least one column.")

    spix_cols_in = [c for c in spix_cols if c in rank_trajectory.columns]
    missing_spix = [c for c in spix_cols if c not in rank_trajectory.columns]
    if grid_col not in rank_trajectory.columns:
        raise KeyError(f"Missing GRID column in rank_trajectory: {grid_col}")
    if missing_spix and not bool(allow_missing):
        raise KeyError(f"Missing SPIX columns in rank_trajectory: {missing_spix}")
    if not spix_cols_in:
        raise KeyError(
            "None of spix_cols were found in rank_trajectory. "
            f"Requested={list(spix_cols)}; available={list(rank_trajectory.columns)}"
        )

    spix = rank_trajectory.loc[:, spix_cols_in].apply(pd.to_numeric, errors="coerce")
    grid = pd.to_numeric(rank_trajectory.loc[:, grid_col], errors="coerce")

    gene_summary = pd.DataFrame(index=rank_trajectory.index.copy())
    gene_summary["SPIX_mean_rank"] = spix.mean(axis=1)
    gene_summary["SPIX_median_rank"] = spix.median(axis=1)
    gene_summary["SPIX_min_rank"] = spix.min(axis=1)
    gene_summary["SPIX_std_rank"] = spix.std(axis=1, ddof=0)
    gene_summary["GRID_rank"] = grid
    gene_summary["rank_delta_GRID_minus_SPIX"] = gene_summary["GRID_rank"] - gene_summary["SPIX_median_rank"]
    gene_summary["rank_robustness_score"] = (
        gene_summary["rank_delta_GRID_minus_SPIX"]
        - 0.25 * gene_summary["SPIX_std_rank"].fillna(0.0)
    )
    gene_summary = gene_summary.sort_values(
        ["rank_robustness_score", "rank_delta_GRID_minus_SPIX", "SPIX_median_rank"],
        ascending=[False, False, True],
    )

    rows: list[dict[str, Any]] = []
    valid_top_ks = [int(k) for k in top_ks if int(k) > 0]
    for k in valid_top_ks:
        spix_sets = []
        for col in spix_cols_in:
            s = spix[col].dropna().sort_values(ascending=True).head(k)
            spix_sets.append(set(map(str, s.index)))
        grid_top = set(map(str, grid.dropna().sort_values(ascending=True).head(k).index))

        if spix_sets:
            spix_union = set().union(*spix_sets)
            spix_intersection = set(spix_sets[0]).intersection(*spix_sets[1:]) if len(spix_sets) > 1 else set(spix_sets[0])
        else:
            spix_union = set()
            spix_intersection = set()

        pairwise = []
        for i in range(len(spix_sets)):
            for j in range(i + 1, len(spix_sets)):
                denom = len(spix_sets[i] | spix_sets[j])
                pairwise.append(len(spix_sets[i] & spix_sets[j]) / denom if denom else np.nan)

        denom_union_grid = len(spix_union | grid_top)
        denom_inter_grid = len(spix_intersection | grid_top)
        rows.append({
            "top_k": int(k),
            "n_spix_cols": int(len(spix_cols_in)),
            "n_missing_spix_cols": int(len(missing_spix)),
            "spix_union_n": int(len(spix_union)),
            "spix_intersection_n": int(len(spix_intersection)),
            "grid_top_n": int(len(grid_top)),
            "spix_union_grid_overlap_n": int(len(spix_union & grid_top)),
            "spix_intersection_grid_overlap_n": int(len(spix_intersection & grid_top)),
            "spix_union_grid_jaccard": float(len(spix_union & grid_top) / denom_union_grid) if denom_union_grid else np.nan,
            "spix_intersection_grid_jaccard": float(len(spix_intersection & grid_top) / denom_inter_grid) if denom_inter_grid else np.nan,
            "spix_pairwise_jaccard_mean": float(np.nanmean(pairwise)) if pairwise else np.nan,
        })

    topk_summary = pd.DataFrame(rows)
    gene_summary.attrs["spix_cols"] = list(spix_cols_in)
    gene_summary.attrs["missing_spix_cols"] = list(missing_spix)
    gene_summary.attrs["grid_col"] = str(grid_col)
    topk_summary.attrs["spix_cols"] = list(spix_cols_in)
    topk_summary.attrs["missing_spix_cols"] = list(missing_spix)
    topk_summary.attrs["grid_col"] = str(grid_col)
    return gene_summary, topk_summary


def top_scale_genes(
    scale_df: pd.DataFrame,
    category: str,
    n: int = 15,
    *,
    category_col: str = "category",
    sort_by: str | List[str] | None = None,
    selected_only: bool = True,
) -> pd.DataFrame:
    """Return the top genes for one scale category from a scale-specific result table.

    The function supports rank-based outputs
    (``specificity_score``, ``gap_abs``, ``mean_<category>``) and Moran-based
    outputs (``score_primary_score``, ``score_gap``, ``score_best_mean``). If
    ``sort_by`` is omitted, an appropriate ordering is inferred from available
    columns.
    """
    if scale_df is None or len(scale_df) == 0:
        return pd.DataFrame(index=getattr(scale_df, "index", None))
    if category_col not in scale_df.columns:
        raise KeyError(f"'{category_col}' is not present in scale_df.columns")

    out = scale_df[scale_df[category_col].astype(str) == str(category)].copy()
    if selected_only:
        for flag_col in ("is_svg_rank", "is_svg_score_primary", "is_svg_hybrid"):
            if flag_col in out.columns:
                out = out[out[flag_col].astype(bool)]
                break

    if sort_by is None:
        if "score_primary_score" in out.columns:
            sort_cols = ["score_primary_score", "score_gap", "score_best_mean"]
            ascending = [False, False, False]
        elif "hybrid_score" in out.columns:
            sort_cols = ["hybrid_score", "rank_pref_gap", "rank_pref_best"]
            ascending = [False, False, False]
        elif {"specificity_score", "gap_abs"}.issubset(out.columns):
            sort_cols = ["specificity_score", "gap_abs"]
            ascending = [False, False]
            if "best_resolution_margin" in out.columns:
                sort_cols.append("best_resolution_margin")
                ascending.append(False)
            mean_col = f"mean_{category}"
            if mean_col in out.columns:
                sort_cols.append(mean_col)
                ascending.append(True)
        elif "best_resolution_score" in out.columns:
            sort_cols = ["best_resolution_score"]
            ascending = [False]
            if "best_resolution_margin" in out.columns:
                sort_cols.append("best_resolution_margin")
                ascending.append(False)
        else:
            sort_cols = []
            ascending = []
    else:
        sort_cols = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        ascending = [False] * len(sort_cols)

    sort_cols = [c for c in sort_cols if c in out.columns]
    if sort_cols:
        ascending = ascending[:len(sort_cols)]
        out = out.sort_values(sort_cols, ascending=ascending)
    return out.head(int(n))
