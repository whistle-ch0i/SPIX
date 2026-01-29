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


def _rank_min_desc(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(-values, kind="mergesort")
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
            if moran is None or moran.empty or "I" not in moran.columns:
                mode = str(pseudo_bulk_kwargs.get("mode", "moran")).lower()
                results_key = f"{mode}C" if mode == "geary" else f"{mode}I"
                full = new_adata.uns.get(results_key, None)
                if isinstance(full, pd.DataFrame) and (not full.empty) and ("I" in full.columns or "C" in full.columns):
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
                    moran = _compute_moranI_fallback(new_adata.X, W, new_adata.var_names)
            del new_adata

            i_vals = moran["I"].to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(i_vals)
            if not np.all(valid):
                i_vals = i_vals[valid]
                moran_index = moran.index[valid]
            else:
                moran_index = moran.index

            ranks = _rank_min_desc(i_vals)
            series = pd.Series(ranks, index=moran_index, name=f"rank_{sid}")
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
        if moran is None or moran.empty or "I" not in moran.columns:
            mode = str(pseudo_bulk_kwargs.get("mode", "moran")).lower()
            results_key = f"{mode}C" if mode == "geary" else f"{mode}I"
            full = new_adata.uns.get(results_key, None)
            if isinstance(full, pd.DataFrame) and (not full.empty) and ("I" in full.columns or "C" in full.columns):
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
                moran = _compute_moranI_fallback(new_adata.X, W, new_adata.var_names)
        del new_adata

        i_vals = moran["I"].to_numpy(dtype=np.float64, copy=False)
        valid = np.isfinite(i_vals)
        if not np.all(valid):
            i_vals = i_vals[valid]
            moran_index = moran.index[valid]
        else:
            moran_index = moran.index

        ranks = _rank_min_desc(i_vals)
        series = pd.Series(ranks, index=moran_index, name=f"rank_{sid}")
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
