import numpy as np
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from anndata import AnnData
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from typing import List
import logging
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.utils import min_max, balance_simplest, equalize_piecewise, spe_equalization, equalize_dp, equalize_adp, ecdf_eq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def equalize_image(
    adata: AnnData,
    dimensions: List[int] = [0, 1, 2],
    embedding: str = 'X_embedding',
    output: str = 'X_embedding_equalize',
    method: str = 'BalanceSimplest',
    N: int = 1,
    smax: float = 1.0,
    sleft: float = 1.0,
    sright: float = 1.0,
    lambda_: float = 0.1,
    up: float = 100.0,
    down: float = 10.0,
    verbose: bool = True,
    n_jobs: int = 1,
    implementation: str = "auto",  # 'auto'|'vectorized'|'legacy'
) -> AnnData:
    """
    Equalize histogram of embeddings.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    dimensions : list of int, optional (default=[0, 1, 2])
        List of embedding dimensions to equalize.
    embedding : str, optional (default='X_embedding')
        Key in adata.obsm where the embeddings are stored.
    output : str, optional (default='X_embedding_equalize')
        Key in adata.obsm where the equalized embedding will be stored.
    method : str, optional (default='BalanceSimplest')
        Equalization method: 'BalanceSimplest', 'EqualizePiecewise', 'SPE', 
        'EqualizeDP', 'EqualizeADP', 'ECDF', 'histogram', 'adaptive'.
    N : int, optional (default=1)
        Number of segments for EqualizePiecewise.
    smax : float, optional (default=1.0)
        Upper limit for contrast stretching in EqualizePiecewise.
    sleft : float, optional (default=1.0)
        Percentage of pixels to saturate on the left side for BalanceSimplest.
    sright : float, optional (default=1.0)
        Percentage of pixels to saturate on the right side for BalanceSimplest.
    lambda_ : float, optional (default=0.1)
        Strength of background correction for SPE.
    up : float, optional (default=100.0)
        Upper color value threshold for EqualizeDP.
    down : float, optional (default=10.0)
        Lower color value threshold for EqualizeDP.
    verbose : bool, optional (default=True)
        Whether to display progress messages.
    n_jobs : int, optional (default=1)
        Number of parallel jobs to run.
    
    Returns
    -------
    AnnData
        The updated AnnData object with equalized embeddings.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")
    
    if verbose:
        print("Starting equalization...")
    
    impl = (implementation or "auto").lower()
    if impl not in {"auto", "vectorized", "legacy"}:
        raise ValueError("implementation must be one of {'auto','vectorized','legacy'}.")

    embedding_full = np.asarray(adata.obsm[embedding]).copy()
    embeddings = embedding_full[:, dimensions].copy()

    # Fast path: fully vectorized BalanceSimplest across dimensions
    if method == "BalanceSimplest" and impl in {"auto", "vectorized"}:
        if verbose:
            print(f"Equalizing {len(dimensions)} dims using vectorized '{method}'")

        # Match legacy numeric behavior: percentiles in float64, clip to original dtype,
        # then min-max normalize with float64 denom before casting back to embeddings dtype.
        data = np.asarray(embeddings)
        lower = np.percentile(data, sleft, axis=0)
        upper = np.percentile(data, 100.0 - sright, axis=0)
        balanced = np.clip(data, lower, upper)
        vmin = balanced.min(axis=0)
        vmax = balanced.max(axis=0)
        denom = (vmax - vmin) + 1e-10
        out = (balanced - vmin) / denom
        embeddings[:] = out.astype(embeddings.dtype, copy=False)

        embedding_out = embedding_full.copy()
        embedding_out[:, dimensions] = embeddings
        adata.obsm[output] = embedding_out
        if verbose:
            print("Logging changes to AnnData.uns['equalize_image_log']")
        adata.uns['equalize_image_log'] = {
            'method': method,
            'parameters': {
                'N': N,
                'smax': smax,
                'sleft': sleft,
                'sright': sright,
                'lambda_': lambda_,
                'up': up,
                'down': down
            },
            'dimensions': dimensions,
            'embedding': embedding,
            'implementation': 'vectorized',
        }
        if verbose:
            print("Histogram equalization completed.")
        return adata
    
    def process_dimension(i, dim):
        if verbose:
            print(f"Equalizing dimension {dim} using method '{method}'")
        data = embeddings[:, i]
        
        if method == 'BalanceSimplest':
            return balance_simplest(data, sleft=sleft, sright=sright)
        elif method == 'EqualizePiecewise':
            return equalize_piecewise(data, N=N, smax=smax)
        elif method == 'SPE':
            return spe_equalization(data, lambda_=lambda_)
        elif method == 'EqualizeDP':
            return equalize_dp(data, down=down, up=up)
        elif method == 'EqualizeADP':
            return equalize_adp(data)
        elif method == 'ECDF':
            return ecdf_eq(data)
        elif method == 'histogram':
            return equalize_hist(data)
        elif method == 'adaptive':
            return equalize_adapthist(data.reshape(1, -1), clip_limit=0.03).flatten()
        else:
            raise ValueError(f"Unknown equalization method '{method}'")

    if impl == "legacy":
        n_jobs = int(n_jobs)
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_dimension, range(len(dimensions)), dimensions))
    else:
        results = [process_dimension(i, dim) for i, dim in enumerate(dimensions)]
    
    for i, result in enumerate(results):
        embeddings[:, i] = result
    embedding_out = embedding_full.copy()
    embedding_out[:, dimensions] = embeddings
    adata.obsm[output] = embedding_out
    
    # Log changes
    if verbose:
        print("Logging changes to AnnData.uns['equalize_image_log']")
    adata.uns['equalize_image_log'] = {
        'method': method,
        'parameters': {
            'N': N,
            'smax': smax,
            'sleft': sleft,
            'sright': sright,
            'lambda_': lambda_,
            'up': up,
            'down': down
        },
        'dimensions': dimensions,
        'embedding': embedding
    }
    
    if verbose:
        print("Histogram equalization completed.")
    
    return adata


def _equalize_1d_for_eval(
    data: np.ndarray,
    *,
    method: str,
    N: int = 1,
    smax: float = 1.0,
    sleft: float = 1.0,
    sright: float = 1.0,
    lambda_: float = 0.1,
    up: float = 100.0,
    down: float = 10.0,
):
    if method == 'BalanceSimplest':
        return balance_simplest(data, sleft=sleft, sright=sright)
    if method == 'EqualizePiecewise':
        return equalize_piecewise(data, N=N, smax=smax)
    if method == 'SPE':
        return spe_equalization(data, lambda_=lambda_)
    if method == 'EqualizeDP':
        return equalize_dp(data, down=down, up=up)
    if method == 'EqualizeADP':
        return equalize_adp(data)
    if method == 'ECDF':
        return ecdf_eq(data)
    if method == 'histogram':
        return equalize_hist(data)
    if method == 'adaptive':
        return equalize_adapthist(data.reshape(1, -1), clip_limit=0.03).flatten()
    raise ValueError(f"Unknown equalization method '{method}'")


def _apply_equalization_matrix_for_eval(
    X: np.ndarray,
    *,
    method: str,
    N: int = 1,
    smax: float = 1.0,
    sleft: float = 1.0,
    sright: float = 1.0,
    lambda_: float = 0.1,
    up: float = 100.0,
    down: float = 10.0,
    n_jobs: int = 1,
):
    X = np.asarray(X, dtype=np.float32)
    if method == "BalanceSimplest":
        lower = np.percentile(X, sleft, axis=0)
        upper = np.percentile(X, 100.0 - sright, axis=0)
        balanced = np.clip(X, lower, upper)
        vmin = balanced.min(axis=0)
        vmax = balanced.max(axis=0)
        denom = (vmax - vmin) + 1e-10
        return ((balanced - vmin) / denom).astype(np.float32, copy=False)

    def _proc_col(j):
        return _equalize_1d_for_eval(
            X[:, j],
            method=method,
            N=N,
            smax=smax,
            sleft=sleft,
            sright=sright,
            lambda_=lambda_,
            up=up,
            down=down,
        )

    n_jobs = int(max(1, n_jobs))
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            cols = list(executor.map(_proc_col, range(X.shape[1])))
    else:
        cols = [_proc_col(j) for j in range(X.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32, copy=False)


def _resolve_equalization_eval_inputs(
    adata: AnnData,
    *,
    embedding: str,
    dimensions,
    obs_mask=None,
    coords=None,
    max_obs_eval: int | None = None,
    random_state: int = 0,
):
    X = np.asarray(adata.obsm[embedding], dtype=np.float32)
    dims = list(range(X.shape[1])) if dimensions is None else list(dimensions)
    X = X[:, dims]
    n_obs = X.shape[0]

    if coords is None:
        coords_arr = None
        try:
            if "spatial" in adata.obsm:
                c = np.asarray(adata.obsm["spatial"], dtype=np.float32)
                if c.shape[0] == n_obs:
                    coords_arr = c[:, :2]
        except Exception:
            coords_arr = None
        if coords_arr is None:
            tiles_df = adata.uns["tiles"]
            tiles_origin = tiles_df[tiles_df["origin"] == 1].copy()
            tiles_origin["barcode"] = tiles_origin["barcode"].astype(str)
            obs_idx = pd.Index(adata.obs_names.astype(str))
            tiles_aligned = tiles_origin.set_index("barcode").reindex(obs_idx)
            coords_arr = tiles_aligned[["x", "y"]].to_numpy(dtype=np.float32)
    else:
        coords_arr = np.asarray(coords, dtype=np.float32)[:, :2]

    if obs_mask is not None:
        if isinstance(obs_mask, (str, bytes)):
            mask = np.asarray(adata.obs[str(obs_mask)]).astype(bool, copy=False)
        else:
            mask = np.asarray(obs_mask).astype(bool, copy=False)
        X = X[mask]
        coords_arr = coords_arr[mask]

    if max_obs_eval is not None and int(max_obs_eval) > 0 and X.shape[0] > int(max_obs_eval):
        rng = np.random.default_rng(int(random_state))
        keep = np.sort(rng.choice(X.shape[0], size=int(max_obs_eval), replace=False))
        X = X[keep]
        coords_arr = coords_arr[keep]

    return X.astype(np.float32, copy=False), coords_arr.astype(np.float32, copy=False), dims


def _make_equalization_candidates(
    *,
    methods,
    N: int,
    smax: float,
    sleft: float,
    sright: float,
    lambda_: float,
    up: float,
    down: float,
    N_grid=None,
    smax_grid=None,
    sleft_grid=None,
    sright_grid=None,
    lambda_grid=None,
    up_grid=None,
    down_grid=None,
):
    methods = [str(m).strip() for m in methods]
    candidates = []
    for method in methods:
        if method == "BalanceSimplest":
            sleft_vals = [float(sleft)] if sleft_grid is None else sorted(set(float(x) for x in sleft_grid))
            sright_vals = [float(sright)] if sright_grid is None else sorted(set(float(x) for x in sright_grid))
            for sl in sleft_vals:
                for sr in sright_vals:
                    candidates.append({"method": method, "sleft": sl, "sright": sr, "N": int(N), "smax": float(smax), "lambda_": float(lambda_), "up": float(up), "down": float(down)})
        elif method == "EqualizePiecewise":
            N_vals = [int(N)] if N_grid is None else sorted(set(int(x) for x in N_grid))
            smax_vals = [float(smax)] if smax_grid is None else sorted(set(float(x) for x in smax_grid))
            for n in N_vals:
                for sm in smax_vals:
                    candidates.append({"method": method, "sleft": float(sleft), "sright": float(sright), "N": int(n), "smax": float(sm), "lambda_": float(lambda_), "up": float(up), "down": float(down)})
        elif method == "SPE":
            lambda_vals = [float(lambda_)] if lambda_grid is None else sorted(set(float(x) for x in lambda_grid))
            for lam in lambda_vals:
                candidates.append({"method": method, "sleft": float(sleft), "sright": float(sright), "N": int(N), "smax": float(smax), "lambda_": float(lam), "up": float(up), "down": float(down)})
        elif method == "EqualizeDP":
            up_vals = [float(up)] if up_grid is None else sorted(set(float(x) for x in up_grid))
            down_vals = [float(down)] if down_grid is None else sorted(set(float(x) for x in down_grid))
            for dn in down_vals:
                for upv in up_vals:
                    candidates.append({"method": method, "sleft": float(sleft), "sright": float(sright), "N": int(N), "smax": float(smax), "lambda_": float(lambda_), "up": float(upv), "down": float(dn)})
        else:
            candidates.append({"method": method, "sleft": float(sleft), "sright": float(sright), "N": int(N), "smax": float(smax), "lambda_": float(lambda_), "up": float(up), "down": float(down)})
    return candidates


def _finalize_equalization_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tradeoff_product"] = (
        df["contrast_gain"].astype(float)
        * df["structure_preserve"].astype(float)
        * (1.0 - np.clip(df["saturation_penalty"].astype(float), 0.0, 1.0))
        / (1.0 + np.clip(df["noise_amplification"].astype(float), 0.0, None))
    )
    return df


def _build_equalization_recommendation(best_row: dict) -> dict:
    method = str(best_row["method"])
    keys = ["method"]
    if method == "BalanceSimplest":
        keys += ["sleft", "sright"]
    elif method == "EqualizePiecewise":
        keys += ["N", "smax"]
    elif method == "SPE":
        keys += ["lambda_"]
    elif method == "EqualizeDP":
        keys += ["down", "up"]
    params = {k: best_row[k] for k in keys if k in best_row and pd.notna(best_row[k])}
    return {
        "candidate_id": int(best_row["candidate_id"]),
        "params": params,
        "summary": ", ".join(f"{k}={v}" for k, v in params.items()),
        "reason": (
            f"Selected candidate {int(best_row['candidate_id'])} with "
            f"contrast_gain={float(best_row.get('contrast_gain', float('nan'))):.4g}, "
            f"structure_preserve={float(best_row.get('structure_preserve', float('nan'))):.4g}, "
            f"noise_amplification={float(best_row.get('noise_amplification', float('nan'))):.4g}, "
            f"saturation_penalty={float(best_row.get('saturation_penalty', float('nan'))):.4g}, "
            f"score={float(best_row.get('score', float('nan'))):.4g}."
        ),
    }


def evaluate_equalization_sweep(
    adata: AnnData,
    *,
    embedding: str = "X_embedding",
    dimensions=None,
    methods=("BalanceSimplest",),
    obs_mask=None,
    coords=None,
    max_obs_eval: int | None = 200_000,
    edge_sample_size: int = 200_000,
    random_state: int = 0,
    N: int = 1,
    smax: float = 1.0,
    sleft: float = 1.0,
    sright: float = 1.0,
    lambda_: float = 0.1,
    up: float = 100.0,
    down: float = 10.0,
    N_grid=None,
    smax_grid=None,
    sleft_grid=None,
    sright_grid=None,
    lambda_grid=None,
    up_grid=None,
    down_grid=None,
    selection: str = "score",
    n_jobs: int | None = None,
    verbose: bool = True,
):
    log = logging.getLogger(__name__)
    if isinstance(methods, (str, bytes)):
        methods = [methods]
    methods = [str(m).strip() for m in methods]
    valid_methods = {"BalanceSimplest", "EqualizePiecewise", "SPE", "EqualizeDP", "EqualizeADP", "ECDF", "histogram", "adaptive"}
    if not methods or any(m not in valid_methods for m in methods):
        raise ValueError(f"methods must be a non-empty subset of {sorted(valid_methods)}.")
    selection_mode = str(selection).strip().lower()
    if selection_mode not in {"score", "product"}:
        raise ValueError("selection must be one of {'score','product'}.")

    X0, coords_arr, dims_used = _resolve_equalization_eval_inputs(
        adata,
        embedding=embedding,
        dimensions=dimensions,
        obs_mask=obs_mask,
        coords=coords,
        max_obs_eval=max_obs_eval,
        random_state=random_state,
    )
    if n_jobs is None or int(n_jobs) == -1:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = int(max(1, n_jobs))
    if verbose:
        log.info(
            "Equalization sweep inputs: n_obs=%d | n_dims=%d | max_obs_eval=%s",
            int(X0.shape[0]),
            int(X0.shape[1]),
            str(max_obs_eval),
        )

    tree = cKDTree(coords_arr, balanced_tree=True, compact_nodes=True)
    _, idx_nn = tree.query(coords_arr, k=min(2, coords_arr.shape[0]), workers=max(1, n_jobs))
    if np.ndim(idx_nn) == 1:
        src = np.arange(coords_arr.shape[0], dtype=np.int32)
        dst = idx_nn.astype(np.int32, copy=False)
    else:
        src = np.arange(coords_arr.shape[0], dtype=np.int32)
        dst = idx_nn[:, 1].astype(np.int32, copy=False)
    if edge_sample_size is not None and int(edge_sample_size) > 0 and src.size > int(edge_sample_size):
        rng = np.random.default_rng(int(random_state))
        keep = np.sort(rng.choice(src.size, size=int(edge_sample_size), replace=False))
        src = src[keep]
        dst = dst[keep]

    d0 = np.abs(X0[src] - X0[dst]).astype(np.float32, copy=False)
    d0_all = d0.mean(axis=0)
    q_low = np.quantile(d0, 0.2, axis=0)
    q_high = np.quantile(d0, 0.8, axis=0)
    low_mask = d0 <= q_low[None, :]
    high_mask = d0 >= q_high[None, :]
    eps = 1e-12

    candidates = _make_equalization_candidates(
        methods=methods,
        N=N,
        smax=smax,
        sleft=sleft,
        sright=sright,
        lambda_=lambda_,
        up=up,
        down=down,
        N_grid=N_grid,
        smax_grid=smax_grid,
        sleft_grid=sleft_grid,
        sright_grid=sright_grid,
        lambda_grid=lambda_grid,
        up_grid=up_grid,
        down_grid=down_grid,
    )

    def _eval_one(item):
        cand_id, cand = item
        X1 = _apply_equalization_matrix_for_eval(
            X0,
            method=cand["method"],
            N=int(cand["N"]),
            smax=float(cand["smax"]),
            sleft=float(cand["sleft"]),
            sright=float(cand["sright"]),
            lambda_=float(cand["lambda_"]),
            up=float(cand["up"]),
            down=float(cand["down"]),
            n_jobs=1,
        )
        d1 = np.abs(X1[src] - X1[dst]).astype(np.float32, copy=False)
        d1_all = d1.mean(axis=0)
        low_ratio = []
        high_ratio = []
        all_ratio = []
        for j in range(X0.shape[1]):
            d0_low = float(d0[:, j][low_mask[:, j]].mean()) if bool(low_mask[:, j].any()) else float(d0[:, j].mean())
            d1_low = float(d1[:, j][low_mask[:, j]].mean()) if bool(low_mask[:, j].any()) else float(d1[:, j].mean())
            d0_hi = float(d0[:, j][high_mask[:, j]].mean()) if bool(high_mask[:, j].any()) else float(d0[:, j].mean())
            d1_hi = float(d1[:, j][high_mask[:, j]].mean()) if bool(high_mask[:, j].any()) else float(d1[:, j].mean())
            low_ratio.append(d1_low / max(eps, d0_low, 0.1 * float(d0_all[j])))
            high_ratio.append(d1_hi / max(eps, d0_hi, 0.25 * float(d0_all[j])))
            all_ratio.append(float(d1_all[j]) / max(eps, float(d0_all[j]), 0.25 * float(d0_all[j])))

        low_ratio_m = float(np.mean(low_ratio))
        high_ratio_m = float(np.mean(high_ratio))
        all_ratio_m = float(np.mean(all_ratio))
        contrast_gain = float(max(0.0, high_ratio_m - 1.0))
        structure_preserve = float(np.exp(-abs(np.log(max(eps, all_ratio_m)))))
        noise_amplification = float(max(0.0, low_ratio_m - 1.0))
        saturation_penalty = float(np.mean((X1 <= 0.01) | (X1 >= 0.99)))
        score = float(
            contrast_gain * structure_preserve / (1.0 + noise_amplification + 2.0 * saturation_penalty)
        )
        return {
            "candidate_id": int(cand_id),
            **cand,
            "contrast_gain": contrast_gain,
            "structure_preserve": structure_preserve,
            "noise_amplification": noise_amplification,
            "saturation_penalty": saturation_penalty,
            "ratio_low": low_ratio_m,
            "ratio_high": high_ratio_m,
            "ratio_all": all_ratio_m,
            "score": score,
        }

    if verbose:
        log.info("Equalization sweep candidate evaluation: candidates=%d | jobs=%d", len(candidates), n_jobs)
    if n_jobs > 1 and len(candidates) > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(candidates))) as executor:
            results = list(executor.map(_eval_one, list(enumerate(candidates))))
    else:
        results = [_eval_one(item) for item in enumerate(candidates)]

    df = _finalize_equalization_scores(pd.DataFrame(results))
    best_col = "tradeoff_product" if selection_mode == "product" else "score"
    best_id = int(df.sort_values([best_col], ascending=False, kind="mergesort").iloc[0]["candidate_id"])
    df["is_best"] = df["candidate_id"] == best_id
    df = df.sort_values(["is_best", best_col], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    best_row = df.iloc[0].to_dict()

    return {
        "best": _build_equalization_recommendation(best_row)["params"],
        "candidates": df,
        "dimensions": dims_used,
        "selection_result": {
            "selected_candidate_id": int(best_id),
            "selected": best_row,
            "selection_mode": selection_mode,
        },
        "recommendation": _build_equalization_recommendation(best_row),
    }


def plot_equalization_sweep(
    selection_result,
    *,
    param_x: str | None = None,
    show: bool = True,
    figsize=(12, 8),
):
    if not isinstance(selection_result, dict) or "candidates" not in selection_result:
        raise ValueError("selection_result must be a dict returned by evaluate_equalization_sweep.")
    df = selection_result["candidates"]
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        raise ValueError("selection_result['candidates'] must be a non-empty DataFrame.")

    candidate_cols = ["sleft", "sright", "N", "smax", "lambda_", "down", "up"]
    if param_x is None:
        varying = [c for c in candidate_cols if c in df.columns and pd.Series(df[c]).nunique(dropna=False) > 1]
        if not varying:
            raise ValueError("No varying equalization parameter found to plot on x-axis.")
        param_x = varying[0]
    plot_df = df.sort_values(param_x, kind="mergesort").reset_index(drop=True)
    selected_mask = plot_df.get("is_best", pd.Series(False, index=plot_df.index)).astype(bool)
    varying = [c for c in candidate_cols if c in df.columns and pd.Series(df[c]).nunique(dropna=False) > 1]
    multi_param = len(varying) > 1
    selection_meta = selection_result.get("selection_result", {})
    recommendation = selection_result.get("recommendation", {})
    metrics = [
        ("score", "score"),
        ("contrast_gain", "contrast_gain"),
        ("structure_preserve", "structure_preserve"),
        ("tradeoff_product", "tradeoff_product"),
        ("noise_amplification", "noise_amplification"),
        ("saturation_penalty", "saturation_penalty"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    x = plot_df[param_x].to_numpy(dtype=float)
    for ax, (col, title) in zip(axes, metrics):
        if col not in plot_df.columns:
            ax.axis("off")
            continue
        y = plot_df[col].to_numpy(dtype=float)
        if multi_param:
            ax.scatter(x, y, s=24, alpha=0.7)
            agg = (
                plot_df[[param_x, col]]
                .groupby(param_x, as_index=False)
                .median()
                .sort_values(param_x, kind="mergesort")
            )
            ax.plot(
                agg[param_x].to_numpy(dtype=float),
                agg[col].to_numpy(dtype=float),
                linewidth=1.5,
                color="black",
                alpha=0.8,
            )
        else:
            ax.plot(x, y, marker="o", linewidth=1.5)
        if np.any(selected_mask):
            ax.scatter(
                plot_df.loc[selected_mask, param_x].to_numpy(dtype=float),
                plot_df.loc[selected_mask, col].to_numpy(dtype=float),
                c="black",
                s=40,
                zorder=5,
            )
        ax.set_title(title)
        ax.set_xlabel(param_x)
    fig.suptitle(
        (
            f"Equalization Sweep | best={recommendation['params']} | aggregated over {', '.join(v for v in varying if v != param_x)}"
            if multi_param and recommendation is not None
            else (f"Equalization Sweep | best={recommendation['params']}" if recommendation is not None else "Equalization Sweep")
        ),
        fontsize=14,
    )
    if recommendation:
        msg = f"selected: {recommendation.get('summary', '')}"
        sel_mode = selection_meta.get("selection_mode", None)
        if sel_mode:
            msg += f"\nrule: {sel_mode}"
        fig.text(
            0.99,
            0.01,
            msg,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
    fig.tight_layout()
    if show:
        plt.show()
    return {"fig": fig, "axes": axes, "param_x": param_x}


def plot_equalization_grid(
    selection_result,
    *,
    x: str,
    y: str,
    facet: str | None = None,
    value: str = "tradeoff_product",
    cmap: str = "viridis",
    show: bool = True,
    figsize=None,
):
    if not isinstance(selection_result, dict) or "candidates" not in selection_result:
        raise ValueError("selection_result must be a dict returned by evaluate_equalization_sweep.")
    df = selection_result["candidates"]
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        raise ValueError("selection_result['candidates'] must be a non-empty DataFrame.")
    for col in [x, y, value]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in candidates DataFrame.")
    if facet is not None and facet not in df.columns:
        raise ValueError(f"Facet column '{facet}' not found in candidates DataFrame.")

    facet_values = [None] if facet is None else list(pd.unique(df[facet]))
    n_panels = len(facet_values)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (4.5 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()
    selected = selection_result.get("selection_result", {}).get("selected", None)
    im_ref = None

    for i, facet_val in enumerate(facet_values):
        ax = axes_flat[i]
        sub = df if facet is None else df[df[facet] == facet_val]
        pivot = pd.pivot_table(
            sub,
            index=y,
            columns=x,
            values=value,
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap=cmap)
        im_ref = im
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(v) for v in pivot.index])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{facet}={facet_val}" if facet is not None else value)

        if selected is not None:
            sel_ok = True
            if facet is not None:
                sel_ok = selected.get(facet) == facet_val
            if sel_ok and selected.get(x) in pivot.columns and selected.get(y) in pivot.index:
                x_idx = list(pivot.columns).index(selected.get(x))
                y_idx = list(pivot.index).index(selected.get(y))
                ax.scatter([x_idx], [y_idx], s=120, facecolors="none", edgecolors="white", linewidths=2.0)
                ax.scatter([x_idx], [y_idx], s=40, c="black", zorder=5)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")
    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes_flat[:n_panels], shrink=0.9, label=value)
    rec = selection_result.get("recommendation", None)
    fig.suptitle(
        f"Equalization Grid | best={rec['params']}" if rec is not None else "Equalization Grid",
        fontsize=13,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return {"fig": fig, "axes": axes, "x": x, "y": y, "facet": facet, "value": value}


def fast_select_equalization(
    adata: AnnData,
    *,
    embedding: str = "X_embedding",
    dimensions=None,
    methods=("BalanceSimplest",),
    obs_mask=None,
    coords=None,
    max_obs_eval: int | None = 100_000,
    edge_sample_size: int = 100_000,
    random_state: int = 0,
    selection: str = "score",
    n_jobs: int | None = None,
    verbose: bool = True,
    **kwargs,
):
    return evaluate_equalization_sweep(
        adata,
        embedding=embedding,
        dimensions=dimensions,
        methods=methods,
        obs_mask=obs_mask,
        coords=coords,
        max_obs_eval=max_obs_eval,
        edge_sample_size=edge_sample_size,
        random_state=random_state,
        selection=selection,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs,
    )
