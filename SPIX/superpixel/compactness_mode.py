from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_COMPACTNESS_SEARCH_MULTIPLIERS: tuple[float, ...] = (0.4, 0.67, 1.0, 1.5, 2.25)
DEFAULT_AUTO_COMPACTNESS_CANDIDATES: tuple[float, ...] = (
    0.003,
    0.005,
    0.008,
    0.012,
    0.02,
    0.03,
    0.05,
    0.08,
    0.12,
    0.2,
    0.35,
    0.6,
)


def _sanitize_labels(labels: pd.Series | np.ndarray | list[object]) -> pd.Series:
    series = pd.Series(labels, dtype="object")
    text = series.astype(str)
    return text.where(~text.isin({"nan", "None", "<NA>"}), other=np.nan)


def _segment_aspect_ratio(coords: np.ndarray) -> float:
    pts = np.asarray(coords, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return 1.0
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-12))
    return float(np.sqrt(eigvals[-1] / eigvals[0]))


def _candidate_metrics(
    *,
    labels: pd.Series | np.ndarray | list[object],
    spatial_coords: np.ndarray,
    embeddings: np.ndarray | None,
    embedding_dims: int,
    spatial_knn: np.ndarray | None = None,
) -> dict[str, float]:
    seg = _sanitize_labels(labels)
    coords_arr = np.asarray(spatial_coords, dtype=float)
    valid = seg.notna().to_numpy() & np.isfinite(coords_arr).all(axis=1)
    if not np.any(valid):
        return {
            "n_segments": 0.0,
            "embedding_var": float("nan"),
            "mean_log_aspect": float("nan"),
            "size_cv": float("nan"),
            "neighbor_disagreement": float("nan"),
        }

    seg_valid = seg.iloc[np.flatnonzero(valid)]
    coords_valid = coords_arr[valid]
    codes = pd.Categorical(seg_valid).codes
    if codes.size == 0:
        return {
            "n_segments": 0.0,
            "embedding_var": float("nan"),
            "mean_log_aspect": float("nan"),
            "size_cv": float("nan"),
            "neighbor_disagreement": float("nan"),
        }

    emb = None
    if embeddings is not None:
        emb = np.asarray(embeddings, dtype=float)
        emb = emb[valid]
        if emb.ndim == 1:
            emb = emb[:, None]
        if emb.shape[1] > int(embedding_dims):
            emb = emb[:, : int(embedding_dims)]

    groups = pd.DataFrame({"code": codes}).groupby("code", sort=False).groups
    counts = []
    emb_rows = []
    aspect_rows = []
    for idx in groups.values():
        take = np.asarray(list(idx), dtype=np.int64)
        n = int(take.size)
        if n == 0:
            continue
        counts.append(float(n))
        if emb is not None:
            arr = emb[take]
            emb_rows.append(0.0 if n < 2 else float(np.nanmean(np.var(arr, axis=0, ddof=0))))
        aspect_rows.append(float(np.log(max(_segment_aspect_ratio(coords_valid[take]), 1.0))))

    counts_arr = np.asarray(counts, dtype=float)
    emb_var = float("nan")
    if emb_rows:
        emb_var = float(np.average(np.asarray(emb_rows, dtype=float), weights=counts_arr))
    mean_log_aspect = float(np.average(np.asarray(aspect_rows, dtype=float), weights=counts_arr))
    size_cv = 0.0
    if counts_arr.size > 1 and float(np.mean(counts_arr)) > 0:
        size_cv = float(np.std(counts_arr) / np.mean(counts_arr))
    neighbor_disagreement = float("nan")
    if spatial_knn is not None and spatial_knn.size > 0:
        valid_idx = np.flatnonzero(valid)
        pos_map = np.full(seg.shape[0], -1, dtype=np.int64)
        pos_map[valid_idx] = np.arange(valid_idx.size, dtype=np.int64)
        neigh = pos_map[np.asarray(spatial_knn[valid_idx], dtype=np.int64)]
        neigh_ok = neigh >= 0
        if np.any(neigh_ok):
            same = np.zeros(neigh.shape, dtype=bool)
            center_codes = np.broadcast_to(codes[:, None], neigh.shape)
            same[neigh_ok] = center_codes[neigh_ok] == codes[neigh[neigh_ok]]
            neighbor_disagreement = float(1.0 - np.mean(same[neigh_ok]))
    return {
        "n_segments": float(counts_arr.size),
        "embedding_var": emb_var,
        "mean_log_aspect": mean_log_aspect,
        "size_cv": size_cv,
        "neighbor_disagreement": neighbor_disagreement,
    }


def prepare_compactness_metric_context(
    *,
    spatial_coords: np.ndarray,
    embeddings: np.ndarray | None,
    compactness_metric_sample_size: int | None = None,
) -> dict[str, Any]:
    coords_arr = np.asarray(spatial_coords, dtype=float)
    emb_arr = None if embeddings is None else np.asarray(embeddings, dtype=float)
    metric_take = None
    sample_size = None if compactness_metric_sample_size is None else int(compactness_metric_sample_size)
    if sample_size is not None and sample_size > 0 and coords_arr.shape[0] > sample_size:
        rng = np.random.default_rng(0)
        metric_take = np.sort(rng.choice(coords_arr.shape[0], size=sample_size, replace=False))
        coords_arr = coords_arr[metric_take]
        if emb_arr is not None:
            emb_arr = emb_arr[metric_take]

    finite_coords = np.isfinite(coords_arr).all(axis=1)
    spatial_knn = None
    if int(np.sum(finite_coords)) >= 2:
        valid_coords = coords_arr[finite_coords]
        k = min(6, int(valid_coords.shape[0]) - 1)
        if k >= 1:
            tree = cKDTree(valid_coords)
            _dists, neigh = tree.query(valid_coords, k=k + 1)
            knn_valid = np.asarray(neigh[:, 1:], dtype=np.int64)
            spatial_knn = np.full((coords_arr.shape[0], k), -1, dtype=np.int64)
            finite_idx = np.flatnonzero(finite_coords)
            spatial_knn[finite_idx] = finite_idx[knn_valid]
    return {
        "spatial_coords": coords_arr,
        "embeddings": emb_arr,
        "metric_take": metric_take,
        "spatial_knn": spatial_knn,
    }


def evaluate_candidate_metrics_from_labels(
    labels: pd.Series | np.ndarray | list[object],
    *,
    metric_context: dict[str, Any],
    compactness_search_dims: int,
) -> dict[str, float]:
    metric_take = metric_context.get("metric_take", None)
    labels_use = labels
    if metric_take is not None:
        labels_use = np.asarray(labels_use)[metric_take]
    return _candidate_metrics(
        labels=labels_use,
        spatial_coords=np.asarray(metric_context["spatial_coords"], dtype=float),
        embeddings=None if metric_context.get("embeddings", None) is None else np.asarray(metric_context["embeddings"], dtype=float),
        embedding_dims=int(compactness_search_dims),
        spatial_knn=metric_context.get("spatial_knn", None),
    )


def _attach_rank_scores(rows: list[dict[str, float]]) -> None:
    def _normalize_cost(metric: str) -> np.ndarray:
        values = np.asarray([float(row.get(metric, np.nan)) for row in rows], dtype=float)
        out = np.full(values.shape, np.nan, dtype=float)
        valid = np.isfinite(values)
        if int(valid.sum()) == 0:
            return out
        if int(valid.sum()) == 1:
            out[valid] = 0.0
            return out
        lo = float(np.nanmin(values[valid]))
        hi = float(np.nanmax(values[valid]))
        if not np.isfinite(hi - lo) or hi - lo <= 1e-12:
            out[valid] = 0.0
            return out
        out[valid] = (values[valid] - lo) / (hi - lo)
        return out

    fidelity_cost = _normalize_cost("embedding_var")
    regularity_components = {
        "mean_log_aspect": 0.30,
        "size_cv": 0.35,
        "neighbor_disagreement": 0.35,
    }
    regularity_cost = np.zeros(len(rows), dtype=float)
    used_weight = np.zeros(len(rows), dtype=float)
    for metric, weight in regularity_components.items():
        cost = _normalize_cost(metric)
        valid = np.isfinite(cost)
        regularity_cost[valid] += weight * cost[valid]
        used_weight[valid] += weight
    regularity_cost = np.divide(
        regularity_cost,
        np.maximum(used_weight, 1e-12),
        out=np.full(len(rows), np.nan, dtype=float),
        where=used_weight > 0,
    )

    fidelity_preservation = 1.0 - fidelity_cost
    regularity_gain = 1.0 - regularity_cost
    tradeoff_gain = np.full(len(rows), np.nan, dtype=float)
    both_valid = np.isfinite(fidelity_preservation) & np.isfinite(regularity_gain)
    tradeoff_gain[both_valid] = (
        2.0
        * fidelity_preservation[both_valid]
        * regularity_gain[both_valid]
        / np.maximum(fidelity_preservation[both_valid] + regularity_gain[both_valid], 1e-12)
    )

    for idx, row in enumerate(rows):
        row["fidelity_cost"] = None if not np.isfinite(fidelity_cost[idx]) else float(fidelity_cost[idx])
        row["fidelity_preservation"] = (
            None if not np.isfinite(fidelity_preservation[idx]) else float(fidelity_preservation[idx])
        )
        row["regularity_cost"] = None if not np.isfinite(regularity_cost[idx]) else float(regularity_cost[idx])
        row["regularity_gain"] = None if not np.isfinite(regularity_gain[idx]) else float(regularity_gain[idx])
        row["tradeoff_gain"] = None if not np.isfinite(tradeoff_gain[idx]) else float(tradeoff_gain[idx])
        if np.isfinite(tradeoff_gain[idx]):
            row["score"] = float(1.0 - tradeoff_gain[idx])
        elif np.isfinite(regularity_cost[idx]):
            row["score"] = float(regularity_cost[idx])
        else:
            row["score"] = 0.0


def _apply_compactness_regularization(
    rows: list[dict[str, float]],
    *,
    scheduled: float | None,
) -> None:
    if not rows:
        return
    compactness_values = np.asarray([float(row["compactness"]) for row in rows], dtype=float)
    lo = float(np.nanmin(compactness_values))
    hi = float(np.nanmax(compactness_values))
    span = hi - lo
    if not np.isfinite(span) or span <= 1e-12:
        for row in rows:
            row["compactness_penalty"] = 0.0
        return

    for row in rows:
        value = float(row["compactness"])
        rel = float((value - lo) / span)
        # Avoid selecting extreme range edges when the evidence is otherwise flat.
        edge_penalty = 0.0
        if rel < 0.10:
            edge_penalty += 0.02 * ((0.10 - rel) / 0.10) ** 2
        if rel > 0.90:
            edge_penalty += 0.02 * ((rel - 0.90) / 0.10) ** 2
        # Prefer staying near the scheduled center when one exists.
        schedule_penalty = 0.0
        if scheduled is not None and np.isfinite(float(scheduled)) and float(scheduled) > 0:
            schedule_penalty = 0.03 * abs(np.log(value / float(scheduled)))
        penalty = float(edge_penalty + schedule_penalty)
        row["compactness_penalty"] = penalty
        row["score"] = float(row["score"] + penalty)


def _clip_candidates(
    candidates: Sequence[float],
    *,
    compactness_min: float | None,
    compactness_max: float | None,
) -> list[float]:
    clipped: list[float] = []
    for value in candidates:
        candidate = float(value)
        if compactness_min is not None:
            candidate = max(candidate, float(compactness_min))
        if compactness_max is not None:
            candidate = min(candidate, float(compactness_max))
        clipped.append(float(candidate))
    deduped = sorted({round(value, 12): value for value in clipped if np.isfinite(value) and value > 0}.values())
    return deduped


def _evaluate_candidates(
    *,
    candidates: Sequence[float],
    segmenter: Callable[[float], pd.Series | np.ndarray | list[object]],
    embeddings: np.ndarray | None,
    spatial_coords: np.ndarray,
    compactness_search_dims: int,
    compactness_search_jobs: int = 1,
    compactness_metric_sample_size: int | None = None,
    verbose: bool = False,
    mode: str = "manual_search",
) -> list[dict[str, float]]:
    t0 = time.perf_counter()
    metric_context = prepare_compactness_metric_context(
        spatial_coords=spatial_coords,
        embeddings=embeddings,
        compactness_metric_sample_size=compactness_metric_sample_size,
    )
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    if verbose:
        print(
            f"[perf] {mode} candidate prep: "
            f"sample={t1 - t0:.3f}s | knn={t2 - t1:.3f}s | "
            f"candidates={len(candidates)} | jobs={max(1, int(compactness_search_jobs))}"
        )

    def _evaluate_one(candidate: float) -> dict[str, float]:
        if verbose:
            print(f"[perf] {mode} candidate start: compactness={float(candidate):.6g}")
        t_seg0 = time.perf_counter()
        labels = segmenter(float(candidate))
        t_seg1 = time.perf_counter()
        metrics = evaluate_candidate_metrics_from_labels(
            labels=labels,
            metric_context=metric_context,
            compactness_search_dims=int(compactness_search_dims),
        )
        t_seg2 = time.perf_counter()
        if verbose:
            print(
                f"[perf] {mode} candidate done: compactness={float(candidate):.6g} | "
                f"segment={t_seg1 - t_seg0:.3f}s | metrics={t_seg2 - t_seg1:.3f}s"
            )
        return {"compactness": float(candidate), **metrics}

    jobs = max(1, int(compactness_search_jobs))
    t3 = time.perf_counter()
    if jobs <= 1 or len(candidates) <= 1:
        rows = [_evaluate_one(float(candidate)) for candidate in candidates]
    else:
        with ThreadPoolExecutor(max_workers=min(jobs, len(candidates))) as executor:
            rows = list(executor.map(_evaluate_one, [float(candidate) for candidate in candidates]))
    t4 = time.perf_counter()
    if verbose:
        print(f"[perf] {mode} candidate execution total: {t4 - t3:.3f}s")
    _attach_rank_scores(rows)
    return rows


def _choose_best_row(
    rows: Sequence[dict[str, float]],
    *,
    scheduled: float | None,
) -> dict[str, float]:
    if scheduled is None:
        return min(rows, key=lambda row: float(row["score"]))
    return min(
        rows,
        key=lambda row: (
            float(row["score"]),
            abs(float(row["compactness"]) - float(scheduled)),
        ),
    )


def finalize_compactness_rows(
    rows: list[dict[str, float]],
    *,
    scheduled: float | None,
    mode: str,
    target_segment_um: float | None,
    compactness_search_jobs: int,
) -> tuple[float, dict[str, Any]]:
    _attach_rank_scores(rows)
    _apply_compactness_regularization(rows, scheduled=scheduled)
    best = _choose_best_row(rows, scheduled=scheduled)
    detail = {
        "mode": str(mode),
        "target_segment_um": None if target_segment_um is None else float(target_segment_um),
        "scheduled_compactness": None if scheduled is None else float(scheduled),
        "selected_compactness": float(best["compactness"]),
        "compactness_search_jobs": int(compactness_search_jobs),
        "candidates": [
            {
                "compactness": float(row["compactness"]),
                "score": float(row["score"]),
                "compactness_penalty": float(row.get("compactness_penalty", 0.0)),
                "tradeoff_gain": None if row.get("tradeoff_gain", None) is None else float(row["tradeoff_gain"]),
                "fidelity_cost": None if row.get("fidelity_cost", None) is None else float(row["fidelity_cost"]),
                "fidelity_preservation": None if row.get("fidelity_preservation", None) is None else float(row["fidelity_preservation"]),
                "regularity_cost": None if row.get("regularity_cost", None) is None else float(row["regularity_cost"]),
                "regularity_gain": None if row.get("regularity_gain", None) is None else float(row["regularity_gain"]),
                "embedding_var": None if not np.isfinite(float(row["embedding_var"])) else float(row["embedding_var"]),
                "mean_log_aspect": None if not np.isfinite(float(row["mean_log_aspect"])) else float(row["mean_log_aspect"]),
                "size_cv": None if not np.isfinite(float(row["size_cv"])) else float(row["size_cv"]),
                "neighbor_disagreement": None if not np.isfinite(float(row["neighbor_disagreement"])) else float(row["neighbor_disagreement"]),
                "n_segments": int(round(float(row["n_segments"]))),
            }
            for row in rows
        ],
    }
    return float(best["compactness"]), detail


def _refine_auto_candidates(
    *,
    seed_candidates: Sequence[float],
    best_compactness: float,
    compactness_min: float | None,
    compactness_max: float | None,
) -> list[float]:
    seeds = np.asarray(sorted(set(float(x) for x in seed_candidates)), dtype=float)
    if seeds.size == 0:
        return [float(best_compactness)]
    idx = int(np.argmin(np.abs(seeds - float(best_compactness))))
    center = float(seeds[idx])
    low = float(seeds[idx - 1]) if idx > 0 else center / 1.8
    high = float(seeds[idx + 1]) if idx + 1 < seeds.size else center * 1.8
    if not np.isfinite(low) or low <= 0:
        low = center / 1.8
    if not np.isfinite(high) or high <= low:
        high = max(center * 1.8, low * 1.2)
    refined = np.geomspace(low, high, num=7)
    refined = list(refined) + [center]
    return _clip_candidates(
        refined,
        compactness_min=compactness_min,
        compactness_max=compactness_max,
    )


def infer_compactness_range(
    *,
    embeddings: np.ndarray | None,
    spatial_coords: np.ndarray,
    target_segment_um: float | None,
    resolution: float | None,
    pitch_um: float,
    compactness_min: float | None,
    compactness_max: float | None,
    embedding_dims: int = 12,
    sample_size: int = 4096,
) -> dict[str, float | int | None]:
    coords = np.asarray(spatial_coords, dtype=float)
    valid = np.isfinite(coords).all(axis=1)
    if embeddings is not None:
        emb = np.asarray(embeddings, dtype=float)
        if emb.ndim == 1:
            emb = emb[:, None]
        valid &= np.isfinite(emb).all(axis=1)
        if emb.shape[1] > int(embedding_dims):
            emb = emb[:, : int(embedding_dims)]
    else:
        emb = None
    coords = coords[valid]
    if emb is not None:
        emb = emb[valid]
    n_points = int(coords.shape[0])
    if n_points == 0:
        lower = 0.01 if compactness_min is None else float(compactness_min)
        upper = 0.1 if compactness_max is None else float(compactness_max)
        center = float(np.sqrt(lower * upper))
        return {
            "center": center,
            "lower": lower,
            "upper": upper,
            "roughness": None,
            "spacing_cv": None,
            "sampled_points": 0,
            "estimated_segment_side_spots": None,
        }

    take_n = min(n_points, int(sample_size))
    if take_n < n_points:
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(n_points, size=take_n, replace=False))
        coords_s = coords[take]
        emb_s = None if emb is None else emb[take]
    else:
        coords_s = coords
        emb_s = emb

    spacing_cv = 0.0
    if coords_s.shape[0] >= 2:
        tree = cKDTree(coords_s)
        k = min(8, int(coords_s.shape[0]))
        dists, neigh = tree.query(coords_s, k=k)
        if k > 1:
            nn_dists = np.asarray(dists[:, 1:], dtype=float)
            mean_nn = float(np.nanmean(nn_dists))
            if np.isfinite(mean_nn) and mean_nn > 0:
                spacing_cv = float(np.nanstd(nn_dists) / mean_nn)
        else:
            neigh = np.zeros((coords_s.shape[0], 1), dtype=int)
    roughness = 1.0
    if emb_s is not None and coords_s.shape[0] >= 3:
        tree = cKDTree(coords_s)
        k = min(8, int(coords_s.shape[0]))
        _dists, neigh = tree.query(coords_s, k=k)
        if k > 1:
            nbr = emb_s[np.asarray(neigh[:, 1:], dtype=int)]
            center = emb_s[:, None, :]
            local_var = float(np.nanmean(np.var(nbr - center, axis=1, ddof=0)))
            global_var = float(np.nanmean(np.var(emb_s, axis=0, ddof=0)))
            if np.isfinite(global_var) and global_var > 1e-12:
                roughness = float(np.clip(local_var / global_var, 0.25, 4.0))

    if target_segment_um is not None and pitch_um > 0:
        segment_side_spots = max(float(target_segment_um) / float(pitch_um), 1.0)
    elif resolution is not None and resolution > 0:
        segment_side_spots = max(np.sqrt(float(n_points) / float(resolution)), 1.0)
    else:
        segment_side_spots = max(np.sqrt(float(n_points) / 128.0), 1.0)

    size_factor = float(np.clip((segment_side_spots / 25.0) ** 0.6, 0.6, 4.0))
    complexity_penalty = float(np.clip(0.85 + 0.70 * roughness + 0.40 * spacing_cv, 0.8, 4.0))
    center = float(0.035 * size_factor / complexity_penalty)
    lower = float(center / 2.2)
    upper = float(center * 2.2)
    clipped = _clip_candidates(
        [lower, center, upper],
        compactness_min=compactness_min,
        compactness_max=compactness_max,
    )
    if len(clipped) >= 3:
        lower, center, upper = clipped[0], clipped[1], clipped[2]
    elif len(clipped) == 2:
        lower, upper = clipped[0], clipped[1]
        center = float(np.sqrt(lower * upper))
    elif len(clipped) == 1:
        lower = center = upper = clipped[0]

    return {
        "center": float(center),
        "lower": float(lower),
        "upper": float(upper),
        "roughness": float(roughness) if np.isfinite(roughness) else None,
        "spacing_cv": float(spacing_cv) if np.isfinite(spacing_cv) else None,
        "sampled_points": int(coords_s.shape[0]),
        "estimated_segment_side_spots": float(segment_side_spots),
    }


def search_compactness(
    *,
    candidates: Sequence[float],
    embeddings: np.ndarray | None,
    spatial_coords: np.ndarray,
    segmenter: Callable[[float], pd.Series | np.ndarray | list[object]],
    compactness_search_dims: int = 12,
    compactness_min: float | None = None,
    compactness_max: float | None = None,
    scheduled: float | None = None,
    mode: str = "manual_search",
    target_segment_um: float | None = None,
    compactness_search_jobs: int = 1,
    compactness_metric_sample_size: int | None = None,
    verbose: bool = False,
) -> tuple[float, dict[str, Any]]:
    clipped = _clip_candidates(
        candidates,
        compactness_min=compactness_min,
        compactness_max=compactness_max,
    )
    if not clipped:
        raise ValueError("No valid compactness candidates remain after clipping.")
    rows = _evaluate_candidates(
        candidates=clipped,
        segmenter=segmenter,
        embeddings=embeddings,
        spatial_coords=spatial_coords,
        compactness_search_dims=int(compactness_search_dims),
        compactness_search_jobs=int(compactness_search_jobs),
        compactness_metric_sample_size=compactness_metric_sample_size,
        verbose=bool(verbose),
        mode=str(mode),
    )
    return finalize_compactness_rows(
        rows,
        scheduled=scheduled,
        mode=str(mode),
        target_segment_um=target_segment_um,
        compactness_search_jobs=int(compactness_search_jobs),
    )


def schedule_compactness(
    *,
    compactness: float | None,
    compactness_mode: str,
    target_segment_um: float | None,
    compactness_base_size_um: float | None,
    compactness_scale_power: float,
    compactness_min: float | None,
    compactness_max: float | None,
) -> float:
    value = 1.0 if compactness is None else float(compactness)
    if (
        str(compactness_mode) in {"scaled", "adaptive_search"}
        and target_segment_um is not None
        and compactness_base_size_um is not None
    ):
        base_size = float(compactness_base_size_um)
        if base_size <= 0:
            raise ValueError("compactness_base_size_um must be > 0.")
        value *= float(target_segment_um / base_size) ** float(compactness_scale_power)
    if compactness_min is not None:
        value = max(value, float(compactness_min))
    if compactness_max is not None:
        value = min(value, float(compactness_max))
    return float(value)


def resolve_compactness(
    *,
    compactness: float | None,
    compactness_mode: str,
    target_segment_um: float | None,
    compactness_base_size_um: float | None,
    compactness_scale_power: float,
    compactness_min: float | None,
    compactness_max: float | None,
    compactness_search_multipliers: Sequence[float] | None,
    compactness_search_dims: int,
    compactness_search_jobs: int,
    embeddings: np.ndarray | None,
    spatial_coords: np.ndarray,
    segmenter: Callable[[float], pd.Series | np.ndarray | list[object]] | None = None,
    compactness_metric_sample_size: int | None = None,
) -> tuple[float, dict[str, Any]]:
    mode = str(compactness_mode or "fixed")
    scheduled = None
    if mode != "auto_search" or compactness is not None:
        scheduled = schedule_compactness(
            compactness=None if compactness is None else float(compactness),
            compactness_mode=mode,
            target_segment_um=None if target_segment_um is None else float(target_segment_um),
            compactness_base_size_um=compactness_base_size_um,
            compactness_scale_power=float(compactness_scale_power),
            compactness_min=compactness_min,
            compactness_max=compactness_max,
        )
    if mode == "auto_search":
        if segmenter is None:
            raise ValueError("compactness_mode='auto_search' requires a segmenter callback.")
        auto_candidates = list(DEFAULT_AUTO_COMPACTNESS_CANDIDATES)
        if scheduled is not None:
            auto_candidates.append(float(scheduled))
        seed_candidates = _clip_candidates(
            auto_candidates,
            compactness_min=compactness_min,
            compactness_max=compactness_max,
        )
        if not seed_candidates:
            seed_candidates = _clip_candidates(
                [float(scheduled) if scheduled is not None else 0.03],
                compactness_min=compactness_min,
                compactness_max=compactness_max,
            )
        seed_rows = _evaluate_candidates(
            candidates=seed_candidates,
            segmenter=segmenter,
            embeddings=embeddings,
            spatial_coords=spatial_coords,
            compactness_search_dims=int(compactness_search_dims),
            compactness_search_jobs=int(compactness_search_jobs),
            compactness_metric_sample_size=compactness_metric_sample_size,
        )
        _apply_compactness_regularization(seed_rows, scheduled=scheduled)
        seed_best = _choose_best_row(seed_rows, scheduled=scheduled)
        refined_candidates = _refine_auto_candidates(
            seed_candidates=seed_candidates,
            best_compactness=float(seed_best["compactness"]),
            compactness_min=compactness_min,
            compactness_max=compactness_max,
        )
        refined_rows = _evaluate_candidates(
            candidates=refined_candidates,
            segmenter=segmenter,
            embeddings=embeddings,
            spatial_coords=spatial_coords,
            compactness_search_dims=int(compactness_search_dims),
            compactness_search_jobs=int(compactness_search_jobs),
            compactness_metric_sample_size=compactness_metric_sample_size,
        )
        combined_rows = {
            round(float(row["compactness"]), 12): row
            for row in seed_rows + refined_rows
        }
        rows = [combined_rows[key] for key in sorted(combined_rows)]
        _attach_rank_scores(rows)
        _apply_compactness_regularization(rows, scheduled=scheduled)
        best = _choose_best_row(rows, scheduled=scheduled)
        detail = {
            "mode": mode,
            "target_segment_um": None if target_segment_um is None else float(target_segment_um),
            "scheduled_compactness": None if scheduled is None else float(scheduled),
            "selected_compactness": float(best["compactness"]),
            "search_strategy": "absolute_broad_then_local_refine",
            "compactness_search_jobs": int(compactness_search_jobs),
            "candidates": [
                {
                    "compactness": float(row["compactness"]),
                    "score": float(row["score"]),
                    "compactness_penalty": float(row.get("compactness_penalty", 0.0)),
                    "tradeoff_gain": None if row.get("tradeoff_gain", None) is None else float(row["tradeoff_gain"]),
                    "fidelity_cost": None if row.get("fidelity_cost", None) is None else float(row["fidelity_cost"]),
                    "fidelity_preservation": None if row.get("fidelity_preservation", None) is None else float(row["fidelity_preservation"]),
                    "regularity_cost": None if row.get("regularity_cost", None) is None else float(row["regularity_cost"]),
                    "regularity_gain": None if row.get("regularity_gain", None) is None else float(row["regularity_gain"]),
                    "embedding_var": None if not np.isfinite(float(row["embedding_var"])) else float(row["embedding_var"]),
                    "mean_log_aspect": None if not np.isfinite(float(row["mean_log_aspect"])) else float(row["mean_log_aspect"]),
                    "size_cv": None if not np.isfinite(float(row["size_cv"])) else float(row["size_cv"]),
                    "neighbor_disagreement": None if not np.isfinite(float(row["neighbor_disagreement"])) else float(row["neighbor_disagreement"]),
                    "n_segments": int(round(float(row["n_segments"]))),
                }
                for row in rows
            ],
        }
        return float(best["compactness"]), detail

    scheduled = float(scheduled)
    if mode != "adaptive_search" or segmenter is None:
        detail = {
            "mode": mode,
            "target_segment_um": None if target_segment_um is None else float(target_segment_um),
            "scheduled_compactness": float(scheduled),
            "selected_compactness": float(scheduled),
            "candidates": [{"compactness": float(scheduled), "score": 0.0}],
        }
        return float(scheduled), detail

    multipliers = compactness_search_multipliers or DEFAULT_COMPACTNESS_SEARCH_MULTIPLIERS
    raw_candidates = [float(scheduled) * float(mult) for mult in multipliers]
    candidates = _clip_candidates(
        raw_candidates,
        compactness_min=compactness_min,
        compactness_max=compactness_max,
    )
    if not candidates:
        candidates = [float(scheduled)]

    rows = _evaluate_candidates(
        candidates=candidates,
        segmenter=segmenter,
        embeddings=embeddings,
        spatial_coords=spatial_coords,
        compactness_search_dims=int(compactness_search_dims),
        compactness_search_jobs=int(compactness_search_jobs),
        compactness_metric_sample_size=compactness_metric_sample_size,
    )
    _apply_compactness_regularization(rows, scheduled=scheduled)
    best = _choose_best_row(rows, scheduled=scheduled)
    detail = {
        "mode": mode,
        "target_segment_um": None if target_segment_um is None else float(target_segment_um),
        "scheduled_compactness": float(scheduled),
        "selected_compactness": float(best["compactness"]),
        "candidates": [
            {
                "compactness": float(row["compactness"]),
                "score": float(row["score"]),
                "compactness_penalty": float(row.get("compactness_penalty", 0.0)),
                "tradeoff_gain": None if row.get("tradeoff_gain", None) is None else float(row["tradeoff_gain"]),
                "fidelity_cost": None if row.get("fidelity_cost", None) is None else float(row["fidelity_cost"]),
                "fidelity_preservation": None if row.get("fidelity_preservation", None) is None else float(row["fidelity_preservation"]),
                "regularity_cost": None if row.get("regularity_cost", None) is None else float(row["regularity_cost"]),
                "regularity_gain": None if row.get("regularity_gain", None) is None else float(row["regularity_gain"]),
                "embedding_var": None if not np.isfinite(float(row["embedding_var"])) else float(row["embedding_var"]),
                "mean_log_aspect": None if not np.isfinite(float(row["mean_log_aspect"])) else float(row["mean_log_aspect"]),
                "size_cv": None if not np.isfinite(float(row["size_cv"])) else float(row["size_cv"]),
                "neighbor_disagreement": None if not np.isfinite(float(row["neighbor_disagreement"])) else float(row["neighbor_disagreement"]),
                "n_segments": int(round(float(row["n_segments"]))),
            }
            for row in rows
        ],
    }
    return float(best["compactness"]), detail
