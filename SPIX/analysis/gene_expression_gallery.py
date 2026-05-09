from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from anndata import AnnData
import pandas as pd

from ..superpixel.segmentation import segment_image
from ..visualization.plotting import image_plot
from .gene_expression_embedding import add_gene_expression_embedding


_GALLERY_ADATA: AnnData | None = None


def _set_gallery_adata(adata: AnnData | None) -> None:
    global _GALLERY_ADATA
    _GALLERY_ADATA = adata


def _build_minimal_gallery_adata(
    adata: AnnData,
    *,
    segment_keys: Sequence[str | None],
    obs_keys: Sequence[str | None] | None = None,
    layer_keys: Sequence[str | None] | None = None,
) -> AnnData:
    """Build a lightweight AnnData for gene-expression rendering workers.

    Process-parallel rendering does not need the full SPIX object. Keeping only
    X, var_names, the requested segment columns, and spatial coordinates
    substantially lowers worker memory use.
    """
    requested_obs = list(segment_keys) + list(obs_keys or [])
    obs_cols = []
    seen_obs: set[str] = set()
    for key in requested_obs:
        if key is None:
            continue
        col = str(key)
        if col in adata.obs.columns and col not in seen_obs:
            obs_cols.append(col)
            seen_obs.add(col)
    obs = adata.obs.loc[:, obs_cols].copy() if obs_cols else pd.DataFrame(index=adata.obs.index.copy())
    var = pd.DataFrame(index=adata.var_names.copy())
    mini = AnnData(X=adata.X, obs=obs, var=var)
    for key in layer_keys or []:
        if key is not None and str(key) in adata.layers:
            mini.layers[str(key)] = adata.layers[str(key)]
    if "spatial" in adata.obsm:
        mini.obsm["spatial"] = np.asarray(adata.obsm["spatial"])
    return mini


def _sanitize_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return token or "item"


def _normalize_gene_list(adata: AnnData, genes: Sequence[str]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    for gene in genes:
        gene_str = str(gene)
        if gene_str in seen:
            continue
        seen.add(gene_str)
        if gene_str in adata.var_names:
            valid.append(gene_str)
        else:
            missing.append(gene_str)

    if not valid:
        raise ValueError("None of the requested genes were found in adata.var_names.")

    return valid, missing


def _gene_expression_point_sizes(values: np.ndarray, point_size_range: tuple[float, float]) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    s0, s1 = float(point_size_range[0]), float(point_size_range[1])
    finite = np.isfinite(vals)
    sizes = np.full(vals.shape, s0, dtype=float)
    if not finite.any():
        return sizes
    vmin = float(np.nanmin(vals[finite]))
    vmax = float(np.nanmax(vals[finite]))
    if vmax <= vmin:
        sizes[finite & (vals > 0)] = s1
        return sizes
    t = np.zeros(vals.shape, dtype=float)
    t[finite] = np.clip((vals[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
    return s0 + t * (s1 - s0)


def _gene_expression_overlay_colors(adata: AnnData, color_by: str | None, palette: Any) -> np.ndarray | str:
    if color_by is None or color_by not in adata.obs:
        return "black"
    import matplotlib.colors as mcolors

    labels = pd.Categorical(adata.obs[color_by].astype("object"))
    categories = list(labels.categories)
    if isinstance(palette, Mapping):
        lut = {cat: palette.get(cat, "#808080") for cat in categories}
    else:
        try:
            palette_list = list(palette) if palette is not None and not isinstance(palette, str) else []
        except Exception:
            palette_list = []
        if palette_list:
            lut = {cat: palette_list[i % len(palette_list)] for i, cat in enumerate(categories)}
        else:
            cmap_name = palette if isinstance(palette, str) else "tab20"
            cmap = plt.get_cmap(cmap_name, max(len(categories), 1))
            lut = {cat: cmap(i) for i, cat in enumerate(categories)}
    return np.array([mcolors.to_rgba(lut.get(label, "#808080")) for label in labels], dtype=float)


def _default_segment_image_kwargs(
    segmentation_dimensions: Sequence[int],
    segmentation_embedding: str,
    show_segment_images: bool,
    verbose: bool,
) -> dict[str, Any]:
    return {
        "dimensions": list(segmentation_dimensions),
        "embedding": segmentation_embedding,
        "method": "image_plot_slic",
        "pitch_um": 2,
        "target_segment_um": 500,
        "runtime_fill_from_boundary": True,
        "runtime_fill_closing_radius": 2,
        "runtime_fill_holes": True,
        "soft_rasterization": True,
        "resolve_center_collisions": False,
        "verbose": verbose,
        "use_cached_image": True,
        "enforce_connectivity": False,
        "origin": True,
        "show_image": show_segment_images,
    }


def _default_image_plot_kwargs(
    image_dimensions: Sequence[int],
    gene_embedding_key: str,
) -> dict[str, Any]:
    return {
        "dimensions": list(image_dimensions),
        "embedding": gene_embedding_key,
    }


def _render_gene_mode_to_file(
    task: tuple[
        str,
        str | None,
        str,
        str,
        dict[str, Any],
        dict[str, Any],
        bool,
        bool,
        str,
        dict[str, Any] | None,
    ],
) -> Path:
    (
        gene,
        segment_key,
        embedding_key,
        title,
        resolved_image_kwargs,
        embed_kwargs,
        normalize_total,
        log1p,
        output_path,
        save_kwargs,
    ) = task
    if _GALLERY_ADATA is None:
        raise RuntimeError("Gallery worker state is not initialized.")
    adata = _GALLERY_ADATA

    add_gene_expression_embedding(
        adata,
        genes=[gene],
        segment_key=segment_key,
        embedding_key=embedding_key,
        normalize_total=normalize_total,
        log1p=log1p,
        **embed_kwargs,
    )

    out = Path(output_path)
    plot_kwargs = dict(resolved_image_kwargs)
    plot_kwargs.update({"show": False, "save": out, "save_kwargs": save_kwargs})
    fig_ax = image_plot(adata, title=title, **plot_kwargs)
    fig = fig_ax[0] if isinstance(fig_ax, tuple) else plt.gcf()
    plt.close(fig)
    return out


def _render_gene_expression_to_file(
    task: tuple[
        str,
        str,
        str | None,
        str,
        str,
        dict[str, Any],
        str | None,
        bool,
        bool,
        str,
        str,
        dict[str, Any] | None,
    ],
) -> dict[str, Any]:
    (
        group,
        gene,
        segment_key,
        mode_label,
        embedding_key,
        image_plot_kwargs,
        layer,
        normalize_total,
        log1p,
        aggregation,
        output_path,
        save_kwargs,
    ) = task
    if _GALLERY_ADATA is None:
        raise RuntimeError("Gallery worker state is not initialized.")
    adata = _GALLERY_ADATA

    add_gene_expression_embedding(
        adata,
        genes=[gene],
        layer=layer,
        segment_key=segment_key,
        embedding_key=embedding_key,
        normalize_total=normalize_total,
        log1p=log1p,
        aggregation=aggregation,
    )

    out = Path(output_path)
    plot_kwargs = dict(image_plot_kwargs)
    plot_kwargs["embedding"] = embedding_key
    if segment_key is not None and "segment_key" not in plot_kwargs:
        plot_kwargs["segment_key"] = segment_key
    plot_kwargs.update({"show": False, "save": out, "save_kwargs": save_kwargs, "return_fig": False})
    fig_ax = image_plot(adata, title=str(gene), **plot_kwargs)
    fig = fig_ax[0] if isinstance(fig_ax, tuple) else plt.gcf()
    plt.close(fig)
    return {
        "scale_group": group,
        "gene": gene,
        "mode": mode_label,
        "segment_key": segment_key,
        "path": out,
    }


def _render_evidence_gene_expression_to_file(task: Mapping[str, Any]) -> dict[str, Any]:
    if _GALLERY_ADATA is None:
        raise RuntimeError("Gallery worker state is not initialized.")
    adata = _GALLERY_ADATA
    gene = str(task["gene"])
    mode_label = str(task["mode"])
    segment_key = task.get("segment_key", None)
    out_path = Path(task["path"])
    fig_ax = plot_gene_expression(
        adata,
        gene,
        segment_key=segment_key,
        layer=task.get("layer", None),
        normalize_total=bool(task.get("normalize_total", True)),
        log1p=bool(task.get("log1p", True)),
        aggregation=str(task.get("aggregation", "sum")),
        embedding_key=str(task.get("embedding_key", "X_gene_embedding")),
        title=str(task.get("title", f"{gene} | {mode_label}")),
        save=out_path,
        save_kwargs=task.get("save_kwargs", None),
        show=False,
        return_fig=True,
        point_size_range=task.get("point_size_range", None),
        **dict(task.get("plot_kwargs", {})),
    )
    if fig_ax is not None:
        fig = fig_ax[0] if isinstance(fig_ax, tuple) else plt.gcf()
        plt.close(fig)
    return {
        "evidence_class": str(task["evidence_class"]),
        "evidence_label": str(task.get("evidence_label", "")),
        "gene": gene,
        "mode": mode_label,
        "segment_key": segment_key,
        "path": out_path,
    }


def _make_contact_sheet(
    records: Sequence[dict[str, Any]],
    *,
    out_path: Path,
    ncols: int = 5,
    thumb_size: tuple[float, float] = (3.2, 3.2),
    title: str | None = None,
    dpi: int = 180,
) -> Path:
    if not records:
        raise ValueError("No records were provided for the contact sheet.")
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(records) / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(float(thumb_size[0]) * ncols, float(thumb_size[1]) * nrows),
        dpi=int(dpi),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, record in zip(axes.ravel(), records):
        img = mpimg.imread(str(record["path"]))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(str(record["gene"]), fontsize=8, pad=2)
    if title:
        fig.suptitle(str(title), fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.97 if title else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_path


def _make_evidence_contact_sheet(
    records: Sequence[dict[str, Any]],
    *,
    out_path: Path,
    genes: Sequence[str],
    modes: Sequence[str],
    title: str | None = None,
    thumb_size: tuple[float, float] = (3.1, 3.1),
    dpi: int = 180,
) -> Path:
    if not records:
        raise ValueError("No records were provided for the evidence contact sheet.")
    genes = [str(g) for g in genes]
    modes = [str(m) for m in modes]
    if not genes or not modes:
        raise ValueError("genes and modes must be non-empty.")

    lookup = {
        (str(record["gene"]), str(record["mode"])): record
        for record in records
    }
    nrows = len(genes)
    ncols = len(modes)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(float(thumb_size[0]) * ncols, float(thumb_size[1]) * nrows),
        dpi=int(dpi),
        squeeze=False,
    )
    for row_idx, gene in enumerate(genes):
        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            record = lookup.get((gene, mode))
            if record is not None and Path(record["path"]).exists():
                img = mpimg.imread(str(record["path"]))
                ax.imshow(img)
            if row_idx == 0:
                ax.set_title(mode, fontsize=10, fontweight="bold", pad=4)
            if col_idx == 0:
                ax.text(
                    -0.02,
                    0.5,
                    gene,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    rotation=90,
                )
    if title:
        fig.suptitle(str(title), fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.965 if title else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_path


def _resolution_response_class_label(value: str) -> str:
    labels = {
        "native_preserved": "Native signal preserved by fine SPIX",
        "shared_native_fine": "Shared Native/Fine SVGs",
        "native_only": "Native-only SVGs",
        "fine_only": "Fine SPIX-only SVGs",
        "fine_denoised_or_rescued": "Fine SPIX denoises or rescues signal",
        "fine_local_signal": "Fine-scale local signal",
        "mid_new_or_enhanced_vs_native": "Mid-scale signal enhanced vs native",
        "mid_scale_refined": "Mid-scale refined signal",
        "mid_response": "Mid-scale response",
        "coarse_new_or_enhanced_vs_native": "Coarse-scale signal enhanced vs native",
        "coarse_scale_refined": "Coarse-scale refined signal",
        "coarse_response": "Coarse-scale response",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def list_gene_expression_segment_keys(
    adata: AnnData,
    *,
    contains: str | None = "Segment",
    include_nunique: bool = True,
    max_keys: int | None = None,
) -> pd.DataFrame:
    """List candidate ``adata.obs`` columns that can be used as segment keys."""
    if not hasattr(adata, "obs"):
        return pd.DataFrame(columns=["segment_key", "n_segments", "dtype"])
    columns = [str(col) for col in adata.obs.columns]
    if contains is not None:
        token = str(contains).lower()
        columns = [col for col in columns if token in col.lower()]
    rows: list[dict[str, Any]] = []
    for col in columns:
        row = {"segment_key": col, "dtype": str(adata.obs[col].dtype)}
        if include_nunique:
            try:
                row["n_segments"] = int(pd.Series(adata.obs[col]).nunique(dropna=True))
            except Exception:
                row["n_segments"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["segment_key", "n_segments", "dtype"] if include_nunique else ["segment_key", "dtype"])
    sort_cols = ["n_segments", "segment_key"] if include_nunique and "n_segments" in out.columns else ["segment_key"]
    ascending = [False, True] if sort_cols[0] == "n_segments" else [True]
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    if max_keys is not None:
        out = out.head(int(max_keys)).copy()
    return out


def _resolve_segment_npz_path(raw_path: Any, base_dir: Path) -> Path:
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
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base_dir / p).resolve()


def _scale_label_from_group(value: Any) -> str | None:
    text = str(value).strip().lower()
    if any(token in text for token in ["fine", "high"]):
        return "fine"
    if "mid" in text:
        return "mid"
    if any(token in text for token in ["coarse", "low"]):
        return "coarse"
    return None


def _is_true_like(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _select_representative_scale(
    sub: pd.DataFrame,
    *,
    representative: str,
    observed_col: str,
    resolution_col: str,
    scale_id_col: str,
) -> pd.Series:
    if sub.empty:
        raise ValueError("Cannot select a representative scale from an empty table.")
    work = sub.copy()
    if representative == "largest_observed_segments":
        values = pd.to_numeric(work.get(observed_col), errors="coerce")
        idx = values.fillna(-np.inf).idxmax()
        return work.loc[idx]
    if representative == "smallest_observed_segments":
        values = pd.to_numeric(work.get(observed_col), errors="coerce")
        idx = values.fillna(np.inf).idxmin()
        return work.loc[idx]
    if representative == "median_resolution":
        values = pd.to_numeric(work.get(resolution_col), errors="coerce")
    else:
        values = pd.to_numeric(work.get(observed_col), errors="coerce")
        if not np.isfinite(values.to_numpy(float)).any():
            values = pd.to_numeric(work.get(resolution_col), errors="coerce")
    if not np.isfinite(values.to_numpy(float)).any():
        return work.sort_values(scale_id_col, kind="mergesort").iloc[len(work) // 2]
    target = float(np.nanmedian(values.to_numpy(float)))
    idx = (values - target).abs().sort_values(kind="mergesort").index[0]
    return work.loc[idx]


def prepare_resolution_response_image_segment_modes(
    adata: AnnData,
    scale_groups: pd.DataFrame,
    segments_index: str | Path | pd.DataFrame,
    *,
    group_col: str | None = None,
    scale_id_col: str = "scale_id",
    path_col: str = "path",
    observed_col: str = "observed_obs_n_segments",
    resolution_col: str = "resolution",
    representative: str = "median_observed_segments",
    representative_scale_ids: Mapping[str, str] | None = None,
    mode_labels: Mapping[str, str] | None = None,
    include_scales: Sequence[str] | None = None,
    obs_key_prefix: str = "__spix_evidence_segment",
    include_native: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load representative fine/mid/coarse multiscale segment labels into ``adata.obs``.

    ``segments_index`` is the multiscale ``segments_index.csv`` produced by
    SPIX.  One representative non-native scale is selected from each
    fine/mid/coarse group in ``scale_groups`` and its ``seg_codes`` vector is
    copied into ``adata.obs``.  The returned ``mode_segment_keys`` can be passed
    directly to :func:`render_resolution_response_evidence_image_panels`.
    """
    if scale_groups is None or scale_groups.empty:
        raise ValueError("scale_groups must be a non-empty DataFrame.")
    if isinstance(segments_index, pd.DataFrame):
        index_df = segments_index.copy()
        base_dir = Path(".").resolve()
    else:
        segments_index_path = Path(segments_index)
        index_df = pd.read_csv(segments_index_path)
        base_dir = segments_index_path.resolve().parent
    if scale_id_col not in index_df.columns:
        raise KeyError(f"segments_index must contain {scale_id_col!r}.")
    if path_col not in index_df.columns:
        raise KeyError(f"segments_index must contain {path_col!r}.")
    if scale_id_col not in scale_groups.columns:
        raise KeyError(f"scale_groups must contain {scale_id_col!r}.")

    if group_col is None:
        for candidate in ["target_group", "scale_group", "preferred_scale_group", "group"]:
            if candidate in scale_groups.columns:
                group_col = candidate
                break
    if group_col is None or group_col not in scale_groups.columns:
        raise KeyError("scale_groups must contain a group column such as 'target_group' or 'scale_group'.")

    groups = scale_groups[[scale_id_col, group_col]].dropna(subset=[scale_id_col, group_col]).copy()
    groups[scale_id_col] = groups[scale_id_col].astype(str)
    groups["_evidence_scale_label"] = groups[group_col].map(_scale_label_from_group)
    groups = groups.loc[groups["_evidence_scale_label"].notna()].copy()
    if groups.empty:
        raise ValueError(f"No fine/mid/coarse-like labels were inferred from scale_groups[{group_col!r}].")

    index_df = index_df.copy()
    index_df[scale_id_col] = index_df[scale_id_col].astype(str)
    meta = index_df.merge(groups, on=scale_id_col, how="inner")
    if meta.empty:
        missing = groups[scale_id_col].astype(str).head(10).tolist()
        raise ValueError(f"No scale_groups scale_id values matched segments_index. First scale_groups ids: {missing}")
    if "native_identity" in meta.columns:
        meta = meta.loc[~meta["native_identity"].map(_is_true_like)].copy()
    if "mode" in meta.columns:
        meta = meta.loc[~meta["mode"].astype(str).str.lower().eq("native_identity")].copy()
    if meta.empty:
        raise ValueError("Only native rows matched; no non-native segment npz rows were available.")

    labels = dict(mode_labels or {"fine": "Fine SPIX", "mid": "Mid SPIX", "coarse": "Coarse SPIX"})
    wanted_scales = ["fine", "mid", "coarse"] if include_scales is None else [
        str(s).strip().lower() for s in include_scales
    ]
    wanted_scales = [s for s in wanted_scales if s in {"fine", "mid", "coarse"}]
    if not wanted_scales:
        raise ValueError("include_scales must contain at least one of: 'fine', 'mid', 'coarse'.")
    mode_segment_keys: dict[str, str | None] = {"Native": None} if include_native else {}
    selected_rows: list[dict[str, Any]] = []
    fixed_scale_ids = {str(k).strip().lower(): str(v) for k, v in (representative_scale_ids or {}).items()}
    for label in wanted_scales:
        sub = meta.loc[meta["_evidence_scale_label"] == label].copy()
        if sub.empty:
            continue
        if label in fixed_scale_ids:
            fixed_id = fixed_scale_ids[label]
            fixed = sub.loc[sub[scale_id_col].astype(str).eq(fixed_id)]
            if fixed.empty:
                raise ValueError(f"Requested representative {label} scale_id={fixed_id!r} was not found.")
            row = fixed.iloc[0]
        else:
            row = _select_representative_scale(
                sub,
                representative=str(representative),
                observed_col=observed_col,
                resolution_col=resolution_col,
                scale_id_col=scale_id_col,
            )
        sid = str(row[scale_id_col])
        obs_key = f"{obs_key_prefix}_{label}_{_sanitize_path_token(sid)}"
        raw_path = row[path_col]
        npz_path = _resolve_segment_npz_path(raw_path, base_dir)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Segment npz for {label} scale {sid!r} was not found: raw={raw_path!r}, resolved={npz_path}"
            )
        with np.load(npz_path, allow_pickle=False) as z:
            if "seg_codes" not in z.files:
                raise KeyError(f"{npz_path} does not contain 'seg_codes'.")
            seg_codes = np.asarray(z["seg_codes"])
        if seg_codes.shape[0] != adata.n_obs:
            raise ValueError(
                f"Segment codes for {sid!r} have length {seg_codes.shape[0]}, but adata.n_obs is {adata.n_obs}."
            )
        adata.obs[obs_key] = pd.Categorical(seg_codes.astype(str))
        mode_segment_keys[labels.get(label, label.title())] = obs_key
        selected = row.to_dict()
        selected.update(
            {
                "evidence_scale_label": label,
                "display_label": labels.get(label, label.title()),
                "segment_key": obs_key,
                "resolved_path": str(npz_path),
            }
        )
        selected_rows.append(selected)
        if verbose:
            n_segments = selected.get(observed_col, "")
            print(f"[segment] {labels.get(label, label.title())}: {sid} -> adata.obs[{obs_key!r}] ({n_segments} segments)")

    return {
        "mode_segment_keys": mode_segment_keys,
        "selected_scales": pd.DataFrame(selected_rows),
        "segments_index": index_df,
    }


def _infer_image_evidence_scales_from_classes(classes: Sequence[str] | None) -> list[str]:
    if classes is None:
        return ["fine", "mid", "coarse"]
    scales: list[str] = []
    for cls in classes:
        text = str(cls).lower()
        if text.startswith("native_") or text.startswith("fine_") or "fine" in text:
            scales.append("fine")
        if text.startswith("mid_") or "mid" in text:
            scales.append("mid")
        if text.startswith("coarse_") or "coarse" in text:
            scales.append("coarse")
    out: list[str] = []
    for scale in scales:
        if scale not in out:
            out.append(scale)
    return out or ["fine", "mid", "coarse"]


def select_resolution_response_image_examples(
    gene_evidence: pd.DataFrame,
    *,
    classes: Sequence[str] | None = None,
    n_per_class: int = 2,
    gene_col: str = "gene",
    class_col: str = "claim_class",
) -> pd.DataFrame:
    """Pick representative SVGs for image-level resolution-response panels.

    The selector is intentionally simple and paper-facing: for preserved fine
    signals, it favors genes that are strong in both native and fine SPIX; for
    rescued/enhanced signals, it favors genes whose target scale improves the
    most over native while still having a strong target-scale anchor.
    """
    if gene_evidence is None or gene_evidence.empty:
        return pd.DataFrame()
    if gene_col not in gene_evidence.columns:
        raise ValueError(f"gene_evidence must contain {gene_col!r}.")
    if class_col not in gene_evidence.columns:
        raise ValueError(f"gene_evidence must contain {class_col!r}.")

    work = gene_evidence.copy()
    if work.columns.duplicated().any():
        work = work.loc[:, ~work.columns.duplicated()].copy()
    work[gene_col] = work[gene_col].astype(str)
    work[class_col] = work[class_col].astype(str)

    default_classes = [
        "native_preserved",
        "fine_denoised_or_rescued",
        "fine_local_signal",
        "mid_new_or_enhanced_vs_native",
        "mid_scale_refined",
        "coarse_new_or_enhanced_vs_native",
        "coarse_scale_refined",
    ]
    class_order = [str(c) for c in (classes if classes is not None else default_classes)]

    for col in [
        "target_anchor_rank_pct",
        "native_rank_pct",
        "fine_rank_pct",
        "fine_non_native_median_rank_pct",
        "mid_peak_rank_pct",
        "coarse_median_rank_pct",
        "native_to_target_gain",
        "fine_to_target_gain",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    selected: list[pd.DataFrame] = []
    for claim in class_order:
        sub = work.loc[work[class_col] == claim].copy()
        if sub.empty:
            continue
        if claim == "native_preserved":
            sort_cols = [
                col
                for col in ["target_anchor_rank_pct", "native_rank_pct", gene_col]
                if col in sub.columns
            ]
            ascending = [True] * len(sort_cols)
        elif claim == "shared_native_fine":
            sort_cols = [
                col
                for col in ["fine_rank_pct", "target_anchor_rank_pct", "native_rank_pct", gene_col]
                if col in sub.columns
            ]
            ascending = [True] * len(sort_cols)
        elif claim == "native_only":
            sort_cols = [
                col
                for col in ["native_rank_pct", "fine_rank_pct", gene_col]
                if col in sub.columns
            ]
            ascending = [True, False, True][: len(sort_cols)]
        elif claim == "fine_only":
            sort_cols = [
                col
                for col in ["fine_rank_pct", "target_anchor_rank_pct", "native_to_target_gain", gene_col]
                if col in sub.columns
            ]
            ascending = [True, True, False, True][: len(sort_cols)]
        elif claim == "fine_local_signal":
            sort_cols = [
                col
                for col in ["target_anchor_rank_pct", "fine_to_target_gain", gene_col]
                if col in sub.columns
            ]
            ascending = [True, False, True][: len(sort_cols)]
        else:
            sort_cols = [
                col
                for col in ["native_to_target_gain", "target_anchor_rank_pct", gene_col]
                if col in sub.columns
            ]
            ascending = [False, True, True][: len(sort_cols)]
        if sort_cols:
            sub = sub.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        sub = sub.head(int(max(1, n_per_class))).copy()
        sub["evidence_class"] = claim
        sub["evidence_label"] = _resolution_response_class_label(claim)
        sub["example_rank"] = np.arange(1, len(sub) + 1)
        selected.append(sub)

    if not selected:
        return pd.DataFrame(columns=list(work.columns) + ["evidence_class", "evidence_label", "example_rank"])
    out = pd.concat(selected, ignore_index=True)
    out["_class_order"] = out["evidence_class"].map({c: i for i, c in enumerate(class_order)})
    out = out.sort_values(["_class_order", "example_rank", gene_col], kind="mergesort").drop(columns=["_class_order"])
    return out.reset_index(drop=True)


def render_resolution_response_evidence_image_panels(
    adata: AnnData,
    gene_evidence: pd.DataFrame,
    *,
    out_dir: str | Path,
    mode_segment_keys: Mapping[str, str | None] | str | None = None,
    scale_groups: pd.DataFrame | None = None,
    segments_index: str | Path | pd.DataFrame | None = None,
    segment_group_col: str | None = None,
    segment_representative: str = "median_observed_segments",
    segment_scale_ids: Mapping[str, str] | None = None,
    segment_include_scales: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
    n_per_class: int = 2,
    gene_col: str = "gene",
    class_col: str = "claim_class",
    layer: str | None = None,
    normalize_total: bool = True,
    log1p: bool = True,
    aggregation: str = "sum",
    embedding_key: str = "X_gene_embedding",
    base_plot_kwargs: Mapping[str, Any] | None = None,
    mode_plot_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    point_size_range: tuple[float, float] | None = None,
    n_jobs: int = 1,
    backend: str = "process",
    make_contact_sheets: bool = True,
    show_panels: bool = False,
    contact_sheet_dpi: int = 180,
    contact_sheet_thumb_size: tuple[float, float] = (3.1, 3.1),
    save_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Render representative image plots for resolution-response evidence classes.

    Rows are representative genes and columns are display modes.  Pass
    ``mode_segment_keys`` explicitly, or set it to ``None``/``"auto"`` with
    ``scale_groups`` and ``segments_index`` to load representative fine/mid/coarse
    segment labels from saved multiscale ``.npz`` files.
    """
    prepared_modes: dict[str, Any] | None = None
    if mode_segment_keys is None or (isinstance(mode_segment_keys, str) and mode_segment_keys.lower() == "auto"):
        if scale_groups is None or segments_index is None:
            raise ValueError(
                "Automatic mode_segment_keys requires both scale_groups and segments_index."
            )
        prepared_modes = prepare_resolution_response_image_segment_modes(
            adata,
            scale_groups,
            segments_index,
            group_col=segment_group_col,
            representative=segment_representative,
            representative_scale_ids=segment_scale_ids,
            include_scales=(
                segment_include_scales
                if segment_include_scales is not None
                else _infer_image_evidence_scales_from_classes(classes)
            ),
            verbose=verbose,
        )
        mode_segment_keys = prepared_modes["mode_segment_keys"]
    if not isinstance(mode_segment_keys, Mapping) or not mode_segment_keys:
        raise ValueError("mode_segment_keys must map display labels to segment keys.")
    out_dir = Path(out_dir)
    image_dir = out_dir / "individual"
    panel_dir = out_dir / "panels"
    image_dir.mkdir(parents=True, exist_ok=True)
    if make_contact_sheets:
        panel_dir.mkdir(parents=True, exist_ok=True)

    examples = select_resolution_response_image_examples(
        gene_evidence,
        classes=classes,
        n_per_class=n_per_class,
        gene_col=gene_col,
        class_col=class_col,
    )
    if examples.empty:
        raise ValueError("No representative genes were selected from gene_evidence.")
    valid_genes, missing_genes = _normalize_gene_list(adata, examples[gene_col].tolist())
    valid_set = set(valid_genes)
    examples = examples.loc[examples[gene_col].isin(valid_set)].copy()
    if examples.empty:
        raise ValueError("No selected representative genes were found in adata.var_names.")
    if missing_genes and verbose:
        print(f"[warn] Missing genes skipped: {', '.join(missing_genes)}")

    missing_modes = [
        str(mode)
        for mode, segment_key in mode_segment_keys.items()
        if segment_key is not None and str(segment_key) not in adata.obs.columns
    ]
    if missing_modes:
        candidates = list_gene_expression_segment_keys(adata, max_keys=30)
        candidate_msg = ""
        if not candidates.empty:
            candidate_msg = (
                " Available segment-like adata.obs columns include: "
                + ", ".join(candidates["segment_key"].astype(str).tolist())
                + "."
            )
        raise ValueError(
            "Some mode segment keys were not found in adata.obs: "
            + ", ".join(missing_modes)
            + ". Check mode_segment_keys."
            + candidate_msg
        )

    plot_kwargs_base = {
        "figsize": (6, 6),
        "fig_dpi": 120,
        "dimensions": [0],
        "embedding": embedding_key,
        "boundary_method": "pixel",
        "show_colorbar": False,
        "show_legend": False,
        "legend_ncol": 1,
        "runtime_fill_from_boundary": False,
        "runtime_fill_holes": True,
        "resolve_center_collisions": True,
        "cmap": "viridis",
        "plot_boundaries": False,
    }
    if base_plot_kwargs:
        plot_kwargs_base.update(dict(base_plot_kwargs))
    plot_kwargs_base.pop("title", None)
    plot_kwargs_base.pop("save", None)
    plot_kwargs_base.pop("show", None)
    plot_kwargs_base.pop("return_fig", None)
    plot_kwargs_base["embedding"] = embedding_key
    mode_overrides = {str(k): dict(v) for k, v in (mode_plot_kwargs or {}).items()}

    saved_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    mode_items = [(str(mode), segment_key) for mode, segment_key in mode_segment_keys.items()]
    save_opts = {"dpi": 250, "bbox_inches": "tight", "pad_inches": 0.03}
    if save_kwargs:
        save_opts.update(dict(save_kwargs))

    tasks: list[dict[str, Any]] = []
    for ex in examples.itertuples(index=False):
        gene = str(getattr(ex, gene_col))
        evidence_class = str(getattr(ex, "evidence_class"))
        evidence_label = str(getattr(ex, "evidence_label"))
        for mode_label, segment_key in mode_items:
            plot_kwargs = dict(plot_kwargs_base)
            plot_kwargs.update(mode_overrides.get(mode_label, {}))
            out_path = (
                image_dir
                / _sanitize_path_token(evidence_class)
                / _sanitize_path_token(mode_label)
                / f"{_sanitize_path_token(gene)}.png"
            )
            tasks.append(
                {
                    "evidence_class": evidence_class,
                    "evidence_label": evidence_label,
                    "gene": gene,
                    "mode": mode_label,
                    "segment_key": segment_key,
                    "path": out_path,
                    "layer": layer,
                    "normalize_total": bool(normalize_total),
                    "log1p": bool(log1p),
                    "aggregation": str(aggregation),
                    "embedding_key": str(embedding_key),
                    "title": f"{gene} | {mode_label}",
                    "save_kwargs": save_opts,
                    "point_size_range": point_size_range,
                    "plot_kwargs": plot_kwargs,
                }
            )

    run_parallel = int(max(1, n_jobs)) > 1 and str(backend).lower() == "process"
    if run_parallel:
        if verbose:
            print(f"[parallel] rendering {len(tasks)} evidence images with processes: jobs={int(max(1, n_jobs))}")
        obs_extra_keys: list[str] = []
        for kwargs in [plot_kwargs_base, *mode_overrides.values()]:
            for key in ["color_by", "alpha_by", "overlap_priority"]:
                value = kwargs.get(key, None)
                if isinstance(value, str) and value != "embedding" and value in adata.obs.columns:
                    obs_extra_keys.append(value)
        segment_keys = [segment_key for _, segment_key in mode_items if segment_key is not None]
        layer_keys = [layer] if layer is not None else []
        worker_adata = _build_minimal_gallery_adata(
            adata,
            segment_keys=segment_keys,
            obs_keys=obs_extra_keys,
            layer_keys=layer_keys,
        )
        ctx = mp.get_context("fork")
        _set_gallery_adata(worker_adata)
        try:
            with ProcessPoolExecutor(
                max_workers=int(max(1, n_jobs)),
                mp_context=ctx,
                initializer=_set_gallery_adata,
                initargs=(worker_adata,),
            ) as ex:
                future_to_task = {ex.submit(_render_evidence_gene_expression_to_file, task): task for task in tasks}
                for fut in as_completed(future_to_task):
                    task = future_to_task[fut]
                    try:
                        record = fut.result()
                        saved_records.append(record)
                        if verbose:
                            print(f"[saved] {record['evidence_class']} | {record['gene']} | {record['mode']}: {record['path']}")
                    except Exception as exc:
                        failures.append(
                            {
                                "evidence_class": task["evidence_class"],
                                "gene": task["gene"],
                                "mode": task["mode"],
                                "segment_key": task["segment_key"],
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if verbose:
                            print(f"[fail] {task['evidence_class']} | {task['gene']} | {task['mode']}: {exc}")
        finally:
            _set_gallery_adata(None)
    else:
        if verbose:
            print(f"[serial] rendering {len(tasks)} evidence images")
        _set_gallery_adata(adata)
        try:
            for task in tasks:
                try:
                    record = _render_evidence_gene_expression_to_file(task)
                    saved_records.append(record)
                    if verbose:
                        print(f"[saved] {record['evidence_class']} | {record['gene']} | {record['mode']}: {record['path']}")
                except Exception as exc:
                    failures.append(
                        {
                            "evidence_class": task["evidence_class"],
                            "gene": task["gene"],
                            "mode": task["mode"],
                            "segment_key": task["segment_key"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if verbose:
                        print(f"[fail] {task['evidence_class']} | {task['gene']} | {task['mode']}: {exc}")
        finally:
            _set_gallery_adata(None)

    # Preserve requested order in the saved table and contact sheets.
    task_order = {
        (
            str(task["evidence_class"]),
            str(task["gene"]),
            str(task["mode"]),
        ): i
        for i, task in enumerate(tasks)
    }
    saved_records = sorted(
        saved_records,
        key=lambda record: task_order.get(
            (str(record["evidence_class"]), str(record["gene"]), str(record["mode"])),
            10**12,
        ),
    )

    saved_df = pd.DataFrame(saved_records)
    failures_df = pd.DataFrame(failures)
    selected_path = out_dir / "selected_image_evidence_examples.csv"
    examples.to_csv(selected_path, index=False)
    if not saved_df.empty:
        saved_df.to_csv(out_dir / "rendered_image_evidence_plots.csv", index=False)
    if not failures_df.empty:
        failures_df.to_csv(out_dir / "render_image_evidence_failures.csv", index=False)

    panel_paths: list[Path] = []
    if make_contact_sheets and not saved_df.empty:
        mode_order = [mode for mode, _ in mode_items]
        for evidence_class, ex_sub in examples.groupby("evidence_class", sort=False):
            genes = ex_sub[gene_col].astype(str).tolist()
            records = saved_df.loc[saved_df["evidence_class"] == str(evidence_class)].to_dict("records")
            if not records:
                continue
            title = _resolution_response_class_label(str(evidence_class))
            out_path = panel_dir / f"{_sanitize_path_token(str(evidence_class))}.png"
            panel_paths.append(
                _make_evidence_contact_sheet(
                    records,
                    out_path=out_path,
                    genes=genes,
                    modes=mode_order,
                    title=title,
                    thumb_size=contact_sheet_thumb_size,
                    dpi=int(contact_sheet_dpi),
                )
            )
            if verbose:
                print(f"[panel] {out_path}")

    if show_panels and panel_paths:
        try:
            from IPython.display import Image, display

            for path in panel_paths:
                display(Image(filename=str(path)))
        except Exception:
            for path in panel_paths:
                if verbose:
                    print(f"[panel] {path}")

    return {
        "selected_examples": examples,
        "selected_examples_path": selected_path,
        "saved_images": saved_df,
        "failures": failures_df,
        "panel_paths": panel_paths,
        "out_dir": out_dir,
        "mode_segment_keys": dict(mode_segment_keys),
        "prepared_modes": prepared_modes,
    }


def render_native_fine_svg_overlap_spatial_panels(
    adata: AnnData,
    native_fine_overlap: pd.DataFrame,
    *,
    out_dir: str | Path,
    scale_groups: pd.DataFrame | None = None,
    segments_index: str | Path | pd.DataFrame | None = None,
    mode_segment_keys: Mapping[str, str | None] | str | None = None,
    classes: Sequence[str] = ("native_only", "fine_only"),
    n_per_class: int = 3,
    gene_col: str = "gene",
    class_col: str = "overlap_class",
    layer: str | None = None,
    normalize_total: bool = True,
    log1p: bool = True,
    aggregation: str = "sum",
    embedding_key: str = "X_gene_embedding",
    segment_group_col: str | None = None,
    segment_representative: str = "median_observed_segments",
    fine_scale_id: str | None = None,
    base_plot_kwargs: Mapping[str, Any] | None = None,
    mode_plot_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    point_size_range: tuple[float, float] | None = None,
    n_jobs: int = 1,
    backend: str = "process",
    make_contact_sheets: bool = True,
    show_panels: bool = True,
    contact_sheet_dpi: int = 180,
    contact_sheet_thumb_size: tuple[float, float] = (3.1, 3.1),
    save_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Render Native-vs-Fine spatial maps for Native-only and Fine-only SVGs.

    This is a focused wrapper around
    :func:`render_resolution_response_evidence_image_panels` for the native/fine
    top-SVG overlap table produced by
    :func:`SPIX.analysis.scale_biology.make_native_fine_svg_overlap_table`.
    The default panel has rows of representative genes and two columns:
    ``Native`` and one representative ``Fine SPIX`` scale.
    """
    if native_fine_overlap is None or native_fine_overlap.empty:
        raise ValueError("native_fine_overlap must be a non-empty DataFrame.")
    if class_col not in native_fine_overlap.columns:
        raise ValueError(f"native_fine_overlap must contain {class_col!r}.")

    if mode_segment_keys is None:
        mode_segment_keys = "auto"
    if mode_segment_keys == "auto" and (scale_groups is None or segments_index is None):
        raise ValueError(
            "Automatic Native/Fine panel rendering requires scale_groups and segments_index. "
            "Alternatively pass mode_segment_keys={'Native': None, 'Fine SPIX': '<adata.obs segment key>'}."
        )

    plot_kwargs = {
        "figsize": (6, 6),
        "fig_dpi": 120,
        "dimensions": [0],
        "embedding": embedding_key,
        "boundary_method": "pixel",
        "show_colorbar": False,
        "show_legend": False,
        "legend_ncol": 1,
        "runtime_fill_from_boundary": False,
        "runtime_fill_holes": True,
        "resolve_center_collisions": True,
        "cmap": "viridis",
        "plot_boundaries": False,
    }
    if base_plot_kwargs:
        plot_kwargs.update(dict(base_plot_kwargs))

    return render_resolution_response_evidence_image_panels(
        adata,
        native_fine_overlap,
        out_dir=out_dir,
        mode_segment_keys=mode_segment_keys,
        scale_groups=scale_groups,
        segments_index=segments_index,
        segment_group_col=segment_group_col,
        segment_representative=segment_representative,
        segment_scale_ids={"fine": str(fine_scale_id)} if fine_scale_id is not None else None,
        segment_include_scales=["fine"],
        classes=classes,
        n_per_class=n_per_class,
        gene_col=gene_col,
        class_col=class_col,
        layer=layer,
        normalize_total=normalize_total,
        log1p=log1p,
        aggregation=aggregation,
        embedding_key=embedding_key,
        base_plot_kwargs=plot_kwargs,
        mode_plot_kwargs=mode_plot_kwargs,
        point_size_range=point_size_range,
        n_jobs=n_jobs,
        backend=backend,
        make_contact_sheets=make_contact_sheets,
        show_panels=show_panels,
        contact_sheet_dpi=contact_sheet_dpi,
        contact_sheet_thumb_size=contact_sheet_thumb_size,
        save_kwargs=save_kwargs,
        verbose=verbose,
    )


def plot_gene_expression_triplets(
    adata: AnnData,
    genes: Sequence[str],
    *,
    segmentation_dimensions: Sequence[int] = tuple(range(30)),
    segmentation_embedding: str = "X_embedding_equalize",
    image_dimensions: Sequence[int] = (0,),
    gene_embedding_key: str = "X_gene_embedding",
    spix_segment_key: str = "Segment_SPIX",
    grid_segment_key: str = "Segment_GRID",
    spix_compactness: float = 0.1,
    grid_compactness: float = 10000000.0,
    normalize_total: bool = True,
    log1p: bool = True,
    force_resegment: bool = False,
    show_segment_images: bool = False,
    display_inline: bool = True,
    out_dir: str | Path | None = None,
    title_template: str = "{gene} | {mode}",
    verbose: bool = True,
    segment_image_kwargs: dict[str, Any] | None = None,
    image_plot_kwargs: dict[str, Any] | None = None,
    save_kwargs: dict[str, Any] | None = None,
    n_jobs: int = 1,
    backend: str = "process",
) -> dict[str, Any]:
    """Render SPIX, GRID, and raw gene-expression image plots for each gene.

    The defaults mirror the repeated cells in `SPIX_figure2_tmp.ipynb`, but the
    segmentation outputs are stored under separate keys so they can be reused
    without overwriting `adata.obs['Segment']`.

    ``out_dir`` saves one PNG per gene/mode. ``save_kwargs`` is forwarded to
    ``Figure.savefig`` through ``image_plot``.
    """
    if not display_inline and out_dir is None:
        raise ValueError("Set display_inline=True or provide out_dir to keep the rendered plots.")

    valid_genes, missing_genes = _normalize_gene_list(adata, genes)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    resolved_segment_kwargs = _default_segment_image_kwargs(
        segmentation_dimensions=segmentation_dimensions,
        segmentation_embedding=segmentation_embedding,
        show_segment_images=show_segment_images,
        verbose=verbose,
    )
    if segment_image_kwargs:
        resolved_segment_kwargs.update(segment_image_kwargs)
    resolved_segment_kwargs.pop("compactness", None)
    resolved_segment_kwargs.pop("Segment", None)

    resolved_image_kwargs = _default_image_plot_kwargs(
        image_dimensions=image_dimensions,
        gene_embedding_key=gene_embedding_key,
    )
    if image_plot_kwargs:
        resolved_image_kwargs.update(image_plot_kwargs)
    resolved_image_kwargs["embedding"] = gene_embedding_key
    resolved_image_kwargs.pop("title", None)

    segment_modes = [
        ("SPIX", spix_segment_key, spix_compactness),
        ("GRID", grid_segment_key, grid_compactness),
    ]
    plot_modes = segment_modes + [("RAW", None, None)]

    for mode_label, segment_key, compactness in segment_modes:
        if force_resegment or segment_key not in adata.obs:
            if verbose:
                print(f"[segment] {mode_label}: building adata.obs['{segment_key}']")
            segment_image(
                adata,
                Segment=segment_key,
                compactness=compactness,
                **resolved_segment_kwargs,
            )
        elif verbose:
            print(f"[segment] {mode_label}: reusing adata.obs['{segment_key}']")

    if missing_genes and verbose:
        print(f"[warn] Missing genes skipped: {', '.join(missing_genes)}")

    saved_paths: list[Path] = []
    embed_kwargs = {"layer": None, "aggregation": "sum"}
    run_parallel = (not display_inline) and (out_dir is not None) and int(max(1, n_jobs)) > 1 and str(backend).lower() == "process"

    if run_parallel:
        if verbose:
            print(f"[parallel] rendering gene panels with processes: jobs={int(max(1, n_jobs))}")
        tasks: list[
            tuple[
                str,
                str | None,
                str,
                str,
                dict[str, Any],
                dict[str, Any],
                bool,
                bool,
                str,
                dict[str, Any] | None,
            ]
        ] = []
        for gene in valid_genes:
            for mode_label, segment_key, _ in plot_modes:
                title = title_template.format(
                    gene=gene,
                    mode=mode_label,
                    segment_key=segment_key if segment_key is not None else "None",
                )
                output_path = out_dir / (
                    f"{_sanitize_path_token(gene)}__{_sanitize_path_token(mode_label.lower())}.png"
                )
                tasks.append(
                    (
                        str(gene),
                        segment_key,
                        str(gene_embedding_key),
                        str(title),
                        dict(resolved_image_kwargs),
                        dict(embed_kwargs),
                        bool(normalize_total),
                        bool(log1p),
                        str(output_path),
                        None if save_kwargs is None else dict(save_kwargs),
                    )
                )
        ctx = mp.get_context("fork")
        worker_adata = _build_minimal_gallery_adata(
            adata,
            segment_keys=[spix_segment_key, grid_segment_key],
        )
        _set_gallery_adata(worker_adata)
        try:
            with ProcessPoolExecutor(
                max_workers=int(max(1, n_jobs)),
                mp_context=ctx,
                initializer=_set_gallery_adata,
                initargs=(worker_adata,),
            ) as ex:
                futs = [ex.submit(_render_gene_mode_to_file, task) for task in tasks]
                for fut in as_completed(futs):
                    out = fut.result()
                    saved_paths.append(Path(out))
                    if verbose:
                        print(f"[saved] {out}")
        finally:
            _set_gallery_adata(None)

        return {
            "genes": valid_genes,
            "missing_genes": missing_genes,
            "saved_paths": saved_paths,
            "spix_segment_key": spix_segment_key,
            "grid_segment_key": grid_segment_key,
        }

    for gene in valid_genes:
        if verbose:
            print(f"[gene] {gene}")

        for mode_label, segment_key, _ in plot_modes:
            add_gene_expression_embedding(
                adata,
                genes=[gene],
                segment_key=segment_key,
                embedding_key=gene_embedding_key,
                normalize_total=normalize_total,
                log1p=log1p,
                **embed_kwargs,
            )
            title = title_template.format(
                gene=gene,
                mode=mode_label,
                segment_key=segment_key if segment_key is not None else "None",
            )

            plot_kwargs = dict(resolved_image_kwargs)
            if out_dir is not None:
                output_path = out_dir / (
                    f"{_sanitize_path_token(gene)}__{_sanitize_path_token(mode_label.lower())}.png"
                )
                plot_kwargs.update({
                    "save": output_path,
                    "save_kwargs": save_kwargs,
                })
                saved_paths.append(output_path)
            plot_kwargs["show"] = bool(display_inline)

            fig_ax = image_plot(adata, title=title, **plot_kwargs)
            fig = fig_ax[0] if isinstance(fig_ax, tuple) else plt.gcf()
            plt.close(fig)

    return {
        "genes": valid_genes,
        "missing_genes": missing_genes,
        "saved_paths": saved_paths,
        "spix_segment_key": spix_segment_key,
        "grid_segment_key": grid_segment_key,
    }


def render_scale_group_top_gene_expression_panels(
    adata: AnnData,
    scale_svg_table: pd.DataFrame,
    *,
    out_dir: str | Path,
    group_col: str = "scale_group",
    gene_col: str = "gene",
    pass_col: str | None = "passes_rule",
    require_pass: bool = True,
    top_n_per_group: int = 12,
    modes: Sequence[tuple[str, str | None]] = (("segment", "Segment"), ("raw", None)),
    mode_plot_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    base_plot_kwargs: Mapping[str, Any] | None = None,
    layer: str | None = None,
    normalize_total: bool = True,
    log1p: bool = True,
    aggregation: str = "sum",
    embedding_key: str = "X_gene_embedding",
    n_jobs: int = 1,
    backend: str = "process",
    make_contact_sheets: bool = True,
    contact_sheet_ncols: int = 4,
    contact_sheet_dpi: int = 180,
    save_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Render top scale-group genes with `plot_gene_expression` settings.

    The function saves individual gene images first, optionally then assembles
    group/mode contact sheets. It is designed for notebook triage of top
    scale-SVG genes where inline rendering would be too slow or noisy.
    """
    try:
        from .scale_biology import make_scale_group_gene_sets
    except ImportError:
        # Notebook kernels can keep an older scale_biology module in sys.modules
        # after SPIX.an has been reloaded. Refresh it once before failing.
        import importlib

        scale_biology = importlib.import_module(".scale_biology", package=__package__)
        scale_biology = importlib.reload(scale_biology)
        make_scale_group_gene_sets = scale_biology.make_scale_group_gene_sets

    if scale_svg_table is None or scale_svg_table.empty:
        raise ValueError("scale_svg_table must be a non-empty DataFrame.")
    out_dir = Path(out_dir)
    image_dir = out_dir / "individual"
    panel_dir = out_dir / "panels"
    image_dir.mkdir(parents=True, exist_ok=True)
    if make_contact_sheets:
        panel_dir.mkdir(parents=True, exist_ok=True)

    gene_sets = make_scale_group_gene_sets(
        scale_svg_table,
        group_col=group_col,
        gene_col=gene_col,
        pass_col=pass_col,
        require_pass=require_pass,
        top_n_per_group=int(top_n_per_group),
    )
    if not gene_sets:
        raise ValueError("No genes were selected from scale_svg_table.")

    all_genes = [gene for genes in gene_sets.values() for gene in genes]
    valid_genes, missing_genes = _normalize_gene_list(adata, all_genes)
    valid_set = set(valid_genes)
    gene_sets = {
        group: [gene for gene in genes if gene in valid_set]
        for group, genes in gene_sets.items()
    }
    gene_sets = {group: genes for group, genes in gene_sets.items() if genes}
    if not gene_sets:
        raise ValueError("No selected genes were found in adata.var_names.")
    if missing_genes and verbose:
        print(f"[warn] Missing genes skipped: {', '.join(missing_genes)}")

    plot_kwargs_base = {
        "figsize": (10, 10),
        "fig_dpi": 100,
        "dimensions": [0],
        "embedding": embedding_key,
        "boundary_method": "pixel",
        "show_colorbar": True,
        "legend_ncol": 1,
        "runtime_fill_from_boundary": False,
        "runtime_fill_holes": True,
        "resolve_center_collisions": True,
        "title": None,
        "cmap": "viridis",
        "plot_boundaries": False,
    }
    if base_plot_kwargs:
        plot_kwargs_base.update(dict(base_plot_kwargs))
    plot_kwargs_base.pop("title", None)
    plot_kwargs_base["embedding"] = embedding_key

    mode_overrides = {str(k): dict(v) for k, v in (mode_plot_kwargs or {}).items()}
    tasks: list[
        tuple[
            str,
            str,
            str | None,
            str,
            str,
            dict[str, Any],
            str | None,
            bool,
            bool,
            str,
            str,
            dict[str, Any] | None,
        ]
    ] = []
    task_records: list[dict[str, Any]] = []
    for group, genes in gene_sets.items():
        group_token = _sanitize_path_token(group)
        for gene in genes:
            gene_token = _sanitize_path_token(gene)
            for mode_label, segment_key in modes:
                mode_label = str(mode_label)
                mode_token = _sanitize_path_token(mode_label)
                plot_kwargs = dict(plot_kwargs_base)
                plot_kwargs.update(mode_overrides.get(mode_label, {}))
                out_path = image_dir / group_token / mode_token / f"{gene_token}.png"
                task = (
                    str(group),
                    str(gene),
                    segment_key,
                    mode_label,
                    str(embedding_key),
                    plot_kwargs,
                    layer,
                    bool(normalize_total),
                    bool(log1p),
                    str(aggregation),
                    str(out_path),
                    None if save_kwargs is None else dict(save_kwargs),
                )
                tasks.append(task)
                task_records.append(
                    {
                        "scale_group": str(group),
                        "gene": str(gene),
                        "mode": mode_label,
                        "segment_key": segment_key,
                        "path": out_path,
                    }
                )

    saved_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_parallel = int(max(1, n_jobs)) > 1 and str(backend).lower() == "process"

    if run_parallel:
        if verbose:
            print(f"[parallel] rendering {len(tasks)} images with processes: jobs={int(max(1, n_jobs))}")
        segment_keys = [segment_key for _, segment_key in modes if segment_key is not None]
        worker_adata = _build_minimal_gallery_adata(adata, segment_keys=segment_keys)
        ctx = mp.get_context("fork")
        _set_gallery_adata(worker_adata)
        try:
            with ProcessPoolExecutor(
                max_workers=int(max(1, n_jobs)),
                mp_context=ctx,
                initializer=_set_gallery_adata,
                initargs=(worker_adata,),
            ) as ex:
                future_to_task = {ex.submit(_render_gene_expression_to_file, task): task for task in tasks}
                for fut in as_completed(future_to_task):
                    task = future_to_task[fut]
                    try:
                        record = fut.result()
                        saved_records.append(record)
                        if verbose:
                            print(f"[saved] {record['path']}")
                    except Exception as exc:
                        failures.append(
                            {
                                "scale_group": task[0],
                                "gene": task[1],
                                "mode": task[3],
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if verbose:
                            print(f"[fail] {task[0]} {task[1]} {task[3]}: {exc}")
        finally:
            _set_gallery_adata(None)
    else:
        if verbose:
            print(f"[serial] rendering {len(tasks)} images")
        for task in tasks:
            try:
                _set_gallery_adata(adata)
                record = _render_gene_expression_to_file(task)
                saved_records.append(record)
                if verbose:
                    print(f"[saved] {record['path']}")
            except Exception as exc:
                failures.append(
                    {
                        "scale_group": task[0],
                        "gene": task[1],
                        "mode": task[3],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if verbose:
                    print(f"[fail] {task[0]} {task[1]} {task[3]}: {exc}")
        _set_gallery_adata(None)

    saved_df = pd.DataFrame(saved_records)
    failures_df = pd.DataFrame(failures)
    if not saved_df.empty:
        saved_df.to_csv(out_dir / "rendered_gene_expression_images.csv", index=False)
    if not failures_df.empty:
        failures_df.to_csv(out_dir / "render_failures.csv", index=False)

    panel_paths: list[Path] = []
    if make_contact_sheets and not saved_df.empty:
        for (group, mode), sub in saved_df.groupby(["scale_group", "mode"], sort=False):
            order = {gene: i for i, gene in enumerate(gene_sets.get(str(group), []))}
            records = sub.assign(_order=sub["gene"].map(order).fillna(10**9)).sort_values("_order").to_dict("records")
            out_path = panel_dir / f"{_sanitize_path_token(group)}__{_sanitize_path_token(mode)}.png"
            panel_paths.append(
                _make_contact_sheet(
                    records,
                    out_path=out_path,
                    ncols=int(contact_sheet_ncols),
                    title=f"{group} | {mode}",
                    dpi=int(contact_sheet_dpi),
                )
            )
            if verbose:
                print(f"[panel] {out_path}")

    return {
        "gene_sets": gene_sets,
        "missing_genes": missing_genes,
        "saved_images": saved_df,
        "failures": failures_df,
        "panel_paths": panel_paths,
        "out_dir": out_dir,
    }


def plot_gene_expression(
    adata: AnnData,
    gene: str,
    *,
    segment_key: str | None = "Segment",
    layer: str | None = None,
    normalize_total: bool = True,
    log1p: bool = True,
    aggregation: str = "sum",
    embedding_key: str = "X_gene_embedding",
    image_dimensions: Sequence[int] = (0,),
    title: str | None = None,
    save: str | Path | None = None,
    save_kwargs: dict[str, Any] | None = None,
    show: bool = True,
    return_fig: bool | None = None,
    point_size_range: tuple[float, float] | None = None,
    **image_plot_kwargs: Any,
):
    """Convenience wrapper to visualize one gene with `image_plot`.

    This mirrors the common notebook pattern:
    1. build a one-gene embedding with `add_gene_expression_embedding`
    2. call `image_plot(...)`

    Parameters
    ----------
    adata
        AnnData object.
    gene
        Gene name to visualize.
    segment_key
        Segment key used for segment-level aggregation. Set to ``None`` for raw per-cell plotting.
    layer
        Optional layer to use instead of ``adata.X``.
    normalize_total
        Whether to normalize total counts before plotting.
    log1p
        Whether to apply ``log1p`` after normalization.
    aggregation
        Aggregation mode passed to `add_gene_expression_embedding`.
    embedding_key
        Temporary obsm key used for the generated gene embedding.
    image_dimensions
        Dimensions passed to `image_plot`; default `(0,)`.
    title
        Plot title. Defaults to the gene name.
    save
        Optional output path. If provided, the generated image plot is saved.
    save_kwargs
        Extra keyword arguments passed to ``Figure.savefig`` through ``image_plot``.
    show
        Whether to display the plot with ``plt.show()``.
    return_fig
        If ``True``, return ``(fig, ax)``. If ``None``, returns ``(fig, ax)``
        when ``show=False``.
    point_size_range
        Optional ``(min_size, max_size)`` marker-size range. If provided, the
        base image plot is kept unchanged and expression-positive points are
        overlaid with marker sizes linearly mapped from this gene's expression.
    **image_plot_kwargs
        Extra keyword arguments forwarded directly to `image_plot`. Common overrides include
        `figsize`, `cmap`, `fixed_boundary_color`, `plot_boundaries`, and `origin`.
    """
    add_gene_expression_embedding(
        adata,
        genes=[str(gene)],
        layer=layer,
        segment_key=segment_key,
        embedding_key=embedding_key,
        normalize_total=normalize_total,
        log1p=log1p,
        aggregation=aggregation,
    )

    resolved_image_kwargs = _default_image_plot_kwargs(
        image_dimensions=image_dimensions,
        gene_embedding_key=embedding_key,
    )
    if image_plot_kwargs:
        resolved_image_kwargs.update(image_plot_kwargs)
    # Keep boundary rendering and any other segment-aware image_plot behavior
    # aligned with the aggregation segment_key unless the caller explicitly
    # overrides it through image_plot kwargs.
    if segment_key is not None and "segment_key" not in resolved_image_kwargs:
        resolved_image_kwargs["segment_key"] = segment_key
    resolved_image_kwargs["embedding"] = embedding_key
    if point_size_range is None:
        resolved_image_kwargs["save"] = save
        resolved_image_kwargs["save_kwargs"] = save_kwargs
        resolved_image_kwargs["show"] = show
        resolved_image_kwargs["return_fig"] = return_fig

        return image_plot(
            adata,
            title=str(gene) if title is None else str(title),
            **resolved_image_kwargs,
        )

    resolved_image_kwargs["save"] = None
    resolved_image_kwargs["save_kwargs"] = None
    resolved_image_kwargs["show"] = False
    resolved_image_kwargs["return_fig"] = True
    fig, ax = image_plot(
        adata,
        title=str(gene) if title is None else str(title),
        **resolved_image_kwargs,
    )

    coords = np.asarray(adata.obsm.get("spatial"))
    values = np.asarray(adata.obsm[embedding_key]).reshape(-1)
    sizes = _gene_expression_point_sizes(values, point_size_range)
    min_size = float(point_size_range[0])
    finite = np.isfinite(values)
    positive = finite & (sizes > min_size)
    if coords.ndim == 2 and coords.shape[0] == adata.n_obs and coords.shape[1] >= 2 and positive.any():
        colors = _gene_expression_overlay_colors(
            adata,
            resolved_image_kwargs.get("color_by", None),
            resolved_image_kwargs.get("palette", None),
        )
        if not isinstance(colors, str):
            colors = colors[positive]
        pixel_shape = str(resolved_image_kwargs.get("pixel_shape", "square")).lower()
        ax.scatter(
            coords[positive, 0],
            coords[positive, 1],
            s=sizes[positive],
            c=colors,
            marker="o" if pixel_shape == "circle" else "s",
            linewidths=0,
            alpha=0.95,
            zorder=10,
        )
    if save is not None:
        save_path = Path(save)
        if save_path.parent:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.03}
        if save_kwargs:
            kwargs.update(dict(save_kwargs))
        fig.savefig(save_path, **kwargs)
    if show:
        plt.show()
    if return_fig is None:
        return_fig = not bool(show)
    if return_fig:
        return fig, ax
    return None
