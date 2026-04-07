from __future__ import annotations

import re
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence
import multiprocessing as mp

import matplotlib.pyplot as plt
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
) -> AnnData:
    """Build a lightweight AnnData for gene-expression rendering workers.

    Process-parallel rendering does not need the full SPIX object. Keeping only
    X, var_names, the requested segment columns, and spatial coordinates
    substantially lowers worker memory use.
    """
    obs_cols = [str(k) for k in segment_keys if (k is not None and str(k) in adata.obs.columns)]
    obs = adata.obs.loc[:, obs_cols].copy() if obs_cols else pd.DataFrame(index=adata.obs.index.copy())
    var = pd.DataFrame(index=adata.var_names.copy())
    mini = AnnData(X=adata.X, obs=obs, var=var)
    if "spatial" in adata.obsm:
        mini.obsm["spatial"] = np.asarray(adata.obsm["spatial"])
    return mini


def _sanitize_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return token or "item"


@contextmanager
def _patched_pyplot_show(suppress: bool):
    if not suppress:
        yield
        return

    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = original_show


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


def _render_gene_mode_to_file(task: tuple[str, str | None, str, str, dict[str, Any], dict[str, Any], bool, bool, str]) -> Path:
    gene, segment_key, embedding_key, title, resolved_image_kwargs, embed_kwargs, normalize_total, log1p, output_path = task
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

    with _patched_pyplot_show(suppress=True):
        image_plot(adata, title=title, **resolved_image_kwargs)

    fig = plt.gcf()
    out = Path(output_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


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
    n_jobs: int = 1,
    backend: str = "process",
) -> dict[str, Any]:
    """Render SPIX, GRID, and raw gene-expression image plots for each gene.

    The defaults mirror the repeated cells in `SPIX_figure2_tmp.ipynb`, but the
    segmentation outputs are stored under separate keys so they can be reused
    without overwriting `adata.obs['Segment']`.
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
        tasks: list[tuple[str, str | None, str, str, dict[str, Any], dict[str, Any], bool, bool, str]] = []
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

            with _patched_pyplot_show(suppress=not display_inline):
                image_plot(adata, title=title, **resolved_image_kwargs)

            fig = plt.gcf()
            if out_dir is not None:
                output_path = out_dir / (
                    f"{_sanitize_path_token(gene)}__{_sanitize_path_token(mode_label.lower())}.png"
                )
                fig.savefig(output_path, bbox_inches="tight")
                saved_paths.append(output_path)
            plt.close(fig)

    return {
        "genes": valid_genes,
        "missing_genes": missing_genes,
        "saved_paths": saved_paths,
        "spix_segment_key": spix_segment_key,
        "grid_segment_key": grid_segment_key,
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

    return image_plot(
        adata,
        title=str(gene) if title is None else str(title),
        **resolved_image_kwargs,
    )
