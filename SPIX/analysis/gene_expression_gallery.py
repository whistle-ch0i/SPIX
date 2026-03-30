from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from anndata import AnnData

from ..superpixel.segmentation import segment_image
from ..visualization.plotting import image_plot
from .gene_expression_embedding import add_gene_expression_embedding


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
        "boundary_method": "pixel",
        "figsize": (10, 10),
        "fixed_boundary_color": "Black",
        "cmap": "viridis",
        "boundary_linewidth": 1,
        "show_colorbar": True,
        "prioritize_high_values": True,
        "runtime_fill_from_boundary": True,
        "runtime_fill_closing_radius": 2,
        "runtime_fill_holes": True,
        "soft_rasterization": True,
        "resolve_center_collisions": False,
        "alpha": 1,
        "plot_boundaries": False,
        "origin": True,
    }


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
