from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CATEGORY_COLORS: Dict[str, str] = {
    "SPIX": "#d95f02",
    "GRID": "#1b9e77",
    "mixed": "#bdbdbd",
}


def _check_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _ordered_categories(
    df: pd.DataFrame,
    *,
    category_col: str,
    focus: str,
    reference: str,
) -> List[str]:
    existing = [str(v) for v in pd.Index(df[category_col].dropna().unique()).tolist()]
    ordered: List[str] = []
    for cat in (focus, reference, "mixed"):
        if cat in existing and cat not in ordered:
            ordered.append(cat)
    for cat in sorted(existing):
        if cat not in ordered:
            ordered.append(cat)
    return ordered


def annotate_svg_specificity(
    summary_df: pd.DataFrame,
    *,
    focus: str = "SPIX",
    reference: str = "GRID",
    category_col: str = "category",
    eps: float = 1e-6,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add threshold-free contrast columns to the notebook summary table.

    The current notebook already computes per-gene `mean_SPIX`, `mean_GRID`,
    `specificity_score`, and `category`. This function turns that table into a
    more quantifiable one by adding:

    - `contrast_<focus>_vs_<reference>`:
      positive when the gene ranks better in `focus` (lower normalized rank)
    - `contrast_abs_<focus>_vs_<reference>`:
      absolute separation between the two groups
    - `contrast_ratio_<focus>_vs_<reference>`:
      contrast scaled by the sum of both means
    - `specificity_percentile`:
      rank percentile of `specificity_score`
    """

    focus_col = f"mean_{focus}"
    reference_col = f"mean_{reference}"
    _check_required_columns(summary_df, [focus_col, reference_col, category_col, "specificity_score"])

    df = summary_df.copy() if copy else summary_df

    df[focus_col] = pd.to_numeric(df[focus_col], errors="coerce")
    df[reference_col] = pd.to_numeric(df[reference_col], errors="coerce")
    df["specificity_score"] = pd.to_numeric(df["specificity_score"], errors="coerce")

    if "gap_abs" in df.columns:
        df["gap_abs"] = pd.to_numeric(df["gap_abs"], errors="coerce")
    if "gap_rel" in df.columns:
        df["gap_rel"] = pd.to_numeric(df["gap_rel"], errors="coerce")
    if "best_resolution_margin" in df.columns:
        df["best_resolution_margin"] = pd.to_numeric(df["best_resolution_margin"], errors="coerce")

    contrast_col = f"contrast_{focus}_vs_{reference}"
    contrast_abs_col = f"contrast_abs_{focus}_vs_{reference}"
    contrast_ratio_col = f"contrast_ratio_{focus}_vs_{reference}"

    df[contrast_col] = df[reference_col] - df[focus_col]
    df[contrast_abs_col] = df[contrast_col].abs()
    df[contrast_ratio_col] = df[contrast_col] / (df[reference_col] + df[focus_col] + float(eps))
    df["specificity_percentile"] = df["specificity_score"].rank(pct=True, method="average")

    return df


def summarize_svg_specificity(
    summary_df: pd.DataFrame,
    *,
    focus: str = "SPIX",
    reference: str = "GRID",
    category_col: str = "category",
) -> pd.DataFrame:
    """
    Summarize the SPIX/GRID/mixed split with stable numeric metrics.
    """

    df = annotate_svg_specificity(
        summary_df,
        focus=focus,
        reference=reference,
        category_col=category_col,
        copy=True,
    )

    categories = _ordered_categories(df, category_col=category_col, focus=focus, reference=reference)
    focus_col = f"mean_{focus}"
    reference_col = f"mean_{reference}"
    contrast_col = f"contrast_{focus}_vs_{reference}"
    contrast_abs_col = f"contrast_abs_{focus}_vs_{reference}"

    rows: List[Dict[str, object]] = []
    total = max(1, int(len(df)))

    for cat in categories:
        sub = df.loc[df[category_col] == cat].copy()
        if sub.empty:
            continue

        sort_cols = [c for c in ["specificity_score", contrast_abs_col, "best_resolution_margin", "gap_abs"] if c in sub.columns]
        top_gene = str(sub.sort_values(sort_cols, ascending=False).index[0]) if sort_cols else str(sub.index[0])

        row: Dict[str, object] = {
            "category": cat,
            "n_genes": int(len(sub)),
            "fraction": float(len(sub) / total),
            f"median_{focus_col}": float(sub[focus_col].median()),
            f"median_{reference_col}": float(sub[reference_col].median()),
            f"median_{contrast_col}": float(sub[contrast_col].median()),
            f"median_{contrast_abs_col}": float(sub[contrast_abs_col].median()),
            "median_specificity_score": float(sub["specificity_score"].median()),
            "median_specificity_percentile": float(sub["specificity_percentile"].median()),
            "top_example": top_gene,
        }

        if "gap_abs" in sub.columns:
            row["median_gap_abs"] = float(sub["gap_abs"].median())
        if "gap_rel" in sub.columns:
            row["median_gap_rel"] = float(sub["gap_rel"].median())
        if "best_resolution_margin" in sub.columns:
            row["median_best_resolution_margin"] = float(sub["best_resolution_margin"].median())

        rows.append(row)

    out = pd.DataFrame(rows).set_index("category")
    if not out.empty:
        out = out.sort_values("n_genes", ascending=False)
    return out


def _boxplot_by_category(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    categories: List[str],
    category_col: str,
    value_col: str,
    colors: Dict[str, str],
    title: str,
    ylabel: str,
) -> None:
    series = [df.loc[df[category_col] == cat, value_col].dropna().to_numpy() for cat in categories]
    non_empty = [len(x) for x in series if len(x) > 0]
    if not non_empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    bp = ax.boxplot(series, labels=categories, patch_artist=True, showfliers=False)
    for patch, cat in zip(bp["boxes"], categories):
        patch.set_facecolor(colors.get(cat, "#cccccc"))
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)


def plot_svg_specificity_dashboard(
    summary_df: pd.DataFrame,
    *,
    focus: str = "SPIX",
    reference: str = "GRID",
    category_col: str = "category",
    figsize: Tuple[float, float] = (14.0, 10.0),
    top_n_labels: int = 8,
    colors: Dict[str, str] | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a compact dashboard for SPIX-specific SVG quantification.

    Panels:
    1. counts per category
    2. specificity score distribution
    3. threshold-free scatter: contrast vs. specificity score
    4. contrast distribution
    """

    df = annotate_svg_specificity(
        summary_df,
        focus=focus,
        reference=reference,
        category_col=category_col,
        copy=True,
    )
    _check_required_columns(df, [category_col, "specificity_score"])

    focus_col = f"mean_{focus}"
    reference_col = f"mean_{reference}"
    contrast_col = f"contrast_{focus}_vs_{reference}"
    contrast_abs_col = f"contrast_abs_{focus}_vs_{reference}"

    colors = {**DEFAULT_CATEGORY_COLORS, **(colors or {})}
    categories = _ordered_categories(df, category_col=category_col, focus=focus, reference=reference)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax_count, ax_score, ax_scatter, ax_contrast = axes.ravel()

    counts = df[category_col].value_counts().reindex(categories).fillna(0).astype(int)
    ax_count.bar(
        counts.index,
        counts.values,
        color=[colors.get(cat, "#cccccc") for cat in counts.index],
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax_count.set_title("SVG Counts by Category")
    ax_count.set_ylabel("Gene count")
    ax_count.grid(axis="y", alpha=0.2)

    _boxplot_by_category(
        ax_score,
        df,
        categories=categories,
        category_col=category_col,
        value_col="specificity_score",
        colors=colors,
        title="Specificity Score Distribution",
        ylabel="specificity_score",
    )

    margin = (
        pd.to_numeric(df["best_resolution_margin"], errors="coerce")
        if "best_resolution_margin" in df.columns
        else pd.Series(0.0, index=df.index)
    ).fillna(0.0)
    sizes = 28.0 + 240.0 * np.clip(margin.to_numpy(dtype=float), 0.0, None)

    for cat in categories:
        sub = df.loc[df[category_col] == cat]
        if sub.empty:
            continue
        ax_scatter.scatter(
            sub[contrast_col],
            sub["specificity_score"],
            s=sizes[df.index.get_indexer(sub.index)],
            alpha=0.75,
            color=colors.get(cat, "#cccccc"),
            edgecolors="white",
            linewidths=0.4,
            label=cat,
        )

    ax_scatter.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_scatter.set_title(f"{focus} Preference vs. Specificity")
    ax_scatter.set_xlabel(f"{reference_col} - {focus_col}  (positive => {focus} better)")
    ax_scatter.set_ylabel("specificity_score")
    ax_scatter.grid(alpha=0.2)

    focus_only = df.loc[df[category_col] == focus].copy()
    if not focus_only.empty and int(top_n_labels) > 0:
        sort_cols = [c for c in ["specificity_score", contrast_abs_col, "best_resolution_margin", "gap_abs"] if c in focus_only.columns]
        top = focus_only.sort_values(sort_cols, ascending=False).head(int(top_n_labels))
        for gene, row in top.iterrows():
            ax_scatter.annotate(
                str(gene),
                (row[contrast_col], row["specificity_score"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color="#333333",
            )

    ax_scatter.legend(frameon=False, loc="best")

    _boxplot_by_category(
        ax_contrast,
        df,
        categories=categories,
        category_col=category_col,
        value_col=contrast_col,
        colors=colors,
        title="Contrast Distribution",
        ylabel=f"{reference_col} - {focus_col}",
    )
    ax_contrast.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)

    return fig, axes
