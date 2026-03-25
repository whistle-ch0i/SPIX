from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


DEFAULT_CATEGORY_ORDER = ("early", "mid", "late")
DEFAULT_CATEGORY_COLORS = {
    "early": "#4C78A8",
    "mid": "#72B7B2",
    "late": "#E45756",
    "mixed": "#B8B8B8",
}


def _resolution_sort_key(label: str) -> float:
    text = str(label)
    if text.startswith("res_"):
        text = text[4:]
    elif text.startswith("rank_r") and "_c" in text:
        text = text[len("rank_r") :].split("_c", 1)[0]
    elif text.startswith("r") and "_c" in text:
        text = text[1:].split("_c", 1)[0]
    try:
        return float(text)
    except Exception:
        return float("inf")


def _resolution_label(label: str, resolution_label_map: Optional[Mapping[int, str]] = None) -> str:
    value = _resolution_sort_key(label)
    if np.isfinite(value):
        value_int = int(value)
        if resolution_label_map and value_int in resolution_label_map:
            return str(resolution_label_map[value_int])
        return str(value_int)
    return str(label)


def _normalized_score_df(traj_df: pd.DataFrame) -> pd.DataFrame:
    score_df = traj_df.copy().apply(pd.to_numeric, errors="coerce")
    score_df.index = score_df.index.astype(str)
    ordered_cols = sorted(score_df.columns.astype(str).tolist(), key=_resolution_sort_key)
    score_df = score_df[ordered_cols]
    score_df = 1.0 - score_df.divide(score_df.max(axis=0).replace(0, np.nan), axis=1).fillna(1.0)
    return score_df


def _top_category_genes(summary_df: pd.DataFrame, category: str, *, n: int) -> pd.DataFrame:
    sub = summary_df[summary_df["category"].astype(str) == str(category)].copy()
    if sub.empty:
        return sub
    sort_cols = [
        col
        for col in ["specificity_score", "gap_abs", "best_resolution_margin", f"mean_{category}"]
        if col in sub.columns
    ]
    ascending = [False, False, False, True][: len(sort_cols)]
    return sub.sort_values(sort_cols, ascending=ascending).head(int(n))


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _plot_counts(ax, summary_df: pd.DataFrame, *, category_order: Sequence[str], colors: Mapping[str, str]) -> pd.DataFrame:
    counts = summary_df["category"].astype(str).value_counts()
    plot_df = pd.DataFrame(
        {
            "category": list(category_order),
            "n_genes": [int(counts.get(cat, 0)) for cat in category_order],
        }
    )
    ax.bar(
        plot_df["category"],
        plot_df["n_genes"],
        color=[colors.get(cat, "#999999") for cat in plot_df["category"]],
        width=0.72,
    )
    for row in plot_df.itertuples(index=False):
        ax.text(row.category, row.n_genes + max(1, row.n_genes * 0.02), f"{row.n_genes}", ha="center", va="bottom", fontsize=10)
    mixed_n = int(counts.get("mixed", 0))
    ax.text(
        0.98,
        0.94,
        f"mixed not shown\nn={mixed_n:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F3F3F3", edgecolor="none"),
    )
    ax.set_title("Geary-specific SVG counts", fontweight="bold", pad=10)
    ax.set_xlabel("Category")
    ax.set_ylabel("Gene count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return plot_df


def _plot_specificity_distribution(
    ax,
    summary_df: pd.DataFrame,
    *,
    category_order: Sequence[str],
    colors: Mapping[str, str],
) -> pd.DataFrame:
    plot_df = summary_df[summary_df["category"].astype(str).isin(category_order)].copy()
    plot_df["category"] = pd.Categorical(plot_df["category"].astype(str), categories=list(category_order), ordered=True)
    plot_df["specificity_score"] = pd.to_numeric(plot_df["specificity_score"], errors="coerce")
    sns.violinplot(
        data=plot_df,
        x="category",
        y="specificity_score",
        order=list(category_order),
        palette=[colors.get(cat, "#999999") for cat in category_order],
        inner=None,
        cut=0,
        linewidth=0,
        ax=ax,
    )
    sns.boxplot(
        data=plot_df,
        x="category",
        y="specificity_score",
        order=list(category_order),
        width=0.22,
        showcaps=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
        whiskerprops={"linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.2},
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Specificity-score distribution", fontweight="bold", pad=10)
    ax.set_xlabel("Category")
    ax.set_ylabel("Specificity score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return plot_df


def _plot_best_resolution_distribution(
    ax,
    summary_df: pd.DataFrame,
    *,
    category_order: Sequence[str],
    colors: Mapping[str, str],
    resolution_label_map: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    plot_df = summary_df[summary_df["category"].astype(str).isin(category_order)].copy()
    plot_df["best_resolution"] = plot_df["best_resolution"].astype(str)
    count_df = (
        plot_df.groupby(["best_resolution", "category"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    count_df = count_df.reindex(
        index=sorted(count_df.index.astype(str).tolist(), key=_resolution_sort_key),
        fill_value=0,
    )
    count_df = count_df.reindex(columns=list(category_order), fill_value=0)

    x = np.arange(count_df.shape[0])
    bottom = np.zeros(count_df.shape[0], dtype=float)
    for category in category_order:
        vals = count_df[category].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.76, color=colors.get(category, "#999999"), label=category)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([_resolution_label(idx, resolution_label_map) for idx in count_df.index], rotation=0)
    ax.set_title("Best-resolution distribution", fontweight="bold", pad=10)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Gene count")
    ax.legend(frameon=False, title="", ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return count_df


def _plot_category_centroid_curves(
    ax,
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    category_order: Sequence[str],
    colors: Mapping[str, str],
    resolution_label_map: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    x = np.arange(score_df.shape[1])
    rows = []
    for category in category_order:
        genes = summary_df.index[summary_df["category"].astype(str) == str(category)].astype(str)
        genes = [gene for gene in genes if gene in score_df.index]
        if not genes:
            continue
        mean_score = score_df.loc[genes].mean(axis=0)
        ax.plot(
            x,
            mean_score.to_numpy(dtype=float),
            "-o",
            lw=2.4,
            ms=5.5,
            color=colors.get(category, "#999999"),
            label=f"{category} (n={len(genes)})",
        )
        for col, val in mean_score.items():
            rows.append({"category": category, "resolution": str(col), "mean_score": float(val)})

    ax.set_xticks(x)
    ax.set_xticklabels([_resolution_label(col, resolution_label_map) for col in score_df.columns], rotation=45, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Category centroid trajectories", fontweight="bold", pad=10)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mean rank score")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=9, loc="best")
    return pd.DataFrame(rows)


def _plot_top_trajectories(
    ax,
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    category: str,
    *,
    n_genes: int,
    colors: Mapping[str, str],
    resolution_label_map: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    top_df = _top_category_genes(summary_df, category, n=n_genes).copy()
    genes = [gene for gene in top_df.index.astype(str).tolist() if gene in score_df.index]
    x = np.arange(score_df.shape[1])
    palette = sns.light_palette(colors.get(category, "#666666"), n_colors=max(len(genes), 3) + 2, reverse=True)
    for idx, gene in enumerate(genes):
        ax.plot(
            x,
            score_df.loc[gene].to_numpy(dtype=float),
            "-o",
            lw=1.9,
            ms=4.2,
            color=palette[idx + 1],
            label=gene,
            alpha=0.95,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_resolution_label(col, resolution_label_map) for col in score_df.columns], rotation=45, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Top {category} markers", fontweight="bold", pad=10)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Rank score")
    ax.grid(alpha=0.18)
    if genes:
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    return top_df.loc[genes].copy()


def _plot_heatmap(
    ax,
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    category_order: Sequence[str],
    top_n_per_category: int,
    resolution_label_map: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    selected_parts = []
    for category in category_order:
        top_df = _top_category_genes(summary_df, category, n=top_n_per_category).copy()
        if top_df.empty:
            continue
        top_df["category"] = category
        selected_parts.append(top_df)

    if not selected_parts:
        ax.text(0.5, 0.5, "No non-mixed genes", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return pd.DataFrame()

    selected_df = pd.concat(selected_parts, axis=0)
    selected_df["gene"] = selected_df.index.astype(str)
    genes = [gene for gene in selected_df["gene"].tolist() if gene in score_df.index]
    selected_df = selected_df.set_index("gene").loc[genes].reset_index()
    heat_df = score_df.loc[genes].copy()
    heat_df.index = [f"{row.gene} ({row.category})" for row in selected_df.itertuples(index=False)]
    heat_df.columns = [_resolution_label(col, resolution_label_map) for col in heat_df.columns]

    sns.heatmap(
        heat_df,
        cmap="mako",
        vmin=0,
        vmax=1,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Rank score"},
        ax=ax,
    )
    ax.set_title(f"Top {top_n_per_category} genes per category", fontweight="bold", pad=10)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    return selected_df


def build_geary_supplementary_figure(
    geary_traj_df: pd.DataFrame,
    geary_summary_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    resolution_label_map: Optional[Mapping[int, str]] = None,
    category_order: Sequence[str] = DEFAULT_CATEGORY_ORDER,
    category_colors: Optional[Mapping[str, str]] = None,
    top_n_per_category: int = 5,
    figsize: tuple[float, float] = (15.5, 11.5),
    dpi: int = 180,
    save_pdf: bool = False,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    category_order = [str(cat) for cat in category_order]
    colors = dict(DEFAULT_CATEGORY_COLORS)
    if category_colors is not None:
        colors.update({str(k): v for k, v in category_colors.items()})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_df = geary_traj_df.copy()
    if "Unnamed: 0" in traj_df.columns:
        traj_df = traj_df.set_index("Unnamed: 0")
    traj_df.index = traj_df.index.astype(str)
    score_df = _normalized_score_df(traj_df)

    summary_df = geary_summary_df.copy()
    if "Unnamed: 0" in summary_df.columns:
        summary_df = summary_df.set_index("Unnamed: 0")
    summary_df.index = summary_df.index.astype(str)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
        }
    )

    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 1.25], hspace=0.52, wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2:4])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[1, 3])
    ax_h = fig.add_subplot(gs[2, :])

    count_df = _plot_counts(ax_a, summary_df, category_order=category_order, colors=colors)
    spec_df = _plot_specificity_distribution(ax_b, summary_df, category_order=category_order, colors=colors)
    best_res_df = _plot_best_resolution_distribution(
        ax_c,
        summary_df,
        category_order=category_order,
        colors=colors,
        resolution_label_map=resolution_label_map,
    )
    centroid_df = _plot_category_centroid_curves(
        ax_d,
        score_df,
        summary_df,
        category_order=category_order,
        colors=colors,
        resolution_label_map=resolution_label_map,
    )
    early_df = _plot_top_trajectories(
        ax_e,
        score_df,
        summary_df,
        "early",
        n_genes=top_n_per_category,
        colors=colors,
        resolution_label_map=resolution_label_map,
    )
    mid_df = _plot_top_trajectories(
        ax_f,
        score_df,
        summary_df,
        "mid",
        n_genes=top_n_per_category,
        colors=colors,
        resolution_label_map=resolution_label_map,
    )
    late_df = _plot_top_trajectories(
        ax_g,
        score_df,
        summary_df,
        "late",
        n_genes=top_n_per_category,
        colors=colors,
        resolution_label_map=resolution_label_map,
    )
    heatmap_df = _plot_heatmap(
        ax_h,
        score_df,
        summary_df,
        category_order=category_order,
        top_n_per_category=top_n_per_category,
        resolution_label_map=resolution_label_map,
    )

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h], list("ABCDEFGH")):
        _panel_label(ax, label)

    fig.suptitle("Supplementary Figure: Geary's C multiscale SVG summary", fontsize=18, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.972])

    png_path = out_dir / "geary_supplementary_figure.png"
    pdf_path = out_dir / "geary_supplementary_figure.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    if save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    selected_genes_parts = []
    for category, df in [("early", early_df), ("mid", mid_df), ("late", late_df)]:
        if df.empty:
            continue
        tmp = df.copy()
        tmp["gene"] = tmp.index.astype(str)
        tmp["category"] = category
        selected_genes_parts.append(tmp.reset_index(drop=True))
    selected_genes_df = pd.concat(selected_genes_parts, axis=0, ignore_index=True) if selected_genes_parts else pd.DataFrame()

    count_df.to_csv(out_dir / "geary_supplementary_category_counts.csv", index=False)
    spec_df.to_csv(out_dir / "geary_supplementary_specificity_distribution.csv", index=False)
    best_res_df.to_csv(out_dir / "geary_supplementary_best_resolution_counts.csv")
    centroid_df.to_csv(out_dir / "geary_supplementary_centroid_scores.csv", index=False)
    selected_genes_df.to_csv(out_dir / "geary_supplementary_top_trajectory_genes.csv", index=False)
    heatmap_df.to_csv(out_dir / "geary_supplementary_heatmap_genes.csv", index=False)

    return {
        "category_counts_df": count_df,
        "specificity_distribution_df": spec_df,
        "best_resolution_count_df": best_res_df,
        "centroid_df": centroid_df,
        "selected_genes_df": selected_genes_df,
        "heatmap_genes_df": heatmap_df,
        "png_path": pd.DataFrame({"path": [str(png_path)]}),
        "pdf_path": pd.DataFrame({"path": [str(pdf_path)]}) if save_pdf else pd.DataFrame({"path": []}),
    }
