from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_NON_MIXED_CATEGORIES = ("early", "mid", "late")


def _coerce_summary(summary_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = summary_df.copy()
    df.index = df.index.astype(str)
    cols = {}
    for col in [
        "category",
        "best_resolution",
        "specificity_score",
        "best_resolution_score",
        "best_resolution_margin",
    ]:
        if col in df.columns:
            cols[col] = f"{prefix}_{col}"
    out = df.rename(columns=cols)[list(cols.values())].copy()
    if f"{prefix}_category" in out.columns:
        out[f"{prefix}_category"] = out[f"{prefix}_category"].astype("string")
    if f"{prefix}_best_resolution" in out.columns:
        out[f"{prefix}_best_resolution"] = out[f"{prefix}_best_resolution"].astype("string")
    for col in [
        f"{prefix}_specificity_score",
        f"{prefix}_best_resolution_score",
        f"{prefix}_best_resolution_margin",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_gene_comparison_df(
    moran_summary_df: pd.DataFrame,
    geary_summary_df: pd.DataFrame,
    *,
    non_mixed_categories: Sequence[str] = DEFAULT_NON_MIXED_CATEGORIES,
) -> pd.DataFrame:
    non_mixed_categories = tuple(str(cat) for cat in non_mixed_categories)
    genes = sorted(set(moran_summary_df.index.astype(str)) | set(geary_summary_df.index.astype(str)))
    comp = pd.DataFrame(index=genes)
    comp.index.name = "gene"
    comp = comp.join(_coerce_summary(moran_summary_df, "moran"), how="left")
    comp = comp.join(_coerce_summary(geary_summary_df, "geary"), how="left")

    moran_cat = comp["moran_category"].astype("string")
    geary_cat = comp["geary_category"].astype("string")
    comp["moran_non_mixed"] = moran_cat.isin(non_mixed_categories)
    comp["geary_non_mixed"] = geary_cat.isin(non_mixed_categories)
    comp["same_category"] = moran_cat.fillna("missing") == geary_cat.fillna("missing")
    comp["same_best_resolution"] = (
        comp["moran_best_resolution"].astype("string").fillna("missing")
        == comp["geary_best_resolution"].astype("string").fillna("missing")
    )

    comp["relationship"] = np.select(
        [
            comp["moran_non_mixed"] & comp["geary_non_mixed"] & comp["same_category"],
            comp["moran_non_mixed"] & comp["geary_non_mixed"] & (~comp["same_category"]),
            comp["moran_non_mixed"] & (~comp["geary_non_mixed"]),
            (~comp["moran_non_mixed"]) & comp["geary_non_mixed"],
        ],
        [
            "shared_same_category",
            "shared_different_category",
            "moran_only_specific",
            "geary_only_specific",
        ],
        default="both_mixed",
    )

    comp["category_shift"] = (
        moran_cat.fillna("missing").astype(str)
        + " -> "
        + geary_cat.fillna("missing").astype(str)
    )
    comp["non_mixed_union"] = comp["moran_non_mixed"] | comp["geary_non_mixed"]
    comp["same_non_mixed_category"] = comp["relationship"] == "shared_same_category"

    moran_spec = pd.to_numeric(comp.get("moran_specificity_score"), errors="coerce").fillna(0.0)
    geary_spec = pd.to_numeric(comp.get("geary_specificity_score"), errors="coerce").fillna(0.0)
    comp["moran_specificity_score"] = moran_spec
    comp["geary_specificity_score"] = geary_spec
    comp["specificity_score_gap"] = geary_spec - moran_spec
    comp["abs_log_specificity_gap"] = np.abs(np.log1p(geary_spec) - np.log1p(moran_spec))
    return comp.reset_index()


def _resolution_value_from_label(label: str) -> float:
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


def _resolution_display(label: str, resolution_label_map: Optional[Mapping[int, str]] = None) -> str:
    value = _resolution_value_from_label(label)
    if np.isfinite(value):
        value_int = int(value)
        if resolution_label_map and value_int in resolution_label_map:
            return str(resolution_label_map[value_int])
        return str(value_int)
    return str(label)


def build_category_transition_tables(
    comparison_df: pd.DataFrame,
    *,
    non_mixed_categories: Sequence[str] = DEFAULT_NON_MIXED_CATEGORIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    non_mixed_categories = [str(cat) for cat in non_mixed_categories]
    comp = comparison_df.copy()
    both_specific = comp[comp["moran_non_mixed"] & comp["geary_non_mixed"]].copy()

    count_mat = pd.crosstab(
        both_specific["moran_category"],
        both_specific["geary_category"],
        dropna=False,
    ).reindex(index=non_mixed_categories, columns=non_mixed_categories, fill_value=0)
    row_frac = count_mat.div(count_mat.sum(axis=1).replace(0, np.nan), axis=0)

    rows = []
    for category in non_mixed_categories:
        moran_mask = comp["moran_category"].astype(str) == category
        geary_mask = comp["geary_category"].astype(str) == category
        shared = int((moran_mask & geary_mask).sum())
        moran_only = int((moran_mask & (~geary_mask)).sum())
        geary_only = int((geary_mask & (~moran_mask)).sum())
        rows.append(
            {
                "category": category,
                "shared": shared,
                "moran_only": moran_only,
                "geary_only": geary_only,
                "moran_total": int(moran_mask.sum()),
                "geary_total": int(geary_mask.sum()),
            }
        )
    same_category_df = pd.DataFrame(rows)
    return count_mat, row_frac, same_category_df


def build_best_resolution_tables(
    comparison_df: pd.DataFrame,
    *,
    same_category_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = comparison_df.copy()
    if same_category_only:
        comp = comp[comp["same_non_mixed_category"]].copy()
    else:
        comp = comp[comp["moran_non_mixed"] & comp["geary_non_mixed"]].copy()

    count_mat = pd.crosstab(
        comp["moran_best_resolution"],
        comp["geary_best_resolution"],
        dropna=False,
    )
    if count_mat.empty:
        return count_mat, count_mat.copy()

    row_order = sorted(count_mat.index.astype(str).tolist(), key=_resolution_value_from_label)
    col_order = sorted(count_mat.columns.astype(str).tolist(), key=_resolution_value_from_label)
    count_mat = count_mat.reindex(index=row_order, columns=col_order, fill_value=0)
    row_frac = count_mat.div(count_mat.sum(axis=1).replace(0, np.nan), axis=0)
    return count_mat, row_frac


def compute_rank_concordance_by_scale(
    moran_rank_df: pd.DataFrame,
    geary_rank_df: pd.DataFrame,
    *,
    top_k: int = 100,
    resolution_label_map: Optional[Mapping[int, str]] = None,
) -> pd.DataFrame:
    common_cols = sorted(set(moran_rank_df.columns) & set(geary_rank_df.columns), key=_resolution_value_from_label)
    rows = []
    for col in common_cols:
        pair = pd.concat(
            [
                pd.to_numeric(moran_rank_df[col], errors="coerce"),
                pd.to_numeric(geary_rank_df[col], errors="coerce"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if pair.empty:
            continue
        pair.columns = ["moran_rank", "geary_rank"]
        rho = pair["moran_rank"].rank(method="average").corr(pair["geary_rank"].rank(method="average"))
        top_moran = set(pair.nsmallest(int(top_k), "moran_rank").index.astype(str))
        top_geary = set(pair.nsmallest(int(top_k), "geary_rank").index.astype(str))
        intersection = len(top_moran & top_geary)
        union = len(top_moran | top_geary)
        rows.append(
            {
                "rank_col": str(col),
                "scale_id": str(col).replace("rank_", "", 1),
                "resolution": _resolution_value_from_label(col),
                "resolution_label": _resolution_display(col, resolution_label_map),
                "n_shared_genes": int(len(pair)),
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                f"top{int(top_k)}_shared": int(intersection),
                f"top{int(top_k)}_overlap_rate": float(intersection / float(top_k)),
                f"top{int(top_k)}_jaccard": float(intersection / union) if union else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)


def select_top_discordant_genes(
    comparison_df: pd.DataFrame,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    comp = comparison_df.copy()
    comp = comp[comp["non_mixed_union"]].copy()
    if comp.empty:
        return comp

    priority = {
        "shared_different_category": 0,
        "geary_only_specific": 1,
        "moran_only_specific": 2,
        "shared_same_category": 3,
    }
    comp["relationship_priority"] = comp["relationship"].map(priority).fillna(9).astype(int)
    comp["best_resolution_changed"] = (
        comp["moran_best_resolution"].astype("string").fillna("missing")
        != comp["geary_best_resolution"].astype("string").fillna("missing")
    )
    comp = comp.sort_values(
        [
            "relationship_priority",
            "best_resolution_changed",
            "abs_log_specificity_gap",
            "geary_specificity_score",
            "moran_specificity_score",
        ],
        ascending=[True, False, False, False, False],
    )
    return comp.head(int(top_n)).copy()


def build_summary_df(comparison_df: pd.DataFrame) -> pd.DataFrame:
    comp = comparison_df.copy()
    same_non_mixed = int(comp["same_non_mixed_category"].sum())
    same_best_resolution = int((comp["same_non_mixed_category"] & comp["same_best_resolution"]).sum())
    rows = [
        {"metric": "moran_non_mixed_genes", "value": int(comp["moran_non_mixed"].sum())},
        {"metric": "geary_non_mixed_genes", "value": int(comp["geary_non_mixed"].sum())},
        {"metric": "non_mixed_union_genes", "value": int(comp["non_mixed_union"].sum())},
        {"metric": "both_non_mixed_genes", "value": int((comp["moran_non_mixed"] & comp["geary_non_mixed"]).sum())},
        {"metric": "same_non_mixed_category_genes", "value": same_non_mixed},
        {
            "metric": "same_best_resolution_within_same_category_genes",
            "value": same_best_resolution,
        },
        {
            "metric": "same_best_resolution_rate_within_same_category",
            "value": float(same_best_resolution / same_non_mixed) if same_non_mixed else np.nan,
        },
    ]
    return pd.DataFrame(rows)


def _trajectory_score_df(traj_df: pd.DataFrame) -> pd.DataFrame:
    df = traj_df.copy()
    df.index = df.index.astype(str)
    score = 1.0 - df.divide(df.max(axis=0).replace(0, np.nan), axis=1).fillna(1.0)
    cols = sorted(score.columns.astype(str).tolist(), key=_resolution_value_from_label)
    return score[cols]


def _save_show(fig: plt.Figure, out_png: Path, *, show: bool = True) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_non_mixed_transition_heatmaps(
    count_mat: pd.DataFrame,
    row_frac: pd.DataFrame,
    out_png: Path,
    *,
    show: bool = True,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=300, gridspec_kw={"wspace": 0.35})
    sns.heatmap(
        count_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax1,
    )
    ax1.set_title("Non-mixed category overlap\n(count)", fontweight="bold", pad=10)
    ax1.set_xlabel("Geary C")
    ax1.set_ylabel("Moran I")

    sns.heatmap(
        row_frac * 100.0,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Row-normalized overlap (%)"},
        ax=ax2,
    )
    ax2.set_title("Non-mixed category overlap\n(row-normalized %)", fontweight="bold", pad=10)
    ax2.set_xlabel("Geary C")
    ax2.set_ylabel("")

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
    _save_show(fig, out_png, show=show)


def _plot_shared_unique_by_category(
    same_category_df: pd.DataFrame,
    out_png: Path,
    *,
    show: bool = True,
) -> None:
    plot_df = same_category_df.copy()
    plot_df = plot_df.set_index("category")[["shared", "moran_only", "geary_only"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=300)

    colors = {"shared": "#4C78A8", "moran_only": "#A6C8E0", "geary_only": "#F58518"}
    bottom = np.zeros(plot_df.shape[0], dtype=float)
    x = np.arange(plot_df.shape[0])
    for col in ["shared", "moran_only", "geary_only"]:
        vals = plot_df[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=col.replace("_", " "), color=colors[col], width=0.72)
        bottom += vals

    for idx, category in enumerate(plot_df.index.tolist()):
        total = int(plot_df.loc[category, ["shared", "moran_only", "geary_only"]].sum())
        ax.text(idx, total + max(2, total * 0.02), str(total), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index.tolist())
    ax.set_xlabel("Category")
    ax.set_ylabel("Gene count")
    ax.set_title("Category-wise shared vs unique genes\n(mixed excluded from categories)", fontweight="bold", pad=10)
    ax.legend(frameon=False, title="")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_show(fig, out_png, show=show)


def _plot_best_resolution_heatmaps(
    count_mat: pd.DataFrame,
    row_frac: pd.DataFrame,
    out_png: Path,
    *,
    resolution_label_map: Optional[Mapping[int, str]] = None,
    show: bool = True,
) -> None:
    if count_mat.empty:
        return

    plot_count = count_mat.copy()
    plot_frac = row_frac.copy()
    plot_count.index = [_resolution_display(idx, resolution_label_map) for idx in plot_count.index]
    plot_count.columns = [_resolution_display(col, resolution_label_map) for col in plot_count.columns]
    plot_frac.index = plot_count.index
    plot_frac.columns = plot_count.columns

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=300, gridspec_kw={"wspace": 0.35})
    sns.heatmap(
        plot_count,
        annot=True,
        fmt="d",
        cmap="mako",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax1,
    )
    ax1.set_title("Best-resolution concordance\n(count)", fontweight="bold", pad=10)
    ax1.set_xlabel("Geary C best resolution")
    ax1.set_ylabel("Moran I best resolution")

    sns.heatmap(
        plot_frac * 100.0,
        annot=True,
        fmt=".1f",
        cmap="mako",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Row-normalized overlap (%)"},
        ax=ax2,
    )
    ax2.set_title("Best-resolution concordance\n(row-normalized %)", fontweight="bold", pad=10)
    ax2.set_xlabel("Geary C best resolution")
    ax2.set_ylabel("")
    _save_show(fig, out_png, show=show)


def _plot_specificity_scatter(
    comparison_df: pd.DataFrame,
    top_discordant_df: pd.DataFrame,
    out_png: Path,
    *,
    show: bool = True,
) -> None:
    plot_df = comparison_df[comparison_df["non_mixed_union"]].copy()
    if plot_df.empty:
        return

    plot_df["moran_log_spec"] = np.log10(1.0 + plot_df["moran_specificity_score"].clip(lower=0))
    plot_df["geary_log_spec"] = np.log10(1.0 + plot_df["geary_specificity_score"].clip(lower=0))
    palette = {
        "shared_same_category": "#4C78A8",
        "shared_different_category": "#E45756",
        "moran_only_specific": "#72B7B2",
        "geary_only_specific": "#F58518",
    }

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=300)
    sns.scatterplot(
        data=plot_df,
        x="moran_log_spec",
        y="geary_log_spec",
        hue="relationship",
        hue_order=list(palette.keys()),
        palette=palette,
        s=28,
        alpha=0.75,
        edgecolor="none",
        ax=ax,
    )
    lim_max = max(float(plot_df["moran_log_spec"].max()), float(plot_df["geary_log_spec"].max()), 0.1) * 1.05
    ax.plot([0, lim_max], [0, lim_max], ls="--", lw=1.0, color="#808080")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("Moran I log10(1 + specificity score)")
    ax.set_ylabel("Geary C log10(1 + specificity score)")
    ax.set_title("Specificity-score agreement\n(non-mixed union genes)", fontweight="bold", pad=10)
    ax.legend(title="", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    for row in top_discordant_df.itertuples(index=False):
        ax.annotate(
            row.gene,
            (np.log10(1.0 + max(float(row.moran_specificity_score), 0.0)), np.log10(1.0 + max(float(row.geary_specificity_score), 0.0))),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    _save_show(fig, out_png, show=show)


def _plot_rank_concordance(
    rank_concordance_df: pd.DataFrame,
    out_png: Path,
    *,
    top_k: int,
    show: bool = True,
) -> None:
    if rank_concordance_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.8),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.18},
    )
    x = np.arange(rank_concordance_df.shape[0])

    ax1.bar(x, rank_concordance_df["spearman_rho"], color="#4C78A8", width=0.65)
    ax1.set_ylim(-0.05, 1.0)
    ax1.set_ylabel("Spearman rho")
    ax1.set_title("Scale-wise rank concordance", fontweight="bold", pad=10)
    ax1.grid(axis="y", alpha=0.15)

    overlap_col = f"top{int(top_k)}_overlap_rate"
    ax2.bar(x, rank_concordance_df[overlap_col] * 100.0, color="#F58518", width=0.65)
    ax2.set_ylabel(f"Top {int(top_k)} overlap (%)")
    ax2.set_xlabel("Resolution")
    ax2.grid(axis="y", alpha=0.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(rank_concordance_df["resolution_label"].tolist())
    _save_show(fig, out_png, show=show)


def _plot_discordant_trajectories(
    top_discordant_df: pd.DataFrame,
    moran_traj: pd.DataFrame,
    geary_traj: pd.DataFrame,
    out_png: Path,
    *,
    resolution_label_map: Optional[Mapping[int, str]] = None,
    show: bool = True,
) -> None:
    if top_discordant_df.empty:
        return

    moran_score = _trajectory_score_df(moran_traj)
    geary_score = _trajectory_score_df(geary_traj)
    common_cols = [col for col in moran_score.columns if col in geary_score.columns]
    common_cols = sorted(common_cols, key=_resolution_value_from_label)
    if not common_cols:
        return

    genes = [gene for gene in top_discordant_df["gene"].astype(str).tolist() if gene in moran_score.index or gene in geary_score.index]
    if not genes:
        return

    n_cols = 4
    n_rows = ceil(len(genes) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows), dpi=300, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    x = np.arange(len(common_cols))
    xticklabels = [_resolution_display(col, resolution_label_map) for col in common_cols]

    meta = top_discordant_df.set_index("gene")
    for ax, gene in zip(axes, genes):
        if gene in moran_score.index:
            ax.plot(x, moran_score.loc[gene, common_cols].to_numpy(dtype=float), "-o", lw=1.8, ms=4.5, color="#4C78A8", label="Moran I")
        if gene in geary_score.index:
            ax.plot(x, geary_score.loc[gene, common_cols].to_numpy(dtype=float), "-o", lw=1.8, ms=4.5, color="#F58518", label="Geary C")
        shift = meta.loc[gene, "category_shift"]
        ax.set_title(f"{gene}\n{shift}", fontsize=9.5, pad=6)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.15)

    for ax in axes[len(genes) :]:
        ax.set_axis_off()

    for ax in axes[: len(genes)]:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    for ax in axes[::n_cols]:
        ax.set_ylabel("Rank score")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Top discordant trajectories", fontweight="bold", y=1.06)
    fig.tight_layout()
    _save_show(fig, out_png, show=show)


def run_moran_geary_comparison(
    moran_summary_df: pd.DataFrame,
    geary_summary_df: pd.DataFrame,
    moran_traj: pd.DataFrame,
    geary_traj: pd.DataFrame,
    moran_rank_df: pd.DataFrame,
    geary_rank_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    non_mixed_categories: Sequence[str] = DEFAULT_NON_MIXED_CATEGORIES,
    resolution_label_map: Optional[Mapping[int, str]] = None,
    top_k: int = 100,
    top_discordant_n: int = 8,
    show: bool = True,
) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = build_gene_comparison_df(
        moran_summary_df,
        geary_summary_df,
        non_mixed_categories=non_mixed_categories,
    )
    count_mat, row_frac, same_category_df = build_category_transition_tables(
        comparison_df,
        non_mixed_categories=non_mixed_categories,
    )
    best_res_count, best_res_row_frac = build_best_resolution_tables(comparison_df, same_category_only=True)
    rank_concordance_df = compute_rank_concordance_by_scale(
        moran_rank_df,
        geary_rank_df,
        top_k=top_k,
        resolution_label_map=resolution_label_map,
    )
    top_discordant_df = select_top_discordant_genes(comparison_df, top_n=top_discordant_n)
    summary_df = build_summary_df(comparison_df)

    comparison_df.to_csv(out_dir / "moran_vs_geary_gene_level_comparison.csv", index=False)
    count_mat.to_csv(out_dir / "moran_vs_geary_non_mixed_transition_counts.csv")
    row_frac.to_csv(out_dir / "moran_vs_geary_non_mixed_transition_row_fraction.csv")
    same_category_df.to_csv(out_dir / "moran_vs_geary_non_mixed_shared_unique.csv", index=False)
    best_res_count.to_csv(out_dir / "moran_vs_geary_best_resolution_counts.csv")
    best_res_row_frac.to_csv(out_dir / "moran_vs_geary_best_resolution_row_fraction.csv")
    rank_concordance_df.to_csv(out_dir / "moran_vs_geary_rank_concordance_by_scale.csv", index=False)
    top_discordant_df.to_csv(out_dir / "moran_vs_geary_top_discordant_genes.csv", index=False)
    summary_df.to_csv(out_dir / "moran_vs_geary_summary_metrics.csv", index=False)

    _plot_non_mixed_transition_heatmaps(
        count_mat,
        row_frac,
        out_dir / "moran_vs_geary_non_mixed_category_overlap.png",
        show=show,
    )
    _plot_shared_unique_by_category(
        same_category_df,
        out_dir / "moran_vs_geary_non_mixed_shared_unique.png",
        show=show,
    )
    _plot_best_resolution_heatmaps(
        best_res_count,
        best_res_row_frac,
        out_dir / "moran_vs_geary_best_resolution_concordance.png",
        resolution_label_map=resolution_label_map,
        show=show,
    )
    _plot_specificity_scatter(
        comparison_df,
        top_discordant_df,
        out_dir / "moran_vs_geary_specificity_scatter.png",
        show=show,
    )
    _plot_rank_concordance(
        rank_concordance_df,
        out_dir / "moran_vs_geary_rank_concordance_by_scale.png",
        top_k=top_k,
        show=show,
    )
    _plot_discordant_trajectories(
        top_discordant_df,
        moran_traj,
        geary_traj,
        out_dir / "moran_vs_geary_top_discordant_trajectories.png",
        resolution_label_map=resolution_label_map,
        show=show,
    )

    return {
        "comparison_df": comparison_df,
        "summary_df": summary_df,
        "transition_count_df": count_mat,
        "transition_row_fraction_df": row_frac,
        "same_category_df": same_category_df,
        "best_resolution_count_df": best_res_count,
        "best_resolution_row_fraction_df": best_res_row_frac,
        "rank_concordance_df": rank_concordance_df,
        "top_discordant_df": top_discordant_df,
    }
