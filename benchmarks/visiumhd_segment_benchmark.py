#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _read_notebook_data_path(nb_path: pathlib.Path) -> Optional[str]:
    try:
        nb = json.loads(nb_path.read_text())
    except Exception:
        return None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source") or "")
        m = re.search(r"ad\.read_zarr\(\s*['\"]([^'\"]+)['\"]\s*\)", src)
        if m:
            return m.group(1)
    return None


def _ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run(cmd: Sequence[str], *, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[{_now_ts()}] $ {' '.join(cmd)}", flush=True)
    subprocess.check_call(list(cmd), cwd=cwd, env=env)


def _default_spix_repo_for_env(env_name: str) -> Optional[str]:
    base = "/home/Data_Drive_8TB/chs1151"
    if env_name == "SPIX_1111":
        return os.path.join(base, "SPIX_1111", "SPIX")
    if env_name == "SPIX_1228":
        return os.path.join(base, "SPIX_1228", "SPIX")
    return None


def _conda_run(env_name: str, args: Sequence[str], *, cwd: Optional[str] = None) -> None:
    env = os.environ.copy()
    env["CONDA_NO_PLUGINS"] = "true"
    # Work around environments where /dev/shm semaphores are restricted and numba/scanpy
    # caching fails unless directed to a writable location.
    env.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba_cache"))
    env.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    env.setdefault("SPIX_CACHE_DIR", os.path.join(os.getcwd(), ".spix_cache_bench"))
    cmd = ["conda", "run", "-n", env_name, *args]
    _run(cmd, cwd=cwd, env=env)


def _maybe_git_rev(repo_dir: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _format_mb(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


class _RSSSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = float(interval_s)
        self._running = False
        self._peak = 0
        self._thread = None

    @staticmethod
    def _rss_bytes() -> int:
        try:
            import psutil  # type: ignore

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            # Linux fallback
            try:
                with open("/proc/self/statm", "r", encoding="utf-8") as f:
                    parts = f.read().strip().split()
                rss_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return int(rss_pages * page_size)
            except Exception:
                return 0

    def start(self) -> None:
        import threading

        self._running = True
        self._peak = self._rss_bytes()

        def _loop() -> None:
            while self._running:
                rss = self._rss_bytes()
                if rss > self._peak:
                    self._peak = rss
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return int(self._peak)


@dataclass
class StageResult:
    name: str
    seconds: float
    rss_start_mb: float
    rss_peak_mb: float
    rss_delta_peak_mb: float


def _measure_stage(name: str, fn) -> Tuple[Any, StageResult]:
    rss0 = _RSSSampler._rss_bytes()
    sampler = _RSSSampler(interval_s=0.05)
    t0 = time.perf_counter()
    sampler.start()
    try:
        out = fn()
    finally:
        peak = sampler.stop()
    dt = time.perf_counter() - t0
    rss_start_mb = _format_mb(rss0)
    rss_peak_mb = _format_mb(peak)
    res = StageResult(
        name=str(name),
        seconds=float(dt),
        rss_start_mb=float(rss_start_mb),
        rss_peak_mb=float(rss_peak_mb),
        rss_delta_peak_mb=float(max(0.0, rss_peak_mb - rss_start_mb)),
    )
    print(
        f"[{_now_ts()}] stage={res.name} time={res.seconds:.2f}s "
        f"rss_start={res.rss_start_mb:.1f}MB rss_peak={res.rss_peak_mb:.1f}MB "
        f"delta_peak={res.rss_delta_peak_mb:.1f}MB",
        flush=True,
    )
    return out, res


def _workflow_main(args: argparse.Namespace) -> None:
    # Resolve output path early (before chdir into the SPIX repo) so relative paths
    # always write into the caller's working directory.
    out_path = pathlib.Path(args.out_json).expanduser().resolve()

    # Ensure we import the desired SPIX repo (not the directory of this script).
    spix_repo = pathlib.Path(args.spix_repo).resolve()
    if not spix_repo.exists():
        raise SystemExit(f"--spix-repo not found: {spix_repo}")

    # Set thread env vars (notebook sets 64; benchmark can override).
    t = int(args.threads) if args.threads is not None else 64
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Remove this script's repo from sys.path, then force the target repo first.
    script_dir = pathlib.Path(__file__).resolve().parent
    script_repo = script_dir.parent
    sys.path = [p for p in sys.path if str(script_dir) not in p and str(script_repo) not in p]
    sys.path.insert(0, str(spix_repo))
    os.chdir(str(spix_repo))

    import anndata as ad  # type: ignore
    import SPIX  # type: ignore

    # Notebook parameters (cells 6,7,10,11,12,13)
    data_path = args.data_path
    if not data_path:
        raise SystemExit("--data-path is required in workflow mode")

    dims30 = list(range(30))
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else 32
    print(f"[{_now_ts()}] Using SPIX from: {getattr(SPIX, '__file__', 'unknown')}", flush=True)
    rev = _maybe_git_rev(str(spix_repo))
    if rev:
        print(f"[{_now_ts()}] SPIX git rev: {rev}", flush=True)
    print(f"[{_now_ts()}] Data: {data_path}", flush=True)
    print(f"[{_now_ts()}] threads={t} n_jobs={n_jobs}", flush=True)

    stage_results: List[StageResult] = []

    def _load():
        return ad.read_zarr(data_path)

    adata, res = _measure_stage("load", _load)
    stage_results.append(res)

    if args.max_obs is not None and int(args.max_obs) > 0:
        max_obs = int(args.max_obs)
        if getattr(adata, "n_obs", 0) > max_obs:
            print(f"[{_now_ts()}] Subsetting adata to n_obs={max_obs} for benchmarking", flush=True)
            adata = adata[:max_obs].copy()

    def _generate():
        return SPIX.tm.generate_embeddings(
            adata,
            dim_reduction="PCA",
            normalization="log_norm",
            n_jobs=n_jobs,
            dimensions=30,
            nfeatures=2000,
            use_coords_as_tiles=True,
            coords_max_gap_factor=None,
            force=True,
            use_hvg_only=True,
            raster_stride=10,
            filter_threshold=1,
            raster_max_pixels_per_tile=400,
            raster_random_seed=42,
        )

    adata, res = _measure_stage("generate_embeddings", _generate)
    stage_results.append(res)

    def _smooth():
        return SPIX.ip.smooth_image(
            adata,
            methods=["graph", "gaussian"],
            embedding="X_embedding",
            embedding_dims=dims30,
            graph_k=30,
            graph_t=20,
            gaussian_sigma=300,
            n_jobs=n_jobs,
            rescale_mode="final",
            backend="threads",
        )

    adata, res = _measure_stage("smooth", _smooth)
    stage_results.append(res)

    def _equalize():
        return SPIX.ip.equalize_image(
            adata,
            dimensions=dims30,
            embedding="X_embedding_smooth",
            sleft=5,
            sright=5,
        )

    adata, res = _measure_stage("equalize", _equalize)
    stage_results.append(res)

    def _cache():
        from SPIX.image_processing.image_cache import cache_embedding_image  # type: ignore

        return cache_embedding_image(
            adata,
            embedding="X_embedding_equalize",
            dimensions=dims30,
            key="image_plot_slic",
            origin=True,
            figsize=(30, 30),
            fig_dpi=100,
            verbose=False,
            show=bool(args.show_plots),
        )

    _cache_out, res = _measure_stage("cache", _cache)
    stage_results.append(res)

    def _segment():
        return SPIX.sp.segment_image(
            adata,
            dimensions=dims30,
            embedding="X_embedding_equalize",
            method="image_plot_slic",
            pitch_um=2,
            target_segment_um=500,
            compactness=0.5,
            verbose=True,
            figsize=(30, 30),
            use_cached_image=True,
            enforce_connectivity=False,
            origin=True,
            show_image=bool(args.show_plots),
        )

    _seg_out, res = _measure_stage("segment", _segment)
    stage_results.append(res)

    out = {
        "env": str(args.env_name),
        "spix_repo": str(spix_repo),
        "spix_file": getattr(SPIX, "__file__", None),
        "spix_git_rev": rev,
        "data_path": str(data_path),
        "max_obs": args.max_obs,
        "show_plots": bool(args.show_plots),
        "timestamp": _now_ts(),
        "stages": [vars(s) for s in stage_results],
        "total_seconds": float(sum(s.seconds for s in stage_results)),
    }
    _ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{_now_ts()}] Wrote results: {out_path}", flush=True)


def _plot_main(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    results = []
    for p in args.results_json:
        results.append(json.loads(pathlib.Path(p).read_text()))
    if len(results) < 2:
        raise SystemExit("--results-json needs >=2 files (SPIX_1111 and SPIX_1228)")

    # Use stage order from the first result.
    stages = [s["name"] for s in results[0]["stages"]]
    # Optional: drop "load" to focus on algorithm stages.
    if args.drop_load and stages and stages[0] == "load":
        stages = stages[1:]

    envs = [r.get("env", f"env{i}") for i, r in enumerate(results)]

    def _by_stage(res: Dict[str, Any], key: str) -> List[float]:
        rows = res["stages"]
        if args.drop_load and rows and rows[0]["name"] == "load":
            rows = rows[1:]
        return [float(r[key]) for r in rows]

    runtimes = [_by_stage(r, "seconds") for r in results]
    mem_deltas = [_by_stage(r, "rss_delta_peak_mb") for r in results]

    # Plot
    x = list(range(len(stages)))
    width = 0.35 if len(envs) == 2 else 0.8 / max(1, len(envs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=150)
    fig.suptitle(args.title or "SPIX benchmark (VisiumHD Segment pipeline)")

    for i, env_name in enumerate(envs):
        offs = [(xi + (i - (len(envs) - 1) / 2) * width) for xi in x]
        ax1.bar(offs, runtimes[i], width=width, label=env_name)
        ax2.bar(offs, mem_deltas[i], width=width, label=env_name)

    ax1.set_title("Runtime by Stage")
    ax1.set_ylabel("Seconds")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=20, ha="right")
    ax1.legend()

    ax2.set_title("Peak RSS Increase by Stage")
    ax2.set_ylabel("Δ peak RSS (MB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, rotation=20, ha="right")
    ax2.legend()

    fig.tight_layout()

    out_png = pathlib.Path(args.out_png).resolve()
    _ensure_dir(out_png.parent)
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[{_now_ts()}] Wrote plot: {out_png}", flush=True)


def _all_main(args: argparse.Namespace) -> None:
    nb_path = pathlib.Path(args.notebook).resolve()
    data_path = args.data_path or _read_notebook_data_path(nb_path)
    if not data_path:
        raise SystemExit("Could not infer --data-path from notebook; pass --data-path explicitly.")

    outdir = pathlib.Path(args.outdir).resolve()
    _ensure_dir(outdir)
    results_dir = outdir / "results"
    _ensure_dir(results_dir)

    script_path = pathlib.Path(__file__).resolve()

    envs = args.envs
    if len(envs) < 2:
        raise SystemExit("--envs needs >=2 env names")

    json_paths: List[str] = []
    for env_name in envs:
        spix_repo = args.spix_repo_map.get(env_name) or _default_spix_repo_for_env(env_name)
        if not spix_repo:
            raise SystemExit(f"Need --spix-repo-map for env: {env_name}")
        out_json = str(results_dir / f"{env_name}.json")
        json_paths.append(out_json)
        run_args = [
            "python",
            "-u",
            str(script_path),
            "--mode",
            "workflow",
            "--env-name",
            env_name,
            "--spix-repo",
            spix_repo,
            "--data-path",
            data_path,
            "--out-json",
            out_json,
            "--show-plots",
            "1" if args.show_plots else "0",
            "--threads",
            str(int(args.threads)),
            "--n-jobs",
            str(int(args.n_jobs)),
        ]
        if args.max_obs is not None:
            run_args.extend(["--max-obs", str(int(args.max_obs))])
        _conda_run(
            env_name,
            run_args,
            cwd=spix_repo,
        )

    # Plot using an env that has matplotlib (default: SPIX_1228).
    plot_env = args.plot_env or "SPIX_1228"
    out_png = str(outdir / "visiumhd_segment_benchmark.png")
    _conda_run(
        plot_env,
        [
            "python",
            "-u",
            str(script_path),
            "--mode",
            "plot",
            "--out-png",
            out_png,
            "--drop-load",
            "1" if args.drop_load else "0",
            "--title",
            args.title or "",
            "--results-json",
            *json_paths,
        ],
        cwd=str(outdir),
    )
    print(f"[{_now_ts()}] Done. Results in: {outdir}", flush=True)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark VisiumHD pipeline (generate_embeddings→smooth→equalize→cache→segment) in SPIX_1111 vs SPIX_1228."
    )
    p.add_argument(
        "--mode",
        choices=["all", "workflow", "plot"],
        default="all",
        help="all: run both envs + plot; workflow: run pipeline in current env; plot: plot existing JSONs.",
    )
    p.add_argument(
        "--notebook",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "VisiumHD_2um_CRC_multiscale_workflow.ipynb"),
        help="Notebook path used to infer defaults (data path).",
    )
    p.add_argument("--data-path", default=None, help="Override Zarr path (adata read_zarr).")
    p.add_argument("--outdir", default="./.spix_benchmarks/visiumhd_segment", help="Output directory.")
    p.add_argument(
        "--envs",
        default="SPIX_1111,SPIX_1228",
        help="Comma-separated conda env names to benchmark.",
    )
    p.add_argument(
        "--spix-repo-map",
        default="",
        help="Comma-separated env=repo mappings, e.g. SPIX_1111=/path/to/SPIX_1111/SPIX,SPIX_1228=/path/to/SPIX_1228/SPIX",
    )
    p.add_argument("--plot-env", default=None, help="Conda env to use for plotting (needs matplotlib).")
    p.add_argument("--title", default=None, help="Plot title.")
    p.add_argument("--drop-load", type=int, default=1, help="If 1, omit 'load' stage from plot.")
    p.add_argument("--max-obs", type=int, default=None, help="Optional: subset n_obs for a quick benchmark.")
    p.add_argument("--show-plots", type=int, default=0, help="If 1, render intermediate images (slower).")
    p.add_argument("--threads", type=int, default=64, help="OMP/BLAS/NumExpr threads (default=64; set 32 to match request).")
    p.add_argument("--n-jobs", type=int, default=32, help="SPIX n_jobs for stages (default=32).")

    # workflow-mode only
    p.add_argument("--env-name", default=None, help="Label for the current run (workflow mode).")
    p.add_argument("--spix-repo", default=None, help="SPIX repo dir to import from (workflow mode).")
    p.add_argument("--out-json", default=None, help="Output JSON path (workflow mode).")

    # plot-mode only
    p.add_argument("--results-json", nargs="*", default=[], help="One or more JSON results files (plot mode).")
    p.add_argument("--out-png", default=None, help="Output PNG path (plot mode).")

    ns = p.parse_args(argv)
    ns.envs = [e.strip() for e in str(ns.envs).split(",") if e.strip()]

    repo_map: Dict[str, str] = {}
    if ns.spix_repo_map:
        for item in str(ns.spix_repo_map).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"Invalid --spix-repo-map entry: {item}")
            k, v = item.split("=", 1)
            repo_map[k.strip()] = v.strip()
    ns.spix_repo_map = repo_map

    ns.drop_load = bool(int(ns.drop_load))
    ns.show_plots = bool(int(ns.show_plots))
    return ns


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if args.mode == "workflow":
        if not args.out_json or not args.spix_repo or not args.env_name:
            raise SystemExit("--mode workflow requires --env-name --spix-repo --out-json")
        _workflow_main(args)
        return
    if args.mode == "plot":
        if not args.out_png:
            raise SystemExit("--mode plot requires --out-png")
        if not args.results_json:
            raise SystemExit("--mode plot requires --results-json ...")
        _plot_main(args)
        return
    _all_main(args)


if __name__ == "__main__":
    main()
