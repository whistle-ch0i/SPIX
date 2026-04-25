import numpy as np
import pandas as pd
import os
import time
import atexit
import glob
import hashlib
import shutil
import socket
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
from typing import Optional, Dict, Any, List, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from skimage.segmentation import find_boundaries
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, binary_dilation
try:
    import zarr
    from numcodecs import Blosc
except Exception:  # pragma: no cover - optional dependency
    zarr = None
    Blosc = None

# no additional utils needed here after function removals
from SPIX.visualization import raster as _raster


_MEMMAP_CLEANUP_PATHS: set[str] = set()
_MEMMAP_SCOPE_DIRS = {"persistent", "transient"}
_CACHE_STORAGE_KINDS = {"lossless_float32", "uint16", "float32_memmap", "float32_zarr"}
_CACHE_FILE_SUFFIXES = (".dat", ".npz", ".npy", ".zarr")
DEFAULT_CACHE_STORAGE = "float32_zarr"
_CACHE_SESSION_ID = os.environ.get(
    "SPIX_CACHE_SESSION_ID",
    f"{socket.gethostname()}-{os.getpid()}-{int(time.time() * 1000)}",
)


def _normalize_gap_collapse_axis_safe(axis):
    normalize_fn = getattr(_raster, "normalize_gap_collapse_axis", None)
    if callable(normalize_fn):
        return normalize_fn(axis)
    if axis is None:
        return None
    text = str(axis).strip().lower()
    if text in {"", "none", "off", "false", "no", "0"}:
        return None
    if text in {"y", "row", "rows", "vertical"}:
        return "y"
    if text in {"x", "col", "cols", "column", "columns", "horizontal"}:
        return "x"
    if text in {"both", "xy", "yx"}:
        return "both"
    return None


def _require_zarr() -> None:
    if zarr is None or Blosc is None:
        raise RuntimeError(
            "cache_storage='float32_zarr' requires optional dependencies 'zarr' and 'numcodecs'."
        )


def _cleanup_registered_memmaps() -> None:  # pragma: no cover
    # Best-effort cleanup; ignore failures (e.g., file already removed).
    for p in list(_MEMMAP_CLEANUP_PATHS):
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except Exception:
            pass


atexit.register(_cleanup_registered_memmaps)


def _safe_cache_component(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "default"
    safe = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in text)
    return safe or "default"


def _default_memmap_dir() -> str:
    # Prefer a stable user cache root over writing inside the current project.
    # Users can override with `memmap_dir=` or env var `SPIX_CACHE_DIR`.
    env = os.environ.get("SPIX_CACHE_DIR")
    if env:
        return env
    try:
        return str(Path.home() / ".cache" / "spix")
    except Exception:
        return os.path.join(os.getcwd(), ".spix_cache")


def _dataset_cache_namespace(adata) -> str:
    try:
        override = getattr(adata, "uns", {}).get("spix_cache_namespace", None)
        if override:
            return _safe_cache_component(override)
    except Exception:
        pass

    hasher = hashlib.sha1()
    try:
        hasher.update(f"n_obs={int(adata.n_obs)}|n_vars={int(adata.n_vars)}".encode("utf-8"))
    except Exception:
        hasher.update(b"n_obs=unknown|n_vars=unknown")

    try:
        obs_names = getattr(adata, "obs_names", None)
        if obs_names is not None:
            n_names = len(obs_names)
            edge_n = min(64, n_names)
            head = [str(v) for v in obs_names[:edge_n]]
            tail = [str(v) for v in obs_names[-edge_n:]] if n_names > edge_n else []
            for token in head + ["__TAIL__"] + tail:
                hasher.update(token.encode("utf-8", errors="ignore"))
                hasher.update(b"\0")
    except Exception:
        pass

    try:
        spatial = np.asarray(adata.obsm.get("spatial"))
        if spatial.ndim == 2 and spatial.shape[0] > 0 and spatial.shape[1] >= 2:
            arr = np.asarray(spatial[:, :2], dtype=np.float64)
            edge_n = min(8, arr.shape[0])
            sample = arr[:edge_n] if arr.shape[0] <= edge_n else np.vstack([arr[:edge_n], arr[-edge_n:]])
            hasher.update(f"spatial_shape={arr.shape}".encode("utf-8"))
            hasher.update(np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0).tobytes())
            mins = np.nanmin(arr, axis=0)
            maxs = np.nanmax(arr, axis=0)
            hasher.update(np.nan_to_num(mins, nan=0.0, posinf=0.0, neginf=0.0).tobytes())
            hasher.update(np.nan_to_num(maxs, nan=0.0, posinf=0.0, neginf=0.0).tobytes())
    except Exception:
        pass

    try:
        filename = getattr(adata, "filename", None)
        if filename:
            hasher.update(str(filename).encode("utf-8", errors="ignore"))
    except Exception:
        pass

    return f"adata_{hasher.hexdigest()[:16]}"


def _resolve_memmap_cache_dir(
    adata,
    *,
    memmap_dir: Optional[str],
    cache_namespace: Optional[str],
    memmap_scope: str,
) -> tuple[str, Optional[str], str, Optional[str]]:
    scope = str(memmap_scope or "persistent").lower()
    if scope not in _MEMMAP_SCOPE_DIRS:
        raise ValueError("memmap_scope must be one of {'persistent','transient'}.")
    if memmap_dir is not None:
        try:
            resolved = str(Path(memmap_dir).expanduser().resolve(strict=False))
        except Exception:
            resolved = str(memmap_dir)
        return resolved, None, scope, None

    root = Path(_default_memmap_dir()).expanduser()
    namespace = _safe_cache_component(cache_namespace or _dataset_cache_namespace(adata))
    session_id = _safe_cache_component(_CACHE_SESSION_ID) if scope == "transient" else None
    if scope == "transient":
        cache_dir = root / "transient" / session_id / namespace
    else:
        cache_dir = root / "persistent" / namespace
    return str(cache_dir.resolve(strict=False)), namespace, scope, session_id


def _prune_empty_cache_dirs(path: Path, *, stop_at: Optional[Path] = None) -> None:
    current = Path(path)
    stop = None if stop_at is None else Path(stop_at)
    while True:
        if stop is not None:
            try:
                if current.resolve(strict=False) == stop.resolve(strict=False):
                    break
            except Exception:
                if str(current) == str(stop):
                    break
        try:
            current.rmdir()
        except Exception:
            break
        if current.parent == current:
            break
        current = current.parent


def _normalize_cache_storage(cache_storage: Optional[str]) -> str:
    storage = str(cache_storage or DEFAULT_CACHE_STORAGE).lower()
    if storage not in _CACHE_STORAGE_KINDS:
        raise ValueError("cache_storage must be one of {'lossless_float32','uint16','float32_memmap','float32_zarr'}.")
    return storage


def _cache_disk_suffix(cache_storage: str) -> str:
    storage = _normalize_cache_storage(cache_storage)
    if storage == "lossless_float32":
        return ".npz"
    if storage == "uint16":
        return ".npy"
    if storage == "float32_zarr":
        return ".zarr"
    return ".dat"


def _cache_disk_path(cache_dir: str, key: str, embedding: str, shape: tuple[int, int, int], cache_storage: str) -> str:
    safe_key = _safe_cache_component(key)
    safe_emb = _safe_cache_component(embedding)
    h, w, dims = (int(shape[0]), int(shape[1]), int(shape[2]))
    suffix = _cache_disk_suffix(cache_storage)
    return os.path.join(cache_dir, f"{safe_key}__{safe_emb}__{h}x{w}x{dims}{suffix}")


def _temporary_cache_work_path(final_path: str) -> str:
    if final_path.endswith(".dat"):
        return final_path
    return f"{final_path}.tmp.dat"


def _zarr_chunk_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    _require_zarr()
    h, w, c = (int(shape[0]), int(shape[1]), int(shape[2]))
    chunk_h = int(max(1, int(os.environ.get("SPIX_ZARR_CHUNK_H", "512"))))
    chunk_w = int(max(1, int(os.environ.get("SPIX_ZARR_CHUNK_W", "512"))))
    chunk_c_default = min(max(1, c), 4 if c > 4 else c)
    chunk_c = int(max(1, int(os.environ.get("SPIX_ZARR_CHUNK_C", str(chunk_c_default)))))
    return (min(h, chunk_h), min(w, chunk_w), min(c, chunk_c))


def _zarr_compressor():
    _require_zarr()
    cname = str(os.environ.get("SPIX_ZARR_COMPRESSOR", "zstd"))
    clevel = int(max(1, min(9, int(os.environ.get("SPIX_ZARR_CLEVEL", "5")))))
    shuffle_mode = str(os.environ.get("SPIX_ZARR_SHUFFLE", "bitshuffle")).strip().lower()
    if shuffle_mode == "shuffle":
        shuffle = Blosc.SHUFFLE
    elif shuffle_mode == "noshuffle":
        shuffle = Blosc.NOSHUFFLE
    else:
        shuffle = Blosc.BITSHUFFLE
    return Blosc(cname=cname, clevel=clevel, shuffle=shuffle)


def _load_cache_image_from_disk(cache: Dict[str, Any], *, channels: Optional[Sequence[int]] = None) -> np.ndarray:
    img_path = cache.get("img_path", None)
    if not img_path:
        raise ValueError("Cached image does not have an on-disk image path.")
    storage = _normalize_cache_storage(cache.get("cache_storage", "float32_memmap"))
    h = int(cache.get("h"))
    w = int(cache.get("w"))
    c = int(cache.get("channels", cache.get("img_channels", 0)))
    if storage == "float32_memmap":
        arr = np.memmap(img_path, dtype=np.float32, mode="r", shape=(h, w, c))
        if channels is None:
            return arr
        return np.asarray(arr[:, :, list(channels)], dtype=np.float32)
    if storage == "lossless_float32":
        with np.load(img_path) as data:
            arr = np.asarray(data["img"], dtype=np.float32)
            if channels is None:
                return arr
            return np.asarray(arr[:, :, list(channels)], dtype=np.float32)
    if storage == "uint16":
        arr = np.load(img_path, mmap_mode="r")
        if channels is not None:
            arr = arr[:, :, list(channels)]
        return np.asarray(arr, dtype=np.float32) / 65535.0
    if storage == "float32_zarr":
        _require_zarr()
        arr = zarr.open(img_path, mode="r")
        if channels is None:
            return np.asarray(arr, dtype=np.float32)
        return np.asarray(arr[:, :, list(channels)], dtype=np.float32)
    raise ValueError(f"Unsupported cache storage: {storage}")


def get_cached_image_shape(cache: Dict[str, Any]) -> tuple[int, int, int]:
    """Return cached image shape without loading the image payload when possible."""
    img = cache.get("img", None)
    if isinstance(img, np.ndarray) and img.ndim == 3:
        return tuple(int(x) for x in img.shape)

    h = cache.get("h", None)
    w = cache.get("w", None)
    c = cache.get("channels", cache.get("img_channels", None))
    if h is None or w is None or c is None:
        raise ValueError("Cached image does not contain shape metadata.")
    return (int(h), int(w), int(c))


def get_cached_image(cache: Dict[str, Any], *, cache_in_memory: bool = False) -> np.ndarray:
    """Return the cached image array, loading it from disk if necessary."""
    return get_cached_image_channels(cache, channels=None, cache_in_memory=cache_in_memory)


def get_cached_image_channels(
    cache: Dict[str, Any],
    *,
    channels: Optional[Sequence[int]] = None,
    cache_in_memory: bool = False,
) -> np.ndarray:
    """Return cached image channels, using partial reads when the storage supports it."""
    img = cache.get("img", None)
    if isinstance(img, np.ndarray) and img.ndim == 3:
        if channels is None:
            return img
        return np.asarray(img[:, :, list(channels)], dtype=np.float32)
    loaded = _load_cache_image_from_disk(cache, channels=channels)
    if cache_in_memory and channels is None:
        cache["img"] = loaded
    return loaded


def _write_cache_image_to_disk(img: np.ndarray, img_path: str, cache_storage: str) -> None:
    storage = _normalize_cache_storage(cache_storage)
    img_f32 = np.asarray(img, dtype=np.float32)
    tmp_path = f"{img_path}.tmpwrite"
    if storage == "float32_memmap":
        if tmp_path != img_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        mm = np.memmap(img_path, dtype=np.float32, mode="w+", shape=img_f32.shape)
        mm[...] = img_f32
        mm.flush()
        return
    if storage == "lossless_float32":
        np.savez_compressed(tmp_path, img=img_f32)
        npz_path = tmp_path if tmp_path.endswith(".npz") else f"{tmp_path}.npz"
        os.replace(npz_path, img_path)
        return
    if storage == "uint16":
        arr = np.rint(np.clip(img_f32, 0.0, 1.0) * 65535.0).astype(np.uint16)
        np.save(tmp_path, arr)
        npy_path = tmp_path if tmp_path.endswith(".npy") else f"{tmp_path}.npy"
        os.replace(npy_path, img_path)
        return
    if storage == "float32_zarr":
        _require_zarr()
        tmp_dir = f"{img_path}.tmpwrite"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.isdir(img_path):
            shutil.rmtree(img_path, ignore_errors=True)
        arr = zarr.open(
            tmp_dir,
            mode="w",
            shape=img_f32.shape,
            chunks=_zarr_chunk_shape(tuple(int(x) for x in img_f32.shape)),
            dtype="f4",
            compressor=_zarr_compressor(),
        )
        arr[...] = img_f32
        shutil.move(tmp_dir, img_path)
        return
    raise ValueError(f"Unsupported cache storage: {storage}")


def _iter_memmap_paths(
    *,
    memmap_dir: Optional[str],
    key: Optional[str],
    embedding: Optional[str],
    cache_namespace: Optional[str],
    scope: str,
    session_id: Optional[str],
) -> tuple[list[Path], Optional[Path]]:
    scope_norm = str(scope or "all").lower()
    if scope_norm not in {"all", "persistent", "transient"}:
        raise ValueError("scope must be one of {'all','persistent','transient'}.")

    root_scope_map: list[tuple[Path, str]] = []
    if memmap_dir is not None:
        root = Path(memmap_dir).expanduser()
        stop_root = root
        has_scoped_children = any((root / scope_name).exists() for scope_name in _MEMMAP_SCOPE_DIRS)
        namespace_safe = _safe_cache_component(cache_namespace) if cache_namespace is not None else None
        if has_scoped_children:
            if scope_norm in {"all", "persistent"}:
                scoped_root = root / "persistent"
                if namespace_safe is not None:
                    scoped_root = scoped_root / namespace_safe
                root_scope_map.append((scoped_root, "persistent"))
            if scope_norm in {"all", "transient"}:
                scoped_root = root / "transient"
                if session_id is not None:
                    scoped_root = scoped_root / _safe_cache_component(session_id)
                    if namespace_safe is not None:
                        scoped_root = scoped_root / namespace_safe
                root_scope_map.append((scoped_root, "transient"))
            roots = [path for path, _ in root_scope_map]
        else:
            roots = [root]
            root_scope_map = [(root, scope_norm)]
    else:
        stop_root = Path(_default_memmap_dir()).expanduser()
        namespace_safe = _safe_cache_component(cache_namespace) if cache_namespace is not None else None
        if scope_norm in {"all", "persistent"}:
            root = stop_root / "persistent"
            if namespace_safe is not None:
                root = root / namespace_safe
            root_scope_map.append((root, "persistent"))
        if scope_norm in {"all", "transient"}:
            root = stop_root / "transient"
            if session_id is not None:
                root = root / _safe_cache_component(session_id)
                if namespace_safe is not None:
                    root = root / namespace_safe
            elif namespace_safe is not None:
                # Namespace is nested one level deeper under session ids.
                root = root
            root_scope_map.append((root, "transient"))
        roots = [root for root, _ in root_scope_map]

    namespace_safe = _safe_cache_component(cache_namespace) if cache_namespace is not None else None
    if not root_scope_map:
        root_scope_map = [(root, scope_norm) for root in roots]

    safe_key = None if key is None else _safe_cache_component(key)
    safe_emb = None if embedding is None else _safe_cache_component(embedding)
    matched: list[Path] = []
    seen: set[str] = set()
    for root, root_scope in root_scope_map:
        if not root.exists():
            continue
        for suffix in _CACHE_FILE_SUFFIXES:
            paths_iter = root.rglob(f"*{suffix}")
            for path in paths_iter:
                name = path.name
                if safe_key is not None and not name.startswith(f"{safe_key}__"):
                    continue
                if safe_emb is not None and f"__{safe_emb}__" not in name:
                    continue
                if (
                    namespace_safe is not None
                    and root_scope == "transient"
                    and session_id is None
                    and namespace_safe not in path.parts
                ):
                    continue
                resolved = str(path.resolve(strict=False))
                if resolved in seen:
                    continue
                seen.add(resolved)
                matched.append(path)
    matched.sort()
    return matched, stop_root


def _release_cache_disk_image(
    cache: Optional[Dict[str, Any]],
    *,
    remove_file: bool = True,
    verbose: bool = False,
    strict: bool = False,
) -> bool:
    if not isinstance(cache, dict):
        return False
    img = cache.get("img", None)
    if isinstance(img, np.memmap):
        try:
            mmap_obj = getattr(img, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        except Exception:
            if strict:
                raise
    img_path = cache.get("img_path", None)
    if not img_path:
        return False
    _MEMMAP_CLEANUP_PATHS.discard(str(img_path))
    if not remove_file:
        return True
    try:
        if os.path.isdir(img_path):
            shutil.rmtree(img_path, ignore_errors=True)
        else:
            os.remove(img_path)
        cache_root = cache.get("cache_root", None)
        if cache_root is not None:
            _prune_empty_cache_dirs(Path(img_path).parent, stop_at=Path(cache_root))
        if verbose:
            print(f"[release_cached_memmap_file] removed={img_path}")
        return True
    except FileNotFoundError:
        return False
    except Exception:
        if strict:
            raise
        return False


def _downsample_support_mask(mask: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape == (int(new_h), int(new_w)):
        return mask_arr
    resized = resize(
        mask_arr.astype(np.float32),
        (int(new_h), int(new_w)),
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return (resized >= 0.5).astype(bool, copy=False)


def _auto_preview_figsize_from_image_shape(shape: Tuple[int, ...]) -> tuple[float, float]:
    """Choose a readable preview figure size from an image shape.

    Uses only display geometry; does not affect rasterization.
    Keeps the long side around 10 inches and the short side at least 5 inches.
    """
    try:
        h = int(shape[0])
        w = int(shape[1])
    except Exception:
        return (10.0, 8.0)
    if h <= 0 or w <= 0:
        return (10.0, 8.0)
    aspect = float(w) / float(h)
    long_side = 10.0
    short_floor = 5.0
    if aspect >= 1.0:
        fig_w = long_side
        fig_h = max(short_floor, long_side / max(aspect, 1e-9))
    else:
        fig_h = long_side
        fig_w = max(short_floor, long_side * aspect)
    return (float(fig_w), float(fig_h))


def _resolve_cached_preview_figsize_and_dpi(
    cache: dict,
    img_shape: Tuple[int, ...],
    *,
    figsize: Optional[tuple] = None,
    fig_dpi: Optional[int] = None,
) -> tuple[Optional[tuple], Optional[int]]:
    """Resolve cached preview size, preferring original raster size when unspecified."""
    resolved_figsize = figsize if figsize is not None else cache.get("figsize", None)
    resolved_dpi = fig_dpi if fig_dpi is not None else cache.get("fig_dpi", None)
    if resolved_dpi is None:
        resolved_dpi = cache.get("raster_dpi", cache.get("effective_dpi", 100))
    if resolved_figsize is None:
        try:
            h = int(img_shape[0])
            w = int(img_shape[1])
            dpi_use = float(resolved_dpi) if resolved_dpi is not None else 100.0
            if h > 0 and w > 0 and dpi_use > 0:
                resolved_figsize = (float(w) / dpi_use, float(h) / dpi_use)
            else:
                resolved_figsize = _auto_preview_figsize_from_image_shape(img_shape)
        except Exception:
            resolved_figsize = _auto_preview_figsize_from_image_shape(img_shape)
    return resolved_figsize, resolved_dpi


def clear_memmap_cache(
    memmap_dir: Optional[str] = None,
    *,
    key: Optional[str] = None,
    embedding: Optional[str] = None,
    cache_namespace: Optional[str] = None,
    scope: str = "all",
    session_id: Optional[str] = None,
    remove_empty_dirs: bool = True,
    verbose: bool = False,
    strict: bool = False,
) -> int:
    """Delete on-disk memmap files created by `cache_embedding_image`.

    Parameters
    ----------
    memmap_dir
        Cache root to clear. Defaults to the same root used by cache generation.
    key, embedding
        If provided, only delete files matching this key/embedding prefix.
    cache_namespace
        Optional persistent/transient namespace to target under the shared cache root.
    scope
        Which cache tree to clear: ``all``, ``persistent``, or ``transient``.
    session_id
        Optional transient session id filter.

    Returns
    -------
    int
        Number of deleted files.
    """
    paths, stop_root = _iter_memmap_paths(
        memmap_dir=memmap_dir,
        key=key,
        embedding=embedding,
        cache_namespace=cache_namespace,
        scope=scope,
        session_id=session_id,
    )
    if verbose:
        roots_msg = memmap_dir or _default_memmap_dir()
        print(f"[clear_memmap_cache] cache_root={roots_msg!r} matched_files={len(paths)}")
    deleted = 0
    for p in paths:
        try:
            if Path(p).is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            deleted += 1
            _MEMMAP_CLEANUP_PATHS.discard(str(p))
            if remove_empty_dirs:
                _prune_empty_cache_dirs(Path(p).parent, stop_at=stop_root)
        except Exception:
            if strict:
                raise
    return deleted


def summarize_memmap_cache(
    memmap_dir: Optional[str] = None,
    *,
    key: Optional[str] = None,
    embedding: Optional[str] = None,
    cache_namespace: Optional[str] = None,
    scope: str = "all",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize on-disk memmap files created by ``cache_embedding_image``."""
    paths, stop_root = _iter_memmap_paths(
        memmap_dir=memmap_dir,
        key=key,
        embedding=embedding,
        cache_namespace=cache_namespace,
        scope=scope,
        session_id=session_id,
    )
    files: list[Dict[str, Any]] = []
    total_bytes = 0
    for path in paths:
        try:
            if path.is_dir():
                size = 0
                for child in path.rglob("*"):
                    if child.is_file():
                        try:
                            size += int(child.stat().st_size)
                        except Exception:
                            continue
            else:
                size = int(path.stat().st_size)
        except Exception:
            size = 0
        total_bytes += size
        files.append(
            {
                "path": str(path.resolve(strict=False)),
                "size_bytes": int(size),
                "size_mb": float(size) / (1024.0 * 1024.0),
            }
        )
    return {
        "cache_root": str((Path(memmap_dir).expanduser() if memmap_dir is not None else stop_root).resolve(strict=False)),
        "scope": str(scope),
        "cache_namespace": None if cache_namespace is None else _safe_cache_component(cache_namespace),
        "session_id": session_id,
        "count": int(len(files)),
        "total_bytes": int(total_bytes),
        "total_mb": float(total_bytes) / (1024.0 * 1024.0),
        "files": files,
    }


def release_cached_image(
    adata,
    key: str,
    *,
    remove_memmap_file: bool = False,
    verbose: bool = False,
    strict: bool = False,
) -> bool:
    """Remove a cached image entry from ``adata.uns`` and optionally delete its memmap file."""
    if not hasattr(adata, "uns") or key not in adata.uns:
        return False
    cache = adata.uns.get(key)
    if remove_memmap_file:
        _release_cache_disk_image(cache, remove_file=True, verbose=verbose, strict=strict)
    adata.uns.pop(key, None)
    return True


def _geometry_cache_bucket_summary(adata) -> Dict[str, int]:
    if not hasattr(adata, "uns"):
        return {"canvas_entries": 0, "fill_entries": 0}
    canvas_key = getattr(_raster, "_PERSISTENT_RASTER_CANVAS_BUCKET_KEY", "_spix_raster_canvas_cache_v1")
    fill_key = getattr(_raster, "_PERSISTENT_FILL_CACHE_BUCKET_KEY", "_spix_raster_fill_cache_v1")
    canvas_bucket = adata.uns.get(canvas_key, None)
    fill_bucket = adata.uns.get(fill_key, None)
    return {
        "canvas_entries": int(len(canvas_bucket)) if isinstance(canvas_bucket, dict) else 0,
        "fill_entries": int(len(fill_bucket)) if isinstance(fill_bucket, dict) else 0,
    }


def _clear_geometry_cache_buckets(adata) -> Dict[str, int]:
    removed = {"canvas_entries": 0, "fill_entries": 0}
    if not hasattr(adata, "uns"):
        return removed
    canvas_key = getattr(_raster, "_PERSISTENT_RASTER_CANVAS_BUCKET_KEY", "_spix_raster_canvas_cache_v1")
    fill_key = getattr(_raster, "_PERSISTENT_FILL_CACHE_BUCKET_KEY", "_spix_raster_fill_cache_v1")
    canvas_bucket = adata.uns.pop(canvas_key, None)
    fill_bucket = adata.uns.pop(fill_key, None)
    if isinstance(canvas_bucket, dict):
        removed["canvas_entries"] = int(len(canvas_bucket))
    if isinstance(fill_bucket, dict):
        removed["fill_entries"] = int(len(fill_bucket))
    try:
        raster_canvas_cache = getattr(_raster, "_RASTER_CANVAS_CACHE", None)
        if raster_canvas_cache is not None:
            raster_canvas_cache.clear()
    except Exception:
        pass
    try:
        fill_assignment_cache = getattr(_raster, "_FILL_ASSIGNMENT_CACHE", None)
        if fill_assignment_cache is not None:
            fill_assignment_cache.clear()
    except Exception:
        pass
    return removed


def summarize_current_cache(
    adata,
    *,
    memmap_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize persistent/transient memmap cache for the current dataset."""
    namespace = _dataset_cache_namespace(adata)
    persistent = summarize_memmap_cache(
        memmap_dir=memmap_dir,
        cache_namespace=namespace,
        scope="persistent",
    )
    transient = summarize_memmap_cache(
        memmap_dir=memmap_dir,
        cache_namespace=namespace,
        scope="transient",
    )
    transient_current_session = summarize_memmap_cache(
        memmap_dir=memmap_dir,
        cache_namespace=namespace,
        scope="transient",
        session_id=_CACHE_SESSION_ID,
    )
    return {
        "cache_namespace": namespace,
        "session_id": _CACHE_SESSION_ID,
        "geometry_cache": _geometry_cache_bucket_summary(adata),
        "persistent": persistent,
        "transient": transient,
        "transient_current_session": transient_current_session,
        "total_bytes": int(persistent["total_bytes"]) + int(transient["total_bytes"]),
        "total_mb": float(persistent["total_mb"]) + float(transient["total_mb"]),
        "count": int(persistent["count"]) + int(transient["count"]),
    }


def clear_current_cache(
    adata,
    *,
    memmap_dir: Optional[str] = None,
    clear_persistent: bool = True,
    clear_transient: bool = True,
    clear_in_memory: bool = True,
    verbose: bool = False,
    strict: bool = False,
) -> Dict[str, int]:
    """Clear cache for the current dataset namespace.

    This is the high-level cleanup entrypoint for most users.
    """
    namespace = _dataset_cache_namespace(adata)
    removed_in_memory = 0
    removed_geometry = {"canvas_entries": 0, "fill_entries": 0}
    if clear_in_memory and hasattr(adata, "uns"):
        for key in list(list_cached_images(adata)):
            cache = adata.uns.get(key, {})
            if isinstance(cache, dict) and cache.get("cache_namespace") == namespace:
                if release_cached_image(adata, key, remove_memmap_file=True, verbose=verbose, strict=strict):
                    removed_in_memory += 1
        removed_geometry = _clear_geometry_cache_buckets(adata)

    removed_persistent = 0
    removed_transient = 0
    if clear_persistent:
        removed_persistent = clear_memmap_cache(
            memmap_dir=memmap_dir,
            cache_namespace=namespace,
            scope="persistent",
            verbose=verbose,
            strict=strict,
        )
    if clear_transient:
        removed_transient = clear_memmap_cache(
            memmap_dir=memmap_dir,
            cache_namespace=namespace,
            scope="transient",
            verbose=verbose,
            strict=strict,
        )
    return {
        "cache_namespace": namespace,
        "removed_in_memory": int(removed_in_memory),
        "removed_geometry_canvas_entries": int(removed_geometry["canvas_entries"]),
        "removed_geometry_fill_entries": int(removed_geometry["fill_entries"]),
        "removed_persistent_files": int(removed_persistent),
        "removed_transient_files": int(removed_transient),
        "removed_files": int(removed_persistent + removed_transient),
    }


def clear_temporary_cache(
    adata=None,
    *,
    memmap_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    include_other_sessions: bool = False,
    verbose: bool = False,
    strict: bool = False,
) -> int:
    """Clear transient cache files.

    By default removes only the current process/session's transient cache files.
    Pass ``adata`` to limit cleanup to the current dataset namespace as well.
    Set ``include_other_sessions=True`` to remove transient cache files from
    other notebooks/processes too.
    """
    namespace = _dataset_cache_namespace(adata) if adata is not None else None
    effective_session_id = None if include_other_sessions else _safe_cache_component(session_id or _CACHE_SESSION_ID)
    return clear_memmap_cache(
        memmap_dir=memmap_dir,
        cache_namespace=namespace,
        scope="transient",
        session_id=effective_session_id,
        verbose=verbose,
        strict=strict,
    )


def clear_all_cache(
    *,
    memmap_dir: Optional[str] = None,
    verbose: bool = False,
    strict: bool = False,
) -> int:
    """Clear every memmap cache file under the shared cache root."""
    return clear_memmap_cache(
        memmap_dir=memmap_dir,
        scope="all",
        verbose=verbose,
        strict=strict,
    )


class _QuietCacheDict(dict):
    """dict subclass with a compact repr to avoid dumping huge arrays/lists in notebooks."""

    @staticmethod
    def _summarize(v: Any) -> str:
        try:
            if isinstance(v, np.ndarray):
                return f"<ndarray shape={v.shape} dtype={v.dtype}>"
            if isinstance(v, list):
                n = len(v)
                if n == 0:
                    return "<list len=0>"
                # Barcodes can be millions; avoid showing first elements.
                return f"<list len={n}>"
            if isinstance(v, dict):
                return f"<dict keys={len(v)}>"
        except Exception:
            pass
        return repr(v)

    def __repr__(self) -> str:  # pragma: no cover
        items = ", ".join(f"{k!r}: {self._summarize(v)}" for k, v in self.items())
        return "{" + items + "}"

    __str__ = __repr__


def cache_embedding_image(
    adata,
    *,
    embedding: str = "X_embedding",
    dimensions: List[int] = [0, 1, 2],
    origin: bool = True,
    key: str = "image_plot_slic",
    coordinate_mode: str = "spatial",  # 'spatial'|'array'|'visium'|'visiumhd'|'auto'
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    figsize: tuple | None = None,
    fig_dpi: Optional[int] = None,
    imshow_tile_size: Optional[float] = None,
    imshow_scale_factor: float = 1.0,
    imshow_tile_size_mode: str = "auto",
    imshow_tile_size_quantile: Optional[float] = None,
    imshow_tile_size_rounding: str = "floor",
    imshow_tile_size_shrink: float = 0.98,
    pixel_perfect: bool = False,
    pixel_shape: str = "square",
    chunk_size: int = 50000,
    downsample_factor: float = 1.0,
    # Storage control for large images
    implementation: str = "auto",  # 'auto'|'optimized'|'legacy'
    store: str = "auto",  # 'auto'|'memory'|'memmap'
    memmap_dir: Optional[str] = None,
    cache_namespace: Optional[str] = None,
    memmap_scope: str = "persistent",  # 'persistent'|'transient'
    cache_storage: str = DEFAULT_CACHE_STORAGE,  # 'lossless_float32'|'uint16'|'float32_memmap'|'float32_zarr'
    memmap_threshold_mb: float = 512.0,
    memmap_prune: str = "none",  # 'none'|'key' (remove older shapes for same key+embedding)
    memmap_cleanup: str = "none",  # 'none'|'on_exit' (delete created file when Python exits)
    # Cache metadata size control
    store_barcodes: str = "auto",  # 'auto'|'always'|'never'
    barcodes_max_n: int = 1_000_000,
    verbose: bool = False,
    show: bool = False,
    show_channels: Optional[Sequence[int]] = None,
    show_cmap: Optional[str] = None,
    # Selected image_plot compatibility options (accepted + stored; some applied):
    color_by: Optional[str] = None,
    palette: Optional[str | Sequence] = None,
    alpha_by: Optional[str] = None,
    alpha_range: Tuple[float, float] = (0.1, 1.0),
    alpha_clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
    alpha_invert: bool = False,
    prioritize_high_values: bool = False,
    overlap_priority: Optional[str] = None,
    segment_color_by_major: bool = False,
    title: Optional[str] = None,
    use_imshow: bool = True,
    brighten_continuous: bool = False,
    continuous_gamma: float = 0.8,
    # Boundary-related options for preview overlay
    plot_boundaries: bool = False,
    boundary_method: str = "pixel",
    boundary_color: str = "black",
    boundary_linewidth: float = 1.0,
    boundary_style: str = "solid",
    boundary_alpha: float = 0.7,
    pixel_smoothing_sigma: float = 0.0,
    highlight_segments: Optional[Sequence[str]] = None,
    highlight_boundary_color: Optional[str] = None,
    highlight_linewidth: Optional[float] = None,
    highlight_brighten_factor: float = 1.2,
    dim_other_segments: float = 0.3,
    dim_to_grey: bool = True,
    segment_key: Optional[str] = "Segment",
    runtime_fill_from_boundary: bool = False,
    runtime_fill_closing_radius: int = 1,
    runtime_fill_external_radius: int = 0,
    runtime_fill_holes: bool = True,
    gap_collapse_axis: str | None = None,
    gap_collapse_max_run_px: int = 1,
    soft_rasterization: bool = False,
    resolve_center_collisions: bool | str = "auto",
    center_collision_radius: int = 2,
    **plot_kwargs,
) -> Dict[str, Any]:
    """Rasterize embeddings + spatial coords to an image and cache in ``adata.uns``.

    The cached dict includes the image, geometry metadata, and a tile→obs
    mapping that allows later algorithms (e.g., SLIC) to sample labels
    for each tile efficiently without re-rasterizing.

    Supports the same origin-preserving raster options used by
    ``image_plot_slic`` and ``image_plot``:
    runtime boundary fill, soft bilinear rasterization, and local center
    collision resolution.

    Returns the cached dict and stores it under ``adata.uns[key]``.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")
    t_total0 = time.perf_counter() if verbose else None

    impl = (implementation or "auto").lower()
    if impl not in {"auto", "optimized", "legacy"}:
        raise ValueError("implementation must be one of {'auto','optimized','legacy'}.")
    if impl == "legacy":
        cache = _cache_embedding_image_legacy(
            adata,
            embedding=embedding,
            dimensions=dimensions,
            origin=origin,
            key=key,
            coordinate_mode=coordinate_mode,
            array_row_key=array_row_key,
            array_col_key=array_col_key,
            figsize=figsize,
            fig_dpi=fig_dpi,
            imshow_tile_size=imshow_tile_size,
            imshow_scale_factor=imshow_scale_factor,
            imshow_tile_size_mode=imshow_tile_size_mode,
            imshow_tile_size_quantile=imshow_tile_size_quantile,
            imshow_tile_size_rounding=imshow_tile_size_rounding,
            imshow_tile_size_shrink=imshow_tile_size_shrink,
            pixel_perfect=pixel_perfect,
            pixel_shape=pixel_shape,
            chunk_size=chunk_size,
            downsample_factor=downsample_factor,
            verbose=verbose,
            show=show,
            show_channels=show_channels,
            show_cmap=show_cmap,
            color_by=color_by,
            palette=palette,
            alpha_by=alpha_by,
            alpha_range=alpha_range,
            alpha_clip=alpha_clip,
            alpha_invert=alpha_invert,
            prioritize_high_values=prioritize_high_values,
            overlap_priority=overlap_priority,
            segment_color_by_major=segment_color_by_major,
            title=title,
            use_imshow=use_imshow,
            brighten_continuous=brighten_continuous,
            continuous_gamma=continuous_gamma,
            plot_boundaries=plot_boundaries,
            boundary_method=boundary_method,
            boundary_color=boundary_color,
            boundary_linewidth=boundary_linewidth,
            boundary_style=boundary_style,
            boundary_alpha=boundary_alpha,
            pixel_smoothing_sigma=pixel_smoothing_sigma,
            highlight_segments=highlight_segments,
            highlight_boundary_color=highlight_boundary_color,
            highlight_linewidth=highlight_linewidth,
            highlight_brighten_factor=highlight_brighten_factor,
            dim_other_segments=dim_other_segments,
            dim_to_grey=dim_to_grey,
            segment_key=segment_key,
            runtime_fill_from_boundary=runtime_fill_from_boundary,
            runtime_fill_closing_radius=runtime_fill_closing_radius,
            runtime_fill_external_radius=runtime_fill_external_radius,
            runtime_fill_holes=runtime_fill_holes,
            gap_collapse_axis=gap_collapse_axis,
            gap_collapse_max_run_px=gap_collapse_max_run_px,
            soft_rasterization=soft_rasterization,
            resolve_center_collisions=resolve_center_collisions,
            center_collision_radius=center_collision_radius,
            **plot_kwargs,
        )
        adata.uns[key] = cache
        return _QuietCacheDict(cache)

    store = (store or "auto").lower()
    if store not in {"auto", "memory", "memmap"}:
        raise ValueError("store must be one of {'auto','memory','memmap'}.")
    memmap_scope = (memmap_scope or "persistent").lower()
    if memmap_scope not in _MEMMAP_SCOPE_DIRS:
        raise ValueError("memmap_scope must be one of {'persistent','transient'}.")
    cache_storage = _normalize_cache_storage(cache_storage)
    memmap_prune = (memmap_prune or "none").lower()
    if memmap_prune not in {"none", "key"}:
        raise ValueError("memmap_prune must be one of {'none','key'}.")
    memmap_cleanup = (memmap_cleanup or "none").lower()
    if memmap_cleanup not in {"none", "on_exit"}:
        raise ValueError("memmap_cleanup must be one of {'none','on_exit'}.")
    store_barcodes = (store_barcodes or "auto").lower()
    if store_barcodes not in {"auto", "always", "never"}:
        raise ValueError("store_barcodes must be one of {'auto','always','never'}.")

    # For origin=True, prefer direct display coordinates from adata.obsm['spatial']
    # to avoid reusing compacted coords-as-tiles geometry from adata.uns['tiles'].
    use_direct_origin_spatial = False
    spatial_direct = None
    if origin:
        try:
            spatial_direct = np.asarray(adata.obsm["spatial"])
        except Exception:
            spatial_direct = None
        if (
            spatial_direct is not None
            and spatial_direct.ndim == 2
            and spatial_direct.shape[0] == adata.n_obs
            and spatial_direct.shape[1] >= 2
        ):
            use_direct_origin_spatial = True

    if use_direct_origin_spatial:
        tile_obs_indices = np.arange(int(adata.n_obs), dtype=np.int64)
        tile_barcodes = adata.obs.index.astype(str).to_numpy()
        spatial_coords = np.asarray(spatial_direct[:, :2], dtype=float)
        valid_mask = np.isfinite(spatial_coords).all(axis=1)
        if not np.any(valid_mask):
            raise ValueError("No finite coordinates found in adata.obsm['spatial'].")
        if verbose:
            missing = int((~valid_mask).sum())
            if missing:
                print(f"[cache_embedding_image] Dropping {missing} rows with non-finite direct spatial coordinates.")
        tile_obs_indices = tile_obs_indices[valid_mask]
        tile_barcodes = tile_barcodes[valid_mask]
        spatial_coords = spatial_coords[valid_mask]
        origin_flags = np.ones(int(spatial_coords.shape[0]), dtype=np.int8)
    else:
        # Align tiles to obs index. Keep raster duplicates when origin=False.
        tiles_df = adata.uns["tiles"]
        inferred_info = adata.uns.get("tiles_inferred_grid_info", {})
        inferred_mode = str(inferred_info.get("mode", "")).lower()
        if origin:
            tiles = tiles_df[tiles_df.get("origin", 1) == 1]
        else:
            if inferred_mode == "boundary_voronoi" and ("origin" in tiles_df.columns):
                tiles = tiles_df[tiles_df["origin"] == 0]
                if verbose:
                    print("[cache_embedding_image] Using inferred boundary support-grid tiles only for origin=False caching.")
            else:
                tiles = tiles_df
        if "barcode" not in tiles.columns:
            raise ValueError("adata.uns['tiles'] must contain a 'barcode' column.")
        if "x" not in tiles.columns or "y" not in tiles.columns:
            raise ValueError("adata.uns['tiles'] must contain 'x' and 'y' columns.")

        obs_index = pd.Index(adata.obs.index.astype(str))
        tiles_idx = tiles.copy()
        tiles_idx["barcode"] = tiles_idx["barcode"].astype(str)
        if tiles_idx["barcode"].duplicated().any():
            origin_vals = pd.to_numeric(tiles_idx.get("origin", 1), errors="coerce").fillna(1)
            has_non_origin = (origin_vals != 1).any()
            keep_dups = (not origin) and has_non_origin
            if keep_dups:
                if verbose:
                    print(
                        "[cache_embedding_image] Duplicated barcodes detected (origin=False raster tiles); keeping duplicates."
                    )
            else:
                if verbose:
                    print(
                        "[cache_embedding_image] Duplicated barcodes detected; keeping first occurrence for caching."
                    )
                tiles_idx = tiles_idx.drop_duplicates("barcode", keep="first")

        tile_barcodes = tiles_idx["barcode"].to_numpy()
        tile_obs_indices = obs_index.get_indexer(tile_barcodes)
        xy = tiles_idx[["x", "y"]]
        valid_mask = (tile_obs_indices >= 0) & (~xy.isna().any(axis=1)).to_numpy()
        if not np.any(valid_mask):
            raise ValueError("No coordinates found after aligning tiles to adata.obs.")
        if verbose:
            missing = int((~valid_mask).sum())
            if missing:
                print(f"[cache_embedding_image] Dropping {missing} tiles with missing coordinates or unmatched barcodes.")

        tile_obs_indices = tile_obs_indices[valid_mask].astype(np.int64, copy=False)
        tile_barcodes = tile_barcodes[valid_mask]
        spatial_coords = xy.to_numpy(dtype=float)[valid_mask]
        origin_flags = pd.to_numeric(tiles_idx.get("origin", 1), errors="coerce").fillna(1).to_numpy()[valid_mask]
    t_coords = time.perf_counter() if verbose else None

    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        if origin and (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
            coord_mode = "visiumhd"
        else:
            coord_mode = "spatial"

    if coord_mode in {"array", "visium", "visiumhd"}:
        if (array_row_key not in adata.obs.columns) or (array_col_key not in adata.obs.columns):
            raise ValueError(
                f"coordinate_mode='array' requires adata.obs['{array_row_key}'] and adata.obs['{array_col_key}']."
            )
        col = pd.to_numeric(adata.obs[array_col_key], errors="coerce").to_numpy(dtype=float)
        row = pd.to_numeric(adata.obs[array_row_key], errors="coerce").to_numpy(dtype=float)
        col_tile = col[tile_obs_indices]
        row_tile = row[tile_obs_indices]
        coord_mask = (~np.isnan(col_tile)) & (~np.isnan(row_tile))
        if verbose:
            missing_array = int((~coord_mask).sum())
            if missing_array:
                print(f"[cache_embedding_image] Dropping {missing_array} tiles missing array coordinates.")
        if coord_mode == "visium":
            parity = (row_tile.astype(np.int64, copy=False) % 2).astype(np.float64)
            x = col_tile.astype(np.float64, copy=False) + 0.5 * parity
            y = row_tile.astype(np.float64, copy=False) * (np.sqrt(3.0) / 2.0)
            spatial_coords = np.column_stack([x, y])[coord_mask]
        else:
            spatial_coords = np.column_stack([col_tile, row_tile])[coord_mask]
        tile_obs_indices = tile_obs_indices[coord_mask]
        tile_barcodes = tile_barcodes[coord_mask]
        origin_flags = origin_flags[coord_mask]
    t_coordmode = time.perf_counter() if verbose else None

    # Select embedding channels (support embeddings that already contain only these dims)
    E_full = np.asarray(adata.obsm[embedding])
    if E_full.ndim != 2:
        raise ValueError(f"Embedding '{embedding}' must be a 2D array.")
    if E_full.shape[0] != adata.n_obs:
        raise ValueError(f"Embedding '{embedding}' must have n_obs rows.")
    if E_full.shape[1] == len(dimensions):
        emb = E_full
    else:
        emb = E_full[:, dimensions]
    embeddings = emb[tile_obs_indices]
    barcodes = tile_barcodes
    dim_cols = [f"dim{i}" for i, _ in enumerate(range(embeddings.shape[1]))]
    t_embed = time.perf_counter() if verbose else None

    segment_series = None
    seg_col: Optional[str] = None
    segment_available = False
    if segment_key is not None:
        candidate = segment_key
        if candidate in adata.obs.columns:
            seg_col = candidate
            segment_available = True
        elif candidate != "Segment" and "Segment" in adata.obs.columns:
            if verbose:
                print(f"[cache_embedding_image] segment_key '{candidate}' not found; using 'Segment' column instead.")
            seg_col = "Segment"
            segment_available = True
        elif verbose:
            print(f"[cache_embedding_image] segment_key '{candidate}' not found in adata.obs; segment overlays disabled.")

    if segment_available and seg_col is not None:
        segment_series = adata.obs[seg_col]
        seg_vals = segment_series.to_numpy()[tile_obs_indices]
        seg_missing = pd.isna(seg_vals)
        if seg_missing.any():
            missing = int(seg_missing.sum())
            if verbose:
                print(f"[cache_embedding_image] '{seg_col}' labels missing for {missing} barcodes. Dropping them.")
            keep = ~seg_missing
            spatial_coords = spatial_coords[keep]
            embeddings = embeddings[keep]
            tile_obs_indices = tile_obs_indices[keep]
            barcodes = barcodes[keep]
            origin_flags = origin_flags[keep]
        segment_series = adata.obs[seg_col]
    else:
        segment_available = False
        seg_col = None

    # Geometry and pixel scaling. Use the full raster tile count for origin=False
    # so auto-sizing preserves all raster tiles instead of collapsing to spot count.
    n_points_eff = int(spatial_coords.shape[0])

    x_min, x_max = spatial_coords[:, 0].min(), spatial_coords[:, 0].max()
    y_min, y_max = spatial_coords[:, 1].min(), spatial_coords[:, 1].max()

    imshow_tile_size_eff = imshow_tile_size
    if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
        imshow_tile_size_eff = 1.0
    user_figsize_input = figsize
    user_fig_dpi_input = fig_dpi
    raster_canvas = _raster.resolve_raster_canvas(
        pts=spatial_coords,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        figsize=figsize,
        fig_dpi=fig_dpi,
        imshow_tile_size=imshow_tile_size_eff,
        imshow_scale_factor=imshow_scale_factor,
        imshow_tile_size_mode=imshow_tile_size_mode,
        imshow_tile_size_quantile=imshow_tile_size_quantile,
        imshow_tile_size_rounding=imshow_tile_size_rounding,
        imshow_tile_size_shrink=imshow_tile_size_shrink,
        pixel_perfect=bool(pixel_perfect),
        n_points=int(n_points_eff),
        logger=None,
        context="cache_embedding_image",
        persistent_store=getattr(adata, "uns", None),
    )
    figsize = raster_canvas["figsize"]
    fig_dpi = raster_canvas["fig_dpi"]
    dpi_in = int(raster_canvas["dpi_in"])
    s_data_units = float(raster_canvas["s_data_units"])
    x_min_eff = float(raster_canvas["x_min_eff"])
    x_max_eff = float(raster_canvas["x_max_eff"])
    y_min_eff = float(raster_canvas["y_min_eff"])
    y_max_eff = float(raster_canvas["y_max_eff"])
    scale = float(raster_canvas["scale"])
    s = int(raster_canvas["tile_px"])
    w = int(raster_canvas["w"])
    h = int(raster_canvas["h"])
    t_canvas = time.perf_counter() if verbose else None

    # Normalize embedding channels 0..1 per channel
    dims = embeddings.shape[1]
    scaler = MinMaxScaler()
    cols = scaler.fit_transform(embeddings.astype(np.float32))
    brighten_applied = bool(brighten_continuous and dims == 3)
    if brighten_applied:
        cols = _apply_brighten_continuous_rgb(cols, continuous_gamma)
    t_scale = time.perf_counter() if verbose else None

    if verbose:
        print(f"Rasterizing → image size (h, w, C)=({h}, {w}, {dims}) | tile_px={s}")

    est_mb = (float(h) * float(w) * float(dims) * 4.0) / (1024.0 * 1024.0)
    use_disk_cache = store == "memmap" or (store == "auto" and est_mb >= float(memmap_threshold_mb))
    img_path: Optional[str] = None
    work_img_path: Optional[str] = None
    cache_dir_resolved: Optional[str] = None
    cache_root_resolved: Optional[str] = None
    cache_namespace_resolved: Optional[str] = None
    cache_session_id: Optional[str] = None
    if use_disk_cache:
        cache_dir, cache_namespace_resolved, memmap_scope, cache_session_id = _resolve_memmap_cache_dir(
            adata,
            memmap_dir=memmap_dir,
            cache_namespace=cache_namespace,
            memmap_scope=memmap_scope,
        )
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_dir_resolved = cache_dir
        if memmap_dir is not None:
            cache_root_resolved = str(Path(memmap_dir).expanduser().resolve(strict=False))
        else:
            cache_root_resolved = str(Path(_default_memmap_dir()).expanduser().resolve(strict=False))
        safe_key = _safe_cache_component(key)
        safe_emb = _safe_cache_component(embedding)
        if downsample_factor and downsample_factor > 1.0:
            factor = int(downsample_factor)
            final_h = max(1, h // factor)
            final_w = max(1, w // factor)
        else:
            final_h, final_w = h, w
        img_path = _cache_disk_path(cache_dir, key, embedding, (final_h, final_w, dims), cache_storage)
        if cache_storage == "float32_memmap":
            work_img_path = img_path
        else:
            work_img_path = os.path.join(cache_dir, f"{safe_key}__{safe_emb}__{h}x{w}x{dims}.tmp.dat")
        if memmap_prune == "key":
            prefix = os.path.join(cache_dir, f"{safe_key}__{safe_emb}__")
            for old in glob.glob(prefix + "*"):
                try:
                    if os.path.isdir(old):
                        shutil.rmtree(old, ignore_errors=True)
                    else:
                        os.remove(old)
                except Exception:
                    pass
        if work_img_path != img_path and os.path.exists(work_img_path):
            try:
                os.remove(work_img_path)
            except Exception:
                pass
        img = np.memmap(work_img_path, dtype=np.float32, mode="w+", shape=(h, w, dims))
        img.fill(1.0)
        if verbose:
            print(
                f"[cache_embedding_image] Using disk-backed raster buffer at: {work_img_path} "
                f"(final={img_path}, storage={cache_storage}, ~{est_mb:.1f} MB raw)"
            )
        if memmap_cleanup == "on_exit":
            if img_path is not None:
                _MEMMAP_CLEANUP_PATHS.add(img_path)
            if work_img_path is not None:
                _MEMMAP_CLEANUP_PATHS.add(work_img_path)
    else:
        img = np.ones((h, w, dims), dtype=np.float32)
    paint_mask = np.zeros((h, w), dtype=bool)
    t_init = time.perf_counter() if verbose else None

    # Precompute pixel offsets for square/circle tiles (relative to center)
    dx, dy = _raster.tile_pixel_offsets(int(s), pixel_shape=pixel_shape)
    soft_enabled = bool(soft_rasterization)
    if soft_enabled and (not bool(pixel_perfect) or int(s) != 1 or str(pixel_shape).lower() != "square"):
        if verbose:
            print(
                "[cache_embedding_image] soft_rasterization requires pixel_perfect=True, tile_px==1, pixel_shape='square'; falling back to hard rasterization."
            )
        soft_enabled = False
    if soft_enabled:
        cx_seed, cy_seed, soft_fx_seed, soft_fy_seed = _raster.scale_points_to_canvas_soft(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=True,
            w=int(w),
            h=int(h),
        )
    else:
        soft_fx_seed = None
        soft_fy_seed = None
        cx_seed, cy_seed = _raster.scale_points_to_canvas(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=bool(pixel_perfect),
            w=int(w),
            h=int(h),
        )
    collision_info = None
    apply_collision_resolve, collision_stats, _ = _raster.should_resolve_center_collisions(
        resolve_center_collisions,
        cx=cx_seed,
        cy=cy_seed,
        w=int(w),
        h=int(h),
        tile_px=int(s),
        pixel_perfect=bool(pixel_perfect),
        soft_enabled=bool(soft_enabled),
        logger=None,
        context="cache_embedding_image",
    )
    if apply_collision_resolve:
        collision_radius_eff = _raster.resolve_collision_search_radius(
            resolve_center_collisions,
            base_radius=int(center_collision_radius),
            collision_stats=collision_stats,
            logger=None,
            context="cache_embedding_image",
        )
        cx_seed, cy_seed, collision_info = _raster.resolve_center_collisions(
            cx_seed,
            cy_seed,
            w=int(w),
            h=int(h),
            max_radius=int(collision_radius_eff),
            logger=None,
            context="cache_embedding_image",
        )
    elif collision_stats is not None:
        collision_info = {
            "applied": False,
            "enabled": _raster.normalize_collision_mode(resolve_center_collisions) != "never",
            "search_radius": int(max(0, center_collision_radius)),
            "n_tiles": int(collision_stats["n_tiles"]),
            "n_unique_centers_before": int(collision_stats["n_unique_centers"]),
            "n_unique_centers_after": int(collision_stats["n_unique_centers"]),
            "collision_fraction_before": float(collision_stats["collision_fraction"]),
            "collision_fraction_after": float(collision_stats["collision_fraction"]),
            "n_collided_tiles_before": int(collision_stats["n_collided_tiles"]),
            "n_collided_tiles_after": int(collision_stats["n_collided_tiles"]),
            "n_reassigned_tiles": 0,
            "max_stack_before": int(collision_stats["max_stack"]),
            "max_stack_after": int(collision_stats["max_stack"]),
            "unresolved_tiles": int(collision_stats["n_collided_tiles"]),
        }
    gap_collapse_axis_norm = _normalize_gap_collapse_axis_safe(gap_collapse_axis)
    gap_collapse_active = bool(
        gap_collapse_axis_norm is not None and int(max(0, int(gap_collapse_max_run_px))) > 0
    )
    if gap_collapse_active and soft_enabled:
        soft_enabled = False
        soft_fx_seed = None
        soft_fy_seed = None
    if gap_collapse_active:
        cx_seed, cy_seed, w, h, gap_collapse_info = _raster.collapse_empty_center_gaps(
            cx=cx_seed,
            cy=cy_seed,
            w=int(w),
            h=int(h),
            axis=gap_collapse_axis_norm,
            max_gap_run_px=int(gap_collapse_max_run_px),
            logger=None,
            context="cache_embedding_image",
        )
    else:
        gap_collapse_info = {
            "applied": False,
            "axis": gap_collapse_axis_norm,
            "max_gap_run_px": int(max(0, int(gap_collapse_max_run_px))),
        }
    t_centers = time.perf_counter() if verbose else None

    # Optional depth ordering similar to image_plot (alpha-driven)
    def _compute_alpha_from_values(values, arange=(0.1, 1.0), clip=None, invert=False):
        v = np.asarray(values, dtype=float)
        if clip is None:
            vmin, vmax = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
        else:
            vmin, vmax = clip
            if vmin is None:
                vmin = np.nanmin(v)
            if vmax is None:
                vmax = np.nanmax(v)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
        if invert:
            t = 1.0 - t
        a0, a1 = float(arange[0]), float(arange[1])
        a = a0 + t * (a1 - a0)
        return a.astype(np.float32)

    order_idx = np.arange(spatial_coords.shape[0])
    if alpha_by is not None:
        if alpha_by == "embedding":
            # Use the first selected dimension as alpha source
            alpha_source = embeddings[:, 0]
        else:
            if alpha_by not in adata.obs.columns:
                raise ValueError(f"alpha_by '{alpha_by}' not found in adata.obs.")
            # Build aligned values for present barcodes
            alpha_source = adata.obs[alpha_by].astype(float).to_numpy()[tile_obs_indices]
        alpha_arr = _compute_alpha_from_values(alpha_source, arange=alpha_range, clip=alpha_clip, invert=alpha_invert)
        if prioritize_high_values:
            # draw high alphas last (front)
            order_idx = np.argsort(alpha_arr)

    # Chunked rasterization in chosen order
    num_tiles = spatial_coords.shape[0]
    pixels_per_tile = int(len(dx))
    # Keep intermediate index arrays bounded.
    target_pixels = int(os.environ.get("SPIX_CACHE_RASTER_TARGET_PIXELS", "25000000"))
    chunk_n = int(max(1, min(int(chunk_size), int(max(1, target_pixels // max(1, pixels_per_tile))))))
    if soft_enabled:
        _raster.rasterize_soft_bilinear(
            img=img,
            occupancy_mask=paint_mask,
            seed_values=cols,
            cx=cx_seed,
            cy=cy_seed,
            fx=soft_fx_seed,
            fy=soft_fy_seed,
            chunk_size=int(chunk_n),
        )
    elif pixels_per_tile == 1 and pixel_shape != "circle":
        # Fast path for s==1 square tiles: assign one pixel per tile (center pixel).
        for i in range(0, num_tiles, chunk_n):
            idx = order_idx[i : min(i + chunk_n, num_tiles)]
            cx0 = cx_seed[idx]
            cy0 = cy_seed[idx]
            img[cy0, cx0] = cols[idx]
            paint_mask[cy0, cx0] = True
    else:
        for i in range(0, num_tiles, chunk_n):
            idx = order_idx[i : min(i + chunk_n, num_tiles)]
            cx0 = cx_seed[idx]
            cy0 = cy_seed[idx]
            all_x = np.clip((cx0[:, None] + dx[None, :]).ravel(), 0, w - 1)
            all_y = np.clip((cy0[:, None] + dy[None, :]).ravel(), 0, h - 1)
            tile_idx_map = np.repeat(idx, pixels_per_tile)
            img[all_y, all_x] = cols[tile_idx_map]
            paint_mask[all_y, all_x] = True
    t_raster = time.perf_counter() if verbose else None
    runtime_fill_info = {
        "applied": False,
        "filled_pixels": 0,
        "support_pixels": int(paint_mask.sum()),
        "closing_radius": int(max(0, int(runtime_fill_closing_radius))),
        "external_radius": int(max(0, int(runtime_fill_external_radius))),
        "fill_holes": bool(runtime_fill_holes),
    }
    fill_active = bool(runtime_fill_from_boundary) or bool(runtime_fill_holes)
    fill_radius = int(runtime_fill_closing_radius) if bool(runtime_fill_from_boundary) else 0
    external_radius = int(runtime_fill_external_radius) if bool(runtime_fill_from_boundary) else 0
    label_paint_mask = paint_mask.copy()
    if fill_active:
        runtime_fill_info = _raster.fill_raster_from_boundary(
            img=img,
            occupancy_mask=paint_mask,
            seed_x=cx_seed,
            seed_y=cy_seed,
            seed_values=cols,
            closing_radius=int(fill_radius),
            external_radius=int(external_radius),
            fill_holes=bool(runtime_fill_holes),
            use_observed_pixels=bool(int(s) == 1),
            persistent_store=getattr(adata, "uns", None),
            logger=None,
            context="cache_embedding_image",
        )
    support_mask = np.asarray(paint_mask, dtype=bool)
    t_fill = time.perf_counter() if verbose else None

    # Optional: build label image for boundary overlay (preview only)
    label_img = None
    if plot_boundaries and segment_available and segment_series is not None:
        # Map per-tile Segment labels
        seg_codes = pd.Categorical(segment_series.to_numpy()[tile_obs_indices]).codes
        label_img = np.full((h, w), -1, dtype=int)
        if soft_enabled:
            value_buf = np.full((h, w), -np.inf, dtype=np.float32)
            chunk_n = int(max(1, chunk_size))
            for i in range(0, num_tiles, chunk_n):
                idx = order_idx[i : min(i + chunk_n, num_tiles)]
                x0, y0, x1, y1, w00, w10, w01, w11 = _raster.bilinear_support_from_centers(
                    cx_seed[idx],
                    cy_seed[idx],
                    soft_fx_seed[idx],
                    soft_fy_seed[idx],
                    w=int(w),
                    h=int(h),
                )
                seg_chunk = seg_codes[idx]
                for px, py, pw in ((x0, y0, w00), (x1, y0, w10), (x0, y1, w01), (x1, y1, w11)):
                    mask_upd = pw > value_buf[py, px]
                    if np.any(mask_upd):
                        label_img[py[mask_upd], px[mask_upd]] = seg_chunk[mask_upd]
                        value_buf[py[mask_upd], px[mask_upd]] = pw[mask_upd]
        else:
            for i in range(0, num_tiles, max(1, chunk_size)):
                idx = order_idx[i : min(i + chunk_size, num_tiles)]
                cx0 = cx_seed[idx]
                cy0 = cy_seed[idx]
                if pixel_shape == "circle":
                    # Paint circle
                    for j, t in enumerate(idx):
                        lx = np.clip(cx0[j] + dx, 0, w - 1)
                        ly = np.clip(cy0[j] + dy, 0, h - 1)
                        label_img[ly, lx] = seg_codes[t]
                else:
                    all_x = np.clip((cx0[:, None] + dx[None, :]).ravel(), 0, w - 1)
                    all_y = np.clip((cy0[:, None] + dy[None, :]).ravel(), 0, h - 1)
                    label_img[all_y, all_x] = np.repeat(seg_codes[idx], len(dx))
        label_img[~label_paint_mask] = -1

    if downsample_factor and downsample_factor > 1.0:
        # Simple average pooling downsample to avoid extra deps
        factor = int(downsample_factor)
        new_h = max(1, h // factor)
        new_w = max(1, w // factor)
        pooled = img[: new_h * factor, : new_w * factor]
        pooled = pooled.reshape(new_h, factor, new_w, factor, dims).mean(axis=(1, 3))
        img = pooled.astype(np.float32)
        h, w = img.shape[:2]
        support_mask = _downsample_support_mask(support_mask, h, w)
        scale /= factor
        s = max(1, int(np.ceil(s_data_units * scale)))
        dpi_effective = float(dpi_in) / float(factor)
    else:
        dpi_effective = float(dpi_in)
    t_downsample = time.perf_counter() if verbose else None

    # If we built a label image, downsample it to match
    if label_img is not None and downsample_factor and downsample_factor > 1.0:
        factor = int(downsample_factor)
        new_h = max(1, label_img.shape[0] // factor)
        new_w = max(1, label_img.shape[1] // factor)
        # nearest pooling for labels
        label_img = label_img[: new_h * factor, : new_w * factor]
        label_img = label_img.reshape(new_h, factor, new_w, factor).max(axis=(1, 3))
    if label_img is not None and support_mask.shape == label_img.shape:
        label_img = np.asarray(label_img, dtype=np.int32)
        label_img[~support_mask] = -1

    # Compute sampling centers per tile (pixel indices)
    if downsample_factor and downsample_factor > 1.0 and soft_enabled:
        cx, cy, soft_fx, soft_fy = _raster.scale_points_to_canvas_soft(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=True,
            w=int(w),
            h=int(h),
        )
    elif downsample_factor and downsample_factor > 1.0 and collision_info is not None and bool(collision_info.get("applied", False)):
        factor = int(max(1, round(float(downsample_factor))))
        cx = np.clip((cx_seed // factor).astype(np.int32, copy=False), 0, max(0, w - 1))
        cy = np.clip((cy_seed // factor).astype(np.int32, copy=False), 0, max(0, h - 1))
    elif downsample_factor and downsample_factor > 1.0:
        cx, cy = _raster.scale_points_to_canvas(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=bool(pixel_perfect),
            w=int(w),
            h=int(h),
        )
    else:
        cx, cy = cx_seed, cy_seed
        soft_fx, soft_fy = soft_fx_seed, soft_fy_seed

    cache = {
        "img": img,
        "h": int(h),
        "w": int(w),
        "total_pixels": int(h * w),
        "channels": int(dims),
        "coordinate_mode": str(coord_mode),
        "array_row_key": str(array_row_key),
        "array_col_key": str(array_col_key),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "x_min_eff": float(x_min_eff),
        "x_max_eff": float(x_max_eff),
        "y_min_eff": float(y_min_eff),
        "y_max_eff": float(y_max_eff),
        "scale": float(scale),
        "s_data_units": float(s_data_units),
        "tile_px": int(s),
        "pixel_perfect": bool(pixel_perfect),
        "pixel_shape": str(pixel_shape),
        "embedding_key": str(embedding),
        "dimensions": list(dimensions),
        "origin": bool(origin),
        "figsize": None if user_figsize_input is None else tuple(user_figsize_input),
        "fig_dpi": None if user_fig_dpi_input is None else int(user_fig_dpi_input),
        "raster_dpi": int(dpi_in),
        "effective_dpi": float(dpi_effective),
        "downsample_factor": float(downsample_factor),
        # per-tile mapping/meta
        "cx": cx.astype(np.int32),
        "cy": cy.astype(np.int32),
        "soft_rasterization": bool(soft_enabled),
        "soft_rasterization_mode": "bilinear" if soft_enabled else None,
        "soft_fx": (soft_fx.astype(np.float16) if soft_enabled else None),
        "soft_fy": (soft_fy.astype(np.float16) if soft_enabled else None),
        "support_mask": support_mask.astype(bool, copy=False),
        "tile_obs_indices": tile_obs_indices.astype(np.int64),
        # Store barcodes only when explicitly requested or when the dataset is small.
        # For huge VisiumHD datasets, storing a Python list of millions of strings
        # can dominate RAM usage; downstream code can use `tile_obs_indices` instead.
        "barcodes": (
            [str(bc) for bc in barcodes.tolist()]
            if (
                store_barcodes == "always"
                or (store_barcodes == "auto" and int(len(barcodes)) <= int(barcodes_max_n))
            )
            else None
        ),
        "barcodes_stored": bool(
            store_barcodes == "always" or (store_barcodes == "auto" and int(len(barcodes)) <= int(barcodes_max_n))
        ),
        "origin_flags": np.asarray(origin_flags, dtype=np.uint8),
        "segment_key": seg_col,
        "segment_key_requested": segment_key,
        "store": store,
        "memmap": bool(use_disk_cache and cache_storage == "float32_memmap"),
        "memmap_dir": cache_dir_resolved,
        "cache_root": cache_root_resolved,
        "cache_namespace": cache_namespace_resolved,
        "cache_scope": memmap_scope if use_disk_cache else None,
        "cache_session_id": cache_session_id,
        "cache_storage": cache_storage if use_disk_cache else "memory",
        "img_path": img_path,
        # provenance
        "created_by": "cache_embedding_image",
        "history": [],
        "center_collision_resolution": (
            dict(collision_info)
            if collision_info is not None
            else {
                "applied": False,
                "enabled": False,
                "search_radius": int(center_collision_radius),
                "ignored_due_to_soft_rasterization": bool(soft_enabled and resolve_center_collisions),
                "n_tiles": int(num_tiles),
                "n_unique_centers_before": None,
                "n_unique_centers_after": None,
                "collision_fraction_before": None,
                "collision_fraction_after": None,
                "n_collided_tiles_before": None,
                "n_collided_tiles_after": None,
                "n_reassigned_tiles": 0,
                "max_stack_before": None,
                "max_stack_after": None,
                "unresolved_tiles": 0,
            }
        ),
        # image_plot compatibility: store provided params for traceability
        "image_plot_params": {
            "color_by": color_by,
            "palette": palette,
            "alpha_by": alpha_by,
            "alpha_range": tuple(alpha_range) if alpha_range is not None else None,
            "alpha_clip": alpha_clip,
            "alpha_invert": bool(alpha_invert),
            "prioritize_high_values": bool(prioritize_high_values),
            "overlap_priority": overlap_priority,
            "segment_color_by_major": bool(segment_color_by_major),
            "title": title,
            "use_imshow": bool(use_imshow),
            "brighten_continuous": bool(brighten_continuous),
            "continuous_gamma": float(continuous_gamma),
            "brighten_applied": bool(brighten_applied),
            # boundary params
            "plot_boundaries": bool(plot_boundaries),
            "boundary_method": boundary_method,
            "boundary_color": boundary_color,
            "boundary_linewidth": float(boundary_linewidth),
            "boundary_style": boundary_style,
            "boundary_alpha": float(boundary_alpha),
            "pixel_smoothing_sigma": float(pixel_smoothing_sigma),
            "highlight_segments": list(highlight_segments) if highlight_segments is not None else None,
            "highlight_boundary_color": highlight_boundary_color,
            "highlight_linewidth": None if highlight_linewidth is None else float(highlight_linewidth),
            "highlight_brighten_factor": float(highlight_brighten_factor),
            "dim_other_segments": float(dim_other_segments),
            "dim_to_grey": bool(dim_to_grey),
            "segment_key": seg_col,
            "segment_key_requested": segment_key,
            "runtime_fill_from_boundary": bool(runtime_fill_from_boundary),
            "runtime_fill_closing_radius": int(max(0, int(runtime_fill_closing_radius))),
            "runtime_fill_external_radius": int(max(0, int(runtime_fill_external_radius))),
            "runtime_fill_holes": bool(runtime_fill_holes),
            "gap_collapse_axis": gap_collapse_axis_norm,
            "gap_collapse_max_run_px": int(max(0, int(gap_collapse_max_run_px))),
            "soft_rasterization": bool(soft_enabled),
            "resolve_center_collisions": _raster.normalize_collision_mode(resolve_center_collisions),
            "center_collision_radius": int(center_collision_radius),
            **plot_kwargs,
        },
        "runtime_fill": runtime_fill_info,
        "gap_collapse": dict(gap_collapse_info),
    }

    # Optional: build preview RGB with boundary overlay
    if plot_boundaries and label_img is not None and boundary_method == "pixel":
        # base RGB from img (1,2,3 channels)
        if img.shape[2] >= 3:
            base = img[:, :, :3].copy()
        elif img.shape[2] == 2:
            base = np.stack([img[:, :, 0], img[:, :, 1], np.zeros_like(img[:, :, 0])], axis=2)
        else:
            base = np.repeat(img, 3, axis=2)
        # boundary mask
        bmask = find_boundaries(label_img, mode="outer", background=-1, connectivity=2)
        if pixel_smoothing_sigma and pixel_smoothing_sigma > 0:
            sm = gaussian_filter(bmask.astype(np.float32), pixel_smoothing_sigma)
            bmask = sm >= 0.5
        # line width via binary dilation
        lw = max(1, int(round(boundary_linewidth)))
        if lw > 1:
            bmask = binary_dilation(bmask, iterations=lw - 1)
        # color setup
        base_col = np.array(mcolors.to_rgb(boundary_color), dtype=np.float32)
        if highlight_segments is not None and segment_available and segment_series is not None:
            cats = pd.Categorical(segment_series).categories
            codes = [cats.get_loc(seg) for seg in highlight_segments if seg in cats]
            if codes:
                hmask = np.isin(label_img, np.array(codes, dtype=int)) & bmask
            else:
                hmask = np.zeros_like(bmask)
        else:
            hmask = np.zeros_like(bmask)
        if highlight_boundary_color is not None:
            high_col = np.array(mcolors.to_rgb(highlight_boundary_color), dtype=np.float32)
        else:
            high_col = base_col
        # blend
        out = base.copy()
        a = float(boundary_alpha)
        for c in range(3):
            # base boundary
            out[..., c] = np.where(bmask & ~hmask, a * base_col[c] + (1 - a) * out[..., c], out[..., c])
            # highlight boundary overrides
            out[..., c] = np.where(hmask, a * high_col[c] + (1 - a) * out[..., c], out[..., c])
        cache["img_preview"] = np.clip(out, 0.0, 1.0).astype(np.float32)
        cache["label_img"] = label_img.astype(np.int32)
    t_postprocess = time.perf_counter() if verbose else None

    if use_disk_cache:
        if isinstance(img, np.memmap):
            try:
                img.flush()
            except Exception:
                pass
        if cache_storage == "float32_memmap":
            cache["img"] = img
        else:
            _write_cache_image_to_disk(img, img_path, cache_storage)
            if isinstance(img, np.memmap):
                try:
                    mmap_obj = getattr(img, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()
                except Exception:
                    pass
            if work_img_path is not None and work_img_path != img_path:
                try:
                    os.remove(work_img_path)
                except Exception:
                    pass
                _MEMMAP_CLEANUP_PATHS.discard(str(work_img_path))
            cache["img"] = None
    t_disk_write = time.perf_counter() if verbose else None

    adata.uns[key] = cache
    t_finalize = time.perf_counter() if verbose else None
    if verbose and all(v is not None for v in [t_total0, t_coords, t_coordmode, t_embed, t_canvas, t_scale, t_init, t_centers, t_raster, t_fill, t_downsample, t_postprocess, t_disk_write, t_finalize]):
        print(
            "[perf] cache_embedding_image detail: "
            f"coords={t_coords - t_total0:.3f}s | "
            f"coord_mode={t_coordmode - t_coords:.3f}s | "
            f"embedding_slice={t_embed - t_coordmode:.3f}s | "
            f"canvas={t_canvas - t_embed:.3f}s | "
            f"scale={t_scale - t_canvas:.3f}s | "
            f"init={t_init - t_scale:.3f}s | "
            f"centers={t_centers - t_init:.3f}s | "
            f"rasterize={t_raster - t_centers:.3f}s | "
            f"fill={t_fill - t_raster:.3f}s | "
            f"downsample={t_downsample - t_fill:.3f}s | "
            f"postprocess={t_postprocess - t_downsample:.3f}s | "
            f"disk_write={t_disk_write - t_postprocess:.3f}s | "
            f"finalize={t_finalize - t_disk_write:.3f}s | "
            f"total={t_finalize - t_total0:.3f}s"
        )
    if show:
        try:
            show_cached_image(adata, key=key, channels=show_channels, cmap=show_cmap)
        except Exception:
            pass
    # Return a compact-repr dict to avoid printing huge tensors/barcode lists in notebooks.
    return _QuietCacheDict(cache)


def _cache_embedding_image_legacy(
    adata,
    *,
    embedding: str,
    dimensions: List[int],
    origin: bool,
    key: str,
    coordinate_mode: str,
    array_row_key: str,
    array_col_key: str,
    figsize: tuple | None,
    fig_dpi: Optional[int],
    imshow_tile_size: Optional[float],
    imshow_scale_factor: float,
    imshow_tile_size_mode: str,
    imshow_tile_size_quantile: Optional[float],
    imshow_tile_size_rounding: str,
    imshow_tile_size_shrink: float,
    pixel_perfect: bool,
    pixel_shape: str,
    chunk_size: int,
    downsample_factor: float,
    verbose: bool,
    show: bool,
    show_channels: Optional[Sequence[int]],
    show_cmap: Optional[str],
    color_by: Optional[str],
    palette: Optional[str | Sequence],
    alpha_by: Optional[str],
    alpha_range: Tuple[float, float],
    alpha_clip: Optional[Tuple[Optional[float], Optional[float]]],
    alpha_invert: bool,
    prioritize_high_values: bool,
    overlap_priority: Optional[str],
    segment_color_by_major: bool,
    title: Optional[str],
    use_imshow: bool,
    brighten_continuous: bool,
    continuous_gamma: float,
    plot_boundaries: bool,
    boundary_method: str,
    boundary_color: str,
    boundary_linewidth: float,
    boundary_style: str,
    boundary_alpha: float,
    pixel_smoothing_sigma: float,
    highlight_segments: Optional[Sequence[str]],
    highlight_boundary_color: Optional[str],
    highlight_linewidth: Optional[float],
    highlight_brighten_factor: float,
    dim_other_segments: float,
    dim_to_grey: bool,
    segment_key: Optional[str],
    runtime_fill_from_boundary: bool,
    runtime_fill_closing_radius: int,
    runtime_fill_external_radius: int,
    runtime_fill_holes: bool,
    gap_collapse_axis: str | None,
    gap_collapse_max_run_px: int,
    soft_rasterization: bool,
    resolve_center_collisions: bool,
    center_collision_radius: int,
    **plot_kwargs,
) -> Dict[str, Any]:
    """Legacy cache implementation (baseline for benchmarking)."""
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    tiles_df = adata.uns["tiles"]
    inferred_info = adata.uns.get("tiles_inferred_grid_info", {})
    inferred_mode = str(inferred_info.get("mode", "")).lower()
    use_direct_origin_spatial = False
    if origin:
        spatial_direct = None
        try:
            spatial_direct = np.asarray(adata.obsm["spatial"])
        except Exception:
            spatial_direct = None
        if (
            spatial_direct is not None
            and spatial_direct.ndim == 2
            and spatial_direct.shape[0] == adata.n_obs
            and spatial_direct.shape[1] >= 2
        ):
            use_direct_origin_spatial = True
        else:
            tiles = tiles_df[tiles_df.get("origin", 1) == 1]
    else:
        if inferred_mode == "boundary_voronoi" and ("origin" in tiles_df.columns):
            tiles = tiles_df[tiles_df["origin"] == 0]
            if verbose:
                print("[cache_embedding_image] Using inferred boundary support-grid tiles only for origin=False caching (legacy).")
        else:
            tiles = tiles_df

    E_full = np.asarray(adata.obsm[embedding])
    if E_full.shape[1] == len(dimensions):
        emb = E_full
    else:
        emb = E_full[:, dimensions]

    dim_cols = [f"dim{i}" for i in range(emb.shape[1])]
    if use_direct_origin_spatial:
        coordinates_df = pd.DataFrame({
            "x": np.asarray(spatial_direct[:, 0], dtype=float),
            "y": np.asarray(spatial_direct[:, 1], dtype=float),
            "barcode": adata.obs.index.astype(str),
            "origin": np.ones(int(adata.n_obs), dtype=np.int8),
        })
        coordinates_df.loc[:, dim_cols] = emb
    else:
        tile_colors = pd.DataFrame(emb, columns=dim_cols)
        tile_colors["barcode"] = adata.obs.index.astype(str)
        coordinates_df = (
            pd.merge(tiles, tile_colors, on="barcode", how="right")
            .dropna()
            .reset_index(drop=True)
        )
        if len(coordinates_df) == 0:
            raise ValueError("No coordinates found after merging tiles with embeddings.")

    coord_mode = str(coordinate_mode or "spatial").lower()
    if coord_mode not in {"spatial", "array", "visium", "visiumhd", "auto"}:
        coord_mode = "spatial"
    if coord_mode == "auto":
        if origin and (array_row_key in adata.obs.columns) and (array_col_key in adata.obs.columns) and int(adata.n_obs) >= 200_000:
            coord_mode = "visiumhd"
        else:
            coord_mode = "spatial"
    if coord_mode in {"array", "visium", "visiumhd"}:
        if (array_row_key not in adata.obs.columns) or (array_col_key not in adata.obs.columns):
            raise ValueError(
                f"coordinate_mode='array' requires adata.obs['{array_row_key}'] and adata.obs['{array_col_key}']."
            )
        coordinates_df[array_col_key] = coordinates_df["barcode"].map(adata.obs[array_col_key])
        coordinates_df[array_row_key] = coordinates_df["barcode"].map(adata.obs[array_row_key])
        col = pd.to_numeric(coordinates_df[array_col_key], errors="coerce")
        row = pd.to_numeric(coordinates_df[array_row_key], errors="coerce")
        if coord_mode == "visium":
            row_i = row.astype("Int64")
            parity = (row_i % 2).astype(float)
            coordinates_df["x"] = col.astype(float) + 0.5 * parity
            coordinates_df["y"] = row.astype(float) * (np.sqrt(3.0) / 2.0)
        else:
            coordinates_df["x"] = col.astype(float)
            coordinates_df["y"] = row.astype(float)

    segment_series = None
    seg_col: Optional[str] = None
    segment_available = False
    if segment_key is not None:
        candidate = segment_key
        if candidate in adata.obs.columns:
            seg_col = candidate
            segment_available = True
        elif candidate != "Segment" and "Segment" in adata.obs.columns:
            if verbose:
                print(
                    f"[cache_embedding_image] segment_key '{candidate}' not found; using 'Segment' column instead."
                )
            seg_col = "Segment"
            segment_available = True
        elif verbose:
            print(
                f"[cache_embedding_image] segment_key '{candidate}' not found in adata.obs; segment overlays disabled."
            )

    if segment_available and seg_col is not None:
        coordinates_df[seg_col] = coordinates_df["barcode"].map(adata.obs[seg_col])
        if coordinates_df[seg_col].isnull().any():
            missing = int(coordinates_df[seg_col].isnull().sum())
            if verbose:
                print(
                    f"[cache_embedding_image] '{seg_col}' labels missing for {missing} barcodes. Dropping them."
                )
            coordinates_df = coordinates_df.dropna(subset=[seg_col])
        coordinates_df["Segment"] = coordinates_df[seg_col]
        segment_series = adata.obs[seg_col]
    else:
        coordinates_df = coordinates_df.drop(columns=["Segment"], errors="ignore")
        segment_available = False
        seg_col = None

    obs_indexer = {bc: i for i, bc in enumerate(adata.obs.index.astype(str))}
    tile_obs_indices = coordinates_df["barcode"].astype(str).map(obs_indexer).to_numpy()
    spatial_coords = coordinates_df[["x", "y"]].to_numpy(dtype=float)
    embeddings = coordinates_df[dim_cols].to_numpy(dtype=float)

    # Geometry and pixel scaling
    # Geometry and pixel scaling. Use the full raster tile count for origin=False
    # so auto-sizing preserves raster detail.
    n_points_eff = int(spatial_coords.shape[0])

    x_min, x_max = spatial_coords[:, 0].min(), spatial_coords[:, 0].max()
    y_min, y_max = spatial_coords[:, 1].min(), spatial_coords[:, 1].max()

    imshow_tile_size_eff = imshow_tile_size
    if coord_mode == "visiumhd" and imshow_tile_size_eff is None:
        imshow_tile_size_eff = 1.0
    user_figsize_input = figsize
    user_fig_dpi_input = fig_dpi
    raster_canvas = _raster.resolve_raster_canvas(
        pts=spatial_coords,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        figsize=figsize,
        fig_dpi=fig_dpi,
        imshow_tile_size=imshow_tile_size_eff,
        imshow_scale_factor=imshow_scale_factor,
        imshow_tile_size_mode=imshow_tile_size_mode,
        imshow_tile_size_quantile=imshow_tile_size_quantile,
        imshow_tile_size_rounding="ceil",
        imshow_tile_size_shrink=imshow_tile_size_shrink,
        pixel_perfect=bool(pixel_perfect),
        n_points=int(n_points_eff),
        logger=None,
        context="cache_embedding_image(legacy)",
        persistent_store=getattr(adata, "uns", None),
    )
    figsize = raster_canvas["figsize"]
    fig_dpi = raster_canvas["fig_dpi"]
    dpi_in = int(raster_canvas["dpi_in"])
    s_data_units = float(raster_canvas["s_data_units"])
    x_min_eff = float(raster_canvas["x_min_eff"])
    x_max_eff = float(raster_canvas["x_max_eff"])
    y_min_eff = float(raster_canvas["y_min_eff"])
    y_max_eff = float(raster_canvas["y_max_eff"])
    scale = float(raster_canvas["scale"])
    s = int(raster_canvas["tile_px"])
    w = int(raster_canvas["w"])
    h = int(raster_canvas["h"])

    dims = embeddings.shape[1]
    scaler = MinMaxScaler()
    cols = scaler.fit_transform(embeddings.astype(np.float32))
    brighten_applied = bool(brighten_continuous and dims == 3)
    if brighten_applied:
        cols = _apply_brighten_continuous_rgb(cols, continuous_gamma)

    if verbose:
        print(f"Rasterizing → image size (h, w, C)=({h}, {w}, {dims}) | tile_px={s}")

    img = np.ones((h, w, dims), dtype=np.float32)

    # Precompute pixel offsets for square/circle tiles (relative to center)
    dx, dy = _raster.tile_pixel_offsets(int(s), pixel_shape=pixel_shape)
    soft_enabled = bool(soft_rasterization)
    if soft_enabled and (not bool(pixel_perfect) or int(s) != 1 or str(pixel_shape).lower() != "square"):
        if verbose:
            print(
                "[cache_embedding_image] soft_rasterization requires pixel_perfect=True, tile_px==1, pixel_shape='square'; falling back to hard rasterization (legacy)."
            )
        soft_enabled = False
    if soft_enabled:
        cx_seed, cy_seed, soft_fx_seed, soft_fy_seed = _raster.scale_points_to_canvas_soft(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=True,
            w=int(w),
            h=int(h),
        )
    else:
        soft_fx_seed = None
        soft_fy_seed = None
        cx_seed, cy_seed = _raster.scale_points_to_canvas(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=bool(pixel_perfect),
            w=int(w),
            h=int(h),
        )
    collision_info = None
    apply_collision_resolve, collision_stats, _ = _raster.should_resolve_center_collisions(
        resolve_center_collisions,
        cx=cx_seed,
        cy=cy_seed,
        w=int(w),
        h=int(h),
        tile_px=int(s),
        pixel_perfect=bool(pixel_perfect),
        soft_enabled=bool(soft_enabled),
        logger=None,
        context="cache_embedding_image(legacy)",
    )
    if apply_collision_resolve:
        collision_radius_eff = _raster.resolve_collision_search_radius(
            resolve_center_collisions,
            base_radius=int(center_collision_radius),
            collision_stats=collision_stats,
            logger=None,
            context="cache_embedding_image(legacy)",
        )
        cx_seed, cy_seed, collision_info = _raster.resolve_center_collisions(
            cx_seed,
            cy_seed,
            w=int(w),
            h=int(h),
            max_radius=int(collision_radius_eff),
            logger=None,
            context="cache_embedding_image(legacy)",
        )
    elif collision_stats is not None:
        collision_info = {
            "applied": False,
            "enabled": _raster.normalize_collision_mode(resolve_center_collisions) != "never",
            "search_radius": int(max(0, center_collision_radius)),
            "n_tiles": int(collision_stats["n_tiles"]),
            "n_unique_centers_before": int(collision_stats["n_unique_centers"]),
            "n_unique_centers_after": int(collision_stats["n_unique_centers"]),
            "collision_fraction_before": float(collision_stats["collision_fraction"]),
            "collision_fraction_after": float(collision_stats["collision_fraction"]),
            "n_collided_tiles_before": int(collision_stats["n_collided_tiles"]),
            "n_collided_tiles_after": int(collision_stats["n_collided_tiles"]),
            "n_reassigned_tiles": 0,
            "max_stack_before": int(collision_stats["max_stack"]),
            "max_stack_after": int(collision_stats["max_stack"]),
            "unresolved_tiles": int(collision_stats["n_collided_tiles"]),
        }
    gap_collapse_axis_norm = _normalize_gap_collapse_axis_safe(gap_collapse_axis)
    gap_collapse_active = bool(
        gap_collapse_axis_norm is not None and int(max(0, int(gap_collapse_max_run_px))) > 0
    )
    if gap_collapse_active and soft_enabled:
        soft_enabled = False
        soft_fx_seed = None
        soft_fy_seed = None
    if gap_collapse_active:
        cx_seed, cy_seed, w, h, gap_collapse_info = _raster.collapse_empty_center_gaps(
            cx=cx_seed,
            cy=cy_seed,
            w=int(w),
            h=int(h),
            axis=gap_collapse_axis_norm,
            max_gap_run_px=int(gap_collapse_max_run_px),
            logger=None,
            context="cache_embedding_image(legacy)",
        )
    else:
        gap_collapse_info = {
            "applied": False,
            "axis": gap_collapse_axis_norm,
            "max_gap_run_px": int(max(0, int(gap_collapse_max_run_px))),
        }

    def _compute_alpha_from_values(values, arange=(0.1, 1.0), clip=None, invert=False):
        v = np.asarray(values, dtype=float)
        if clip is None:
            vmin, vmax = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
        else:
            vmin, vmax = clip
            if vmin is None:
                vmin = np.nanmin(v)
            if vmax is None:
                vmax = np.nanmax(v)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
        if invert:
            t = 1.0 - t
        a0, a1 = float(arange[0]), float(arange[1])
        a = a0 + t * (a1 - a0)
        return a.astype(np.float32)

    order_idx = np.arange(spatial_coords.shape[0])
    if alpha_by is not None:
        if alpha_by == "embedding":
            alpha_source = embeddings[:, 0]
        else:
            if alpha_by not in adata.obs.columns:
                raise ValueError(f"alpha_by '{alpha_by}' not found in adata.obs.")
            vals_map = adata.obs[alpha_by].astype(float)
            alpha_source = coordinates_df["barcode"].map(vals_map).to_numpy(dtype=float)
        alpha_arr = _compute_alpha_from_values(
            alpha_source, arange=alpha_range, clip=alpha_clip, invert=alpha_invert
        )
        if prioritize_high_values:
            order_idx = np.argsort(alpha_arr)

    num_tiles = spatial_coords.shape[0]
    paint_mask = np.zeros((h, w), dtype=bool)
    if soft_enabled:
        _raster.rasterize_soft_bilinear(
            img=img,
            occupancy_mask=paint_mask,
            seed_values=cols,
            cx=cx_seed,
            cy=cy_seed,
            fx=soft_fx_seed,
            fy=soft_fy_seed,
            chunk_size=max(1, chunk_size),
        )
    else:
        for i in range(0, num_tiles, max(1, chunk_size)):
            idx = order_idx[i : min(i + chunk_size, num_tiles)]
            cx0 = cx_seed[idx]
            cy0 = cy_seed[idx]
            all_x = np.clip((cx0[:, None] + dx[None, :]).ravel(), 0, w - 1)
            all_y = np.clip((cy0[:, None] + dy[None, :]).ravel(), 0, h - 1)
            tile_idx_map = np.repeat(idx, len(dx))
            img[all_y, all_x] = cols[tile_idx_map]
            paint_mask[all_y, all_x] = True

    runtime_fill_info = {
        "applied": False,
        "filled_pixels": 0,
        "support_pixels": int(paint_mask.sum()),
        "closing_radius": int(max(0, int(runtime_fill_closing_radius))),
        "external_radius": int(max(0, int(runtime_fill_external_radius))),
        "fill_holes": bool(runtime_fill_holes),
    }
    fill_active = bool(runtime_fill_from_boundary) or bool(runtime_fill_holes)
    fill_radius = int(runtime_fill_closing_radius) if bool(runtime_fill_from_boundary) else 0
    external_radius = int(runtime_fill_external_radius) if bool(runtime_fill_from_boundary) else 0
    label_paint_mask = paint_mask.copy()
    if fill_active:
        runtime_fill_info = _raster.fill_raster_from_boundary(
            img=img,
            occupancy_mask=paint_mask,
            seed_x=cx_seed,
            seed_y=cy_seed,
            seed_values=cols,
            closing_radius=int(fill_radius),
            external_radius=int(external_radius),
            fill_holes=bool(runtime_fill_holes),
            use_observed_pixels=bool(int(s) == 1),
            persistent_store=getattr(adata, "uns", None),
            logger=None,
            context="cache_embedding_image(legacy)",
        )
    support_mask = np.asarray(paint_mask, dtype=bool)

    label_img = None
    if plot_boundaries and segment_available and segment_series is not None:
        seg_codes = pd.Categorical(coordinates_df["barcode"].map(segment_series)).codes
        label_img = np.full((h, w), -1, dtype=int)
        if soft_enabled:
            value_buf = np.full((h, w), -np.inf, dtype=np.float32)
            for i in range(0, num_tiles, max(1, chunk_size)):
                idx = order_idx[i : min(i + chunk_size, num_tiles)]
                x0, y0, x1, y1, w00, w10, w01, w11 = _raster.bilinear_support_from_centers(
                    cx_seed[idx],
                    cy_seed[idx],
                    soft_fx_seed[idx],
                    soft_fy_seed[idx],
                    w=int(w),
                    h=int(h),
                )
                seg_chunk = seg_codes[idx]
                for px, py, pw in ((x0, y0, w00), (x1, y0, w10), (x0, y1, w01), (x1, y1, w11)):
                    mask_upd = pw > value_buf[py, px]
                    if np.any(mask_upd):
                        label_img[py[mask_upd], px[mask_upd]] = seg_chunk[mask_upd]
                        value_buf[py[mask_upd], px[mask_upd]] = pw[mask_upd]
        else:
            for i in range(0, num_tiles, max(1, chunk_size)):
                idx = order_idx[i : min(i + chunk_size, num_tiles)]
                cx0 = cx_seed[idx]
                cy0 = cy_seed[idx]
                if pixel_shape == "circle":
                    for j, t in enumerate(idx):
                        lx = np.clip(cx0[j] + dx, 0, w - 1)
                        ly = np.clip(cy0[j] + dy, 0, h - 1)
                        label_img[ly, lx] = seg_codes[t]
                else:
                    all_x = np.clip((cx0[:, None] + dx[None, :]).ravel(), 0, w - 1)
                    all_y = np.clip((cy0[:, None] + dy[None, :]).ravel(), 0, h - 1)
                    label_img[all_y, all_x] = np.repeat(seg_codes[idx], len(dx))
        label_img[~label_paint_mask] = -1

    if downsample_factor and downsample_factor > 1.0:
        factor = int(downsample_factor)
        new_h = max(1, h // factor)
        new_w = max(1, w // factor)
        pooled = img[: new_h * factor, : new_w * factor]
        pooled = pooled.reshape(new_h, factor, new_w, factor, dims).mean(axis=(1, 3))
        img = pooled.astype(np.float32)
        h, w = img.shape[:2]
        support_mask = _downsample_support_mask(support_mask, h, w)
        scale /= factor
        s = max(1, int(np.ceil(s_data_units * scale)))
        dpi_effective = float(dpi_in) / float(factor)
    else:
        dpi_effective = float(dpi_in)

    if label_img is not None and downsample_factor and downsample_factor > 1.0:
        factor = int(downsample_factor)
        new_h = max(1, label_img.shape[0] // factor)
        new_w = max(1, label_img.shape[1] // factor)
        label_img = label_img[: new_h * factor, : new_w * factor]
        label_img = label_img.reshape(new_h, factor, new_w, factor).max(axis=(1, 3))
    if label_img is not None and support_mask.shape == label_img.shape:
        label_img = np.asarray(label_img, dtype=np.int32)
        label_img[~support_mask] = -1

    if downsample_factor and downsample_factor > 1.0 and soft_enabled:
        cx, cy, soft_fx, soft_fy = _raster.scale_points_to_canvas_soft(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=True,
            w=int(w),
            h=int(h),
        )
    elif downsample_factor and downsample_factor > 1.0 and collision_info is not None and bool(collision_info.get("applied", False)):
        factor = int(max(1, round(float(downsample_factor))))
        cx = np.clip((cx_seed // factor).astype(np.int32, copy=False), 0, max(0, w - 1))
        cy = np.clip((cy_seed // factor).astype(np.int32, copy=False), 0, max(0, h - 1))
    elif downsample_factor and downsample_factor > 1.0:
        cx, cy = _raster.scale_points_to_canvas(
            spatial_coords,
            x_min_eff=float(x_min_eff),
            y_min_eff=float(y_min_eff),
            scale=float(scale),
            pixel_perfect=bool(pixel_perfect),
            w=int(w),
            h=int(h),
        )
    else:
        cx, cy = cx_seed, cy_seed
        soft_fx, soft_fy = soft_fx_seed, soft_fy_seed

    cache = {
        "img": img,
        "h": int(h),
        "w": int(w),
        "total_pixels": int(h * w),
        "channels": int(dims),
        "coordinate_mode": str(coord_mode),
        "array_row_key": str(array_row_key),
        "array_col_key": str(array_col_key),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "x_min_eff": float(x_min_eff),
        "x_max_eff": float(x_max_eff),
        "y_min_eff": float(y_min_eff),
        "y_max_eff": float(y_max_eff),
        "scale": float(scale),
        "s_data_units": float(s_data_units),
        "tile_px": int(s),
        "pixel_perfect": bool(pixel_perfect),
        "pixel_shape": str(pixel_shape),
        "embedding_key": str(embedding),
        "dimensions": list(dimensions),
        "origin": bool(origin),
        "figsize": None if user_figsize_input is None else tuple(user_figsize_input),
        "fig_dpi": None if user_fig_dpi_input is None else int(user_fig_dpi_input),
        "raster_dpi": int(dpi_in),
        "effective_dpi": float(dpi_effective),
        "downsample_factor": float(downsample_factor),
        "cx": cx.astype(np.int32),
        "cy": cy.astype(np.int32),
        "soft_rasterization": bool(soft_enabled),
        "soft_rasterization_mode": "bilinear" if soft_enabled else None,
        "soft_fx": (soft_fx.astype(np.float16) if soft_enabled else None),
        "soft_fy": (soft_fy.astype(np.float16) if soft_enabled else None),
        "support_mask": support_mask.astype(bool, copy=False),
        "tile_obs_indices": tile_obs_indices.astype(np.int64),
        "barcodes": coordinates_df["barcode"].astype(str).to_list(),
        "origin_flags": coordinates_df.get("origin", pd.Series(index=coordinates_df.index, data=1))
        .astype(int)
        .to_numpy(),
        "segment_key": seg_col,
        "segment_key_requested": segment_key,
        "store": "memory",
        "memmap": False,
        "img_path": None,
        "created_by": "cache_embedding_image_legacy",
        "history": [],
        "center_collision_resolution": (
            dict(collision_info)
            if collision_info is not None
            else {
                "applied": False,
                "enabled": False,
                "search_radius": int(center_collision_radius),
                "ignored_due_to_soft_rasterization": bool(soft_enabled and resolve_center_collisions),
                "n_tiles": int(num_tiles),
                "n_unique_centers_before": None,
                "n_unique_centers_after": None,
                "collision_fraction_before": None,
                "collision_fraction_after": None,
                "n_collided_tiles_before": None,
                "n_collided_tiles_after": None,
                "n_reassigned_tiles": 0,
                "max_stack_before": None,
                "max_stack_after": None,
                "unresolved_tiles": 0,
            }
        ),
        "image_plot_params": {
            "color_by": color_by,
            "palette": palette,
            "alpha_by": alpha_by,
            "alpha_range": tuple(alpha_range) if alpha_range is not None else None,
            "alpha_clip": alpha_clip,
            "alpha_invert": bool(alpha_invert),
            "prioritize_high_values": bool(prioritize_high_values),
            "overlap_priority": overlap_priority,
            "segment_color_by_major": bool(segment_color_by_major),
            "title": title,
            "use_imshow": bool(use_imshow),
            "brighten_continuous": bool(brighten_continuous),
            "continuous_gamma": float(continuous_gamma),
            "brighten_applied": bool(brighten_applied),
            "plot_boundaries": bool(plot_boundaries),
            "boundary_method": boundary_method,
            "boundary_color": boundary_color,
            "boundary_linewidth": float(boundary_linewidth),
            "boundary_style": boundary_style,
            "boundary_alpha": float(boundary_alpha),
            "pixel_smoothing_sigma": float(pixel_smoothing_sigma),
            "highlight_segments": list(highlight_segments)
            if highlight_segments is not None
            else None,
            "highlight_boundary_color": highlight_boundary_color,
            "highlight_linewidth": None
            if highlight_linewidth is None
            else float(highlight_linewidth),
            "highlight_brighten_factor": float(highlight_brighten_factor),
            "dim_other_segments": float(dim_other_segments),
            "dim_to_grey": bool(dim_to_grey),
            "segment_key": seg_col,
            "segment_key_requested": segment_key,
            "runtime_fill_from_boundary": bool(runtime_fill_from_boundary),
            "runtime_fill_closing_radius": int(max(0, int(runtime_fill_closing_radius))),
            "runtime_fill_external_radius": int(max(0, int(runtime_fill_external_radius))),
            "runtime_fill_holes": bool(runtime_fill_holes),
            "gap_collapse_axis": gap_collapse_axis_norm,
            "gap_collapse_max_run_px": int(max(0, int(gap_collapse_max_run_px))),
            "soft_rasterization": bool(soft_enabled),
            "resolve_center_collisions": _raster.normalize_collision_mode(resolve_center_collisions),
            "center_collision_radius": int(center_collision_radius),
            **plot_kwargs,
        },
        "runtime_fill": runtime_fill_info,
        "gap_collapse": dict(gap_collapse_info),
    }

    if plot_boundaries and label_img is not None and boundary_method == "pixel":
        if img.shape[2] >= 3:
            base = img[:, :, :3].copy()
        elif img.shape[2] == 2:
            base = np.stack([img[:, :, 0], img[:, :, 1], np.zeros_like(img[:, :, 0])], axis=2)
        else:
            base = np.repeat(img, 3, axis=2)
        bmask = find_boundaries(label_img, mode="outer", background=-1, connectivity=2)
        if pixel_smoothing_sigma and pixel_smoothing_sigma > 0:
            sm = gaussian_filter(bmask.astype(np.float32), pixel_smoothing_sigma)
            bmask = sm >= 0.5
        lw = max(1, int(round(boundary_linewidth)))
        if lw > 1:
            bmask = binary_dilation(bmask, iterations=lw - 1)
        base_col = np.array(mcolors.to_rgb(boundary_color), dtype=np.float32)
        if highlight_segments is not None and segment_available and segment_series is not None:
            cats = pd.Categorical(segment_series).categories
            codes = [cats.get_loc(seg) for seg in highlight_segments if seg in cats]
            if codes:
                hmask = np.isin(label_img, np.array(codes, dtype=int)) & bmask
            else:
                hmask = np.zeros_like(bmask)
        else:
            hmask = np.zeros_like(bmask)
        if highlight_boundary_color is not None:
            high_col = np.array(mcolors.to_rgb(highlight_boundary_color), dtype=np.float32)
        else:
            high_col = base_col
        out = base.copy()
        a = float(boundary_alpha)
        for c in range(3):
            out[..., c] = np.where(bmask & ~hmask, a * base_col[c] + (1 - a) * out[..., c], out[..., c])
            out[..., c] = np.where(hmask, a * high_col[c] + (1 - a) * out[..., c], out[..., c])
        cache["img_preview"] = np.clip(out, 0.0, 1.0).astype(np.float32)
        cache["label_img"] = label_img.astype(np.int32)

    if show:
        try:
            adata.uns[key] = cache
            show_cached_image(adata, key=key, channels=show_channels, cmap=show_cmap)
        except Exception:
            pass
    return cache




def rebuild_cached_image_from_obsm(
    adata,
    *,
    key: str = "image_plot_slic",
    embedding: Optional[str] = None,
    dimensions: Optional[Sequence[int]] = None,
    in_place: bool = True,
    out_key: Optional[str] = None,
    show: bool = False,
    show_channels: Optional[Sequence[int]] = None,
    show_cmap: Optional[str] = None,
    normalize_channels: bool = True,
):
    """Rebuild a cached image from an embedding in ``adata.obsm``.

    Uses cached geometry (extent, scale, tile size, centers, and optional soft
    rasterization/fill settings) to keep pixel alignment identical. This is used
    after smoothing/equalization of embeddings.
    """
    if key not in adata.uns:
        raise KeyError(f"No cached image found at adata.uns['{key}']")
    cache = adata.uns[key]

    emb_key = embedding or cache.get("embedding_key", "X_embedding")
    dims = list(dimensions) if dimensions is not None else list(cache.get("dimensions", [0, 1, 2]))

    if emb_key not in adata.obsm:
        raise KeyError(f"Embedding '{emb_key}' not found in adata.obsm.")

    # geometry
    h = int(cache["h"])
    w = int(cache["w"])
    s = int(cache["tile_px"])
    pixel_shape = str(cache.get("pixel_shape", "square"))
    cx = np.asarray(cache.get("cx"), dtype=int)
    cy = np.asarray(cache.get("cy"), dtype=int)
    soft_enabled = bool(cache.get("soft_rasterization", False))
    soft_fx = cache.get("soft_fx", None)
    soft_fy = cache.get("soft_fy", None)
    tile_obs_indices = cache.get("tile_obs_indices", None)
    barcodes = list(cache.get("barcodes", []) or [])
    if tile_obs_indices is None and not barcodes:
        raise ValueError("Cached image missing both 'tile_obs_indices' and 'barcodes' order.")
    if tile_obs_indices is not None:
        idx = np.asarray(tile_obs_indices, dtype=int)
        if cx.size != idx.size or cy.size != idx.size:
            raise ValueError("Cached centers length mismatch with tile_obs_indices.")
    else:
        if cx.size != len(barcodes) or cy.size != len(barcodes):
            raise ValueError("Cached centers length mismatch with barcodes.")

    # collect embeddings aligned to cached barcodes order
    if tile_obs_indices is not None:
        rows = np.asarray(tile_obs_indices, dtype=int)
    else:
        obs_idx_map = {str(bc): i for i, bc in enumerate(adata.obs.index.astype(str))}
        rows = np.fromiter((obs_idx_map[str(bc)] for bc in barcodes), dtype=int)
    E = np.asarray(adata.obsm[emb_key])
    # If the embedding only contains the selected dims (e.g., smoothed/equalized output),
    # treat dims as relative indices [0..len(dims)-1]. Otherwise, use absolute dim indices.
    if E.shape[1] == len(dims):
        cols_idx = np.arange(E.shape[1])
    else:
        cols_idx = np.asarray(dims, dtype=int)
    emb = E[rows][:, cols_idx].astype(np.float32)

    # normalize per channel if requested
    if normalize_channels:
        scaler = MinMaxScaler()
        emb = scaler.fit_transform(emb)
    plot_params = cache.get("image_plot_params", {})
    brighten_applied = False
    if plot_params.get("brighten_continuous", False) and emb.shape[1] == 3:
        emb = _apply_brighten_continuous_rgb(emb, plot_params.get("continuous_gamma", 0.8))
        brighten_applied = True

    if soft_enabled:
        if soft_fx is None or soft_fy is None:
            raise ValueError("Soft cache missing soft_fx/soft_fy metadata.")
        soft_fx = np.asarray(soft_fx, dtype=np.float32)
        soft_fy = np.asarray(soft_fy, dtype=np.float32)
        if soft_fx.size != cx.size or soft_fy.size != cy.size:
            raise ValueError("Cached soft raster metadata length mismatch with centers.")

    img = np.ones((h, w, emb.shape[1]), dtype=np.float32)
    paint_mask = np.zeros((h, w), dtype=bool)

    # reconstruct top-left from centers and tile size
    xi = np.clip(cx - s // 2, 0, w - 1)
    yi = np.clip(cy - s // 2, 0, h - 1)
    if pixel_shape == "circle":
        yy, xx = np.ogrid[:s, :s]
        center = (s - 1) / 2
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= (s / 2) ** 2
        dy, dx = np.nonzero(mask)
    else:
        rng = np.arange(s)
        gx, gy = np.meshgrid(rng, rng)
        dx, dy = gx.flatten(), gy.flatten()

    n = int(cx.size)
    pixels_per_tile = int(len(dx))
    if soft_enabled:
        _raster.rasterize_soft_bilinear(
            img=img,
            occupancy_mask=paint_mask,
            seed_values=emb,
            cx=np.asarray(cx, dtype=np.int32),
            cy=np.asarray(cy, dtype=np.int32),
            fx=soft_fx,
            fy=soft_fy,
            chunk_size=max(1, int(os.environ.get("SPIX_CACHE_REBUILD_CHUNK", "50000"))),
        )
    elif pixels_per_tile == 1 and pixel_shape != "circle":
        yy = np.clip(cy, 0, h - 1)
        xx = np.clip(cx, 0, w - 1)
        img[yy, xx] = emb
        paint_mask[yy, xx] = True
    else:
        chunk_n = int(os.environ.get("SPIX_CACHE_REBUILD_CHUNK", "50000"))
        for start in range(0, n, max(1, chunk_n)):
            end = min(start + max(1, chunk_n), n)
            all_x = np.clip((xi[start:end, None] + dx).ravel(), 0, w - 1)
            all_y = np.clip((yi[start:end, None] + dy).ravel(), 0, h - 1)
            tile_idx_map = np.repeat(np.arange(start, end), pixels_per_tile)
            img[all_y, all_x] = emb[tile_idx_map]
            paint_mask[all_y, all_x] = True

    runtime_fill_cfg = dict(cache.get("runtime_fill", {}) or {})
    fill_active = bool(runtime_fill_cfg.get("applied", False)) or bool(plot_params.get("runtime_fill_from_boundary", False)) or bool(plot_params.get("runtime_fill_holes", runtime_fill_cfg.get("fill_holes", False)))
    fill_radius = int(plot_params.get("runtime_fill_closing_radius", runtime_fill_cfg.get("closing_radius", 1))) if bool(plot_params.get("runtime_fill_from_boundary", False)) else 0
    external_radius = int(plot_params.get("runtime_fill_external_radius", runtime_fill_cfg.get("external_radius", 0))) if bool(plot_params.get("runtime_fill_from_boundary", False)) else 0
    if fill_active:
        _raster.fill_raster_from_boundary(
            img=img,
            occupancy_mask=paint_mask,
            seed_x=np.asarray(cx, dtype=np.int32),
            seed_y=np.asarray(cy, dtype=np.int32),
            seed_values=emb,
            closing_radius=int(fill_radius),
            external_radius=int(external_radius),
            fill_holes=bool(plot_params.get("runtime_fill_holes", runtime_fill_cfg.get("fill_holes", True))),
            use_observed_pixels=bool(int(s) == 1),
            persistent_store=getattr(adata, "uns", None),
            logger=None,
            context="rebuild_cached_image_from_obsm",
        )

    out_cache = cache if in_place else dict(cache)
    out_cache["img"] = img
    out_cache["embedding_key"] = emb_key
    out_cache["dimensions"] = list(dims)
    out_cache["total_pixels"] = int(h * w)
    history = out_cache.get("history", [])
    if isinstance(history, np.ndarray):
        history = history.tolist()
    if not isinstance(history, list):
        history = [history]
    history.append({
        "op": "rebuild_image",
        "embedding": emb_key,
        "dims": list(dims),
    })
    out_cache["history"] = history
    out_plot_params = dict(out_cache.get("image_plot_params", {}) or {})
    out_plot_params["brighten_applied"] = bool(brighten_applied)
    out_cache["image_plot_params"] = out_plot_params
    out_cache.pop("img_preview", None)
    img_path = out_cache.get("img_path", None)
    cache_storage = out_cache.get("cache_storage", "memory")
    if img_path and cache_storage != "memory":
        _write_cache_image_to_disk(img, img_path, str(cache_storage))
        if str(cache_storage) != "float32_memmap":
            out_cache["img"] = None

    if not in_place and out_key:
        adata.uns[out_key] = out_cache

    if show:
        try:
            show_cached_image(adata, key=out_key if (not in_place and out_key) else key, channels=show_channels, cmap=show_cmap)
        except Exception:
            pass
    return out_cache


def _prepare_display_image(img: np.ndarray, channels: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return an RGB image from an arbitrary C-channel array for display.

    - If channels is provided, select those channels (expects length 1–3).
    - If C > 3 and no channels given: use first 3 channels.
    - If C == 2: pad with zeros to make 3.
    - If C == 1: repeat to 3.
    Values are clipped to [0, 1] for imshow.
    """
    h, w, c = img.shape
    if channels is not None:
        sel = list(channels)
        if len(sel) == 1:
            rgb = np.repeat(img[:, :, sel[0:1]], 3, axis=2)
        elif len(sel) == 2:
            a = img[:, :, sel[0]]
            b = img[:, :, sel[1]]
            rgb = np.stack([a, b, np.zeros_like(a)], axis=2)
        else:
            rgb = img[:, :, sel[:3]]
    else:
        if c >= 3:
            rgb = img[:, :, :3]
        elif c == 2:
            rgb = np.stack([img[:, :, 0], img[:, :, 1], np.zeros_like(img[:, :, 0])], axis=2)
        else:
            rgb = np.repeat(img, 3, axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def _apply_brighten_continuous_rgb(rgb: np.ndarray, continuous_gamma: float) -> np.ndarray:
    """Apply image_plot-style brightening/gamma to RGB arrays."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.shape[-1] != 3:
        return arr
    max_per_pixel = arr.max(axis=-1, keepdims=True)
    scale = np.where(max_per_pixel > 0, 1.0 / max_per_pixel, 0.0)
    arr = np.clip(arr * scale, 0.0, 1.0)
    try:
        gamma = float(continuous_gamma)
        if gamma > 0 and gamma != 1.0:
            arr = np.clip(np.power(arr, gamma), 0.0, 1.0)
    except Exception:
        pass
    return arr


def _auto_title_fontsize_from_figsize(figsize: Tuple[float, float]) -> int:
    """Choose a title fontsize that scales with figure size.

    Uses sqrt(area) as a simple proxy. Clamped to a sensible range.
    """
    try:
        w, h = float(figsize[0]), float(figsize[1])
    except Exception:
        w, h = 8.0, 8.0
    size_factor = (max(w, 0.1) * max(h, 0.1)) ** 0.5
    base = 12.0
    fs = base + (size_factor - 8.0) * 1.2
    return int(np.clip(round(fs), 8, 32))


def list_cached_images(adata) -> List[str]:
    """Return keys in ``adata.uns`` that look like cached embedding images."""
    keys: List[str] = []
    if not hasattr(adata, "uns"):
        return keys
    for k, v in adata.uns.items():
        try:
            if isinstance(v, dict) and (
                (isinstance(v.get("img", None), np.ndarray) and v["img"].ndim == 3)
                or (v.get("img_path", None) is not None and int(v.get("channels", 0)) > 0)
            ):
                keys.append(k)
        except Exception:
            continue
    return keys


def summarize_cached_overlap(
    adata,
    key: str = "image_plot_slic",
) -> Dict[str, Any]:
    """Summarize tile overlap/collision statistics for a cached raster image."""
    if key not in adata.uns:
        raise KeyError(f"No cached image under adata.uns['{key}']")
    cache = adata.uns[key]
    img_shape = get_cached_image_shape(cache)
    support_mask = np.asarray(cache.get("support_mask"), dtype=bool)
    cx = np.asarray(cache.get("cx"), dtype=np.int32).ravel()
    cy = np.asarray(cache.get("cy"), dtype=np.int32).ravel()
    n_tiles = int(min(cx.size, cy.size))
    if n_tiles <= 0:
        raise ValueError("Cached image does not contain cx/cy tile centers.")
    flat = cy.astype(np.int64, copy=False) * np.int64(max(1, img_shape[1])) + cx.astype(np.int64, copy=False)
    unique, counts = np.unique(flat, return_counts=True)
    unique_centers = int(unique.size)
    collided_tiles = int(n_tiles - unique_centers)
    support_pixels = int(np.count_nonzero(support_mask))
    canvas_pixels = int(support_mask.size)
    out = {
        "key": str(key),
        "shape": tuple(int(x) for x in img_shape),
        "n_tiles": int(n_tiles),
        "unique_centers": int(unique_centers),
        "collided_tiles": int(collided_tiles),
        "collision_fraction": float(collided_tiles / max(1, n_tiles)),
        "max_stack": int(counts.max()) if counts.size else 1,
        "support_pixels": int(support_pixels),
        "canvas_pixels": int(canvas_pixels),
        "support_fraction": float(support_pixels / max(1, canvas_pixels)),
        "soft_rasterization": bool(cache.get("soft_rasterization", False)),
        "tile_px": int(cache.get("tile_px", 1)),
        "scale": float(cache.get("scale", 1.0)),
    }
    return out


def show_cached_image(
    adata,
    key: str = "image_plot_slic",
    *,
    channels: Optional[Sequence[int]] = None,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    fig_dpi: Optional[int] = None,
    prefer_preview: bool = True,
    brighten_continuous: Optional[bool] = None,
    continuous_gamma: Optional[float] = None,
):
    """Display a single cached image stored in ``adata.uns[key]``.

    - For multi-channel images, defaults to RGB display. You can select
      specific channels via ``channels=[i,j,k]``.
    - For 1-channel images, uses ``cmap`` if provided; otherwise converts to RGB.
    - ``brighten_continuous``/``continuous_gamma`` can brighten RGB display;
      defaults to cached ``image_plot_params`` when present.
    """
    if key not in adata.uns:
        raise KeyError(f"No cached image under adata.uns['{key}']")
    cache = adata.uns[key]
    use_preview = bool(prefer_preview and ("img_preview" in cache) and channels is None)
    # Prefer preview with boundaries if available only for default RGB display.
    if use_preview:
        img = np.asarray(cache.get("img_preview"))
        display_channels = None
    else:
        img = np.asarray(get_cached_image_channels(cache, channels=channels))
        display_channels = None if channels is not None else channels
    if img.ndim != 3:
        raise ValueError("Cached object does not contain a 3D image array.")

    figsize, fig_dpi = _resolve_cached_preview_figsize_and_dpi(
        cache,
        img.shape,
        figsize=figsize,
        fig_dpi=fig_dpi,
    )

    plot_params = cache.get("image_plot_params", {}) or {}
    apply_brighten = (
        bool(brighten_continuous)
        if brighten_continuous is not None
        else bool(plot_params.get("brighten_continuous", False))
    )
    gamma = (
        float(continuous_gamma)
        if continuous_gamma is not None
        else float(plot_params.get("continuous_gamma", 0.8))
    )
    cache_channels = int(cache.get("channels", img.shape[2]))
    already_applied = bool(plot_params.get("brighten_applied", False)) and cache_channels == 3

    plt.figure(figsize=figsize, dpi=fig_dpi)
    if img.shape[2] == 1 and cmap is not None and display_channels is None:
        plt.imshow(img[:, :, 0], origin="lower", cmap=cmap, interpolation="nearest")
    else:
        rgb = _prepare_display_image(img, channels=display_channels)
        if apply_brighten and not already_applied:
            rgb = _apply_brighten_continuous_rgb(rgb, gamma)
        plt.imshow(rgb, origin="lower", interpolation="nearest")

    if title is None:
        emb = cache.get("embedding_key", "?")
        dims = cache.get("dimensions", [])
        # resolve used channels for display title
        if channels is not None and len(channels) > 0:
            used_ch = list(channels)
        else:
            c = cache_channels
            if c >= 3:
                used_ch = [0, 1, 2]
            elif c == 2:
                used_ch = [0, 1]
            else:
                used_ch = [0]
        raster_dpi = cache.get("raster_dpi", cache.get("effective_dpi", None))
        total_px = cache.get("total_pixels", img.shape[0] * img.shape[1])
        title = (
            f"{key} | {img.shape[1]}x{img.shape[0]} C={cache_channels} "
            f"| emb={emb} "
            f"| ch=[{','.join(map(str,used_ch))}] "
            f"| dpi={raster_dpi} | px={total_px}"
        )
    fs = _auto_title_fontsize_from_figsize(figsize)
    plt.title(title, fontsize=fs)
    plt.axis("off")
    plt.show()


def show_all_cached_images(
    adata,
    keys: Optional[Sequence[str]] = None,
    *,
    max_cols: int = 3,
    channels: Optional[Sequence[int]] = None,
    cmap: Optional[str] = None,
    tight_layout: bool = True,
    prefer_preview: bool = True,
    brighten_continuous: Optional[bool] = None,
    continuous_gamma: Optional[float] = None,
):
    """Display all cached images (or a provided subset of keys) as a grid."""
    if keys is None:
        keys = list_cached_images(adata)
    keys = list(keys)
    if len(keys) == 0:
        print("No cached images found in adata.uns.")
        return

    n = len(keys)
    cols = max(1, min(max_cols, n))
    rows = int(np.ceil(n / cols))
    # auto figure size: 4x4 per tile
    fig_w = 4 * cols
    fig_h = 4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    for i, k in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        cache = adata.uns[k]
        use_preview = bool(prefer_preview and ("img_preview" in cache) and channels is None)
        if use_preview:
            img = np.asarray(cache.get("img_preview"))
            display_channels = None
        else:
            img = np.asarray(get_cached_image_channels(cache, channels=channels))
            display_channels = None if channels is not None else channels
        if img.ndim != 3:
            ax.set_visible(False)
            continue
        plot_params = cache.get("image_plot_params", {}) or {}
        apply_brighten = (
            bool(brighten_continuous)
            if brighten_continuous is not None
            else bool(plot_params.get("brighten_continuous", False))
        )
        gamma = (
            float(continuous_gamma)
            if continuous_gamma is not None
            else float(plot_params.get("continuous_gamma", 0.8))
        )
        cache_channels = int(cache.get("channels", img.shape[2]))
        already_applied = bool(plot_params.get("brighten_applied", False)) and cache_channels == 3
        if img.shape[2] == 1 and cmap is not None and display_channels is None:
            ax.imshow(img[:, :, 0], origin="lower", cmap=cmap, interpolation="nearest")
        else:
            rgb = _prepare_display_image(img, channels=display_channels)
            if apply_brighten and not already_applied:
                rgb = _apply_brighten_continuous_rgb(rgb, gamma)
            ax.imshow(rgb, origin="lower", interpolation="nearest")
        emb = cache.get("embedding_key", "?")
        dims = cache.get("dimensions", [])
        raster_dpi = cache.get("raster_dpi", cache.get("effective_dpi", None))
        total_px = cache.get("total_pixels", img.shape[0] * img.shape[1])
        cache_channels = int(cache.get("channels", img.shape[2]))
        tile_fs = _auto_title_fontsize_from_figsize((fig_w / cols, fig_h / rows))
        ax.set_title(
            f"{k}\n{img.shape[1]}x{img.shape[0]} C={cache_channels} | emb={emb} | dpi={raster_dpi} | px={total_px}",
            fontsize=tile_fs,
        )
        ax.axis("off")

    # Hide any extra axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].set_visible(False)

    if tight_layout:
        plt.tight_layout()
    plt.show()
