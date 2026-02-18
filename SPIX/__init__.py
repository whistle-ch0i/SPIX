# SPIX/__init__.py
from importlib import import_module

_SUBMODULE_ALIASES = {
    "tm": "tiles_embeddings",
    "pl": "visualization",
    "ip": "image_processing",
    "sp": "superpixel",
    "op": "optimization",
    "an": "analysis",
    "utils": "utils",
}

__all__ = list(_SUBMODULE_ALIASES)


def __getattr__(name):
    module_name = _SUBMODULE_ALIASES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
