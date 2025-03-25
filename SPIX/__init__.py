# SPIX/__init__.py

from . import tiles_embeddings as tm
from . import visualization as pl
from . import image_processing as ip
from . import superpixel as sp
from . import optimization as op
from . import analysis as an
from . import utils

__all__ = [
    'tm',
    'pl',
    'ip',
    'sp',
    'op',
    'an',
    'utils'
]