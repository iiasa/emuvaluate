"""
emuvaluate — evaluate climate emulators against the simulations they emulate.

Four layers, in the order you use them:

    data_preparation   load raw CMIP6 scenario files into ensemble arrays
    preprocessing      turn raw ensembles into scoring-ready arrays
    metrics            compute every error value, packaged as ErrorData
    plots              render ErrorData as maps, bars and ranking grids

`baseline_methods` supplies the Pattern Scaling reference emulator to
compare against, and `transforms` holds the stateless array operations the
other modules build on.
"""

from . import (
    baseline_methods,
    data_preparation,
    metrics,
    plots,
    preprocessing,
    transforms,
)

__all__ = [
    "baseline_methods",
    "data_preparation",
    "metrics",
    "plots",
    "preprocessing",
    "transforms",
]
