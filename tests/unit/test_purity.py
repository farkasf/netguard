"""Model-purity guard: netguard.ml must not pull in any ML framework.

Importing every module under ``netguard.ml`` must not bring sklearn / xgboost /
lightgbm / catboost / torch / tensorflow into ``sys.modules``.
"""

from __future__ import annotations

import importlib
import sys

ML_MODULES = [
    "netguard.ml.tree",
    "netguard.ml.losses",
    "netguard.ml.gbm",
    "netguard.ml.encoders",
    "netguard.ml.persistence",
]

FORBIDDEN = ["sklearn", "xgboost", "lightgbm", "catboost", "torch", "tensorflow"]


def test_ml_engine_imports_no_frameworks():
    # Drop any previously-imported forbidden modules so this test is meaningful
    # regardless of what earlier tests imported.
    for name in list(sys.modules):
        if any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith("netguard.ml"):
            del sys.modules[name]

    for mod in ML_MODULES:
        importlib.import_module(mod)

    for forbidden in FORBIDDEN:
        assert forbidden not in sys.modules, f"{forbidden} leaked into netguard.ml import graph"


def test_only_numpy_is_the_numeric_dep():
    # tree/losses/gbm/encoders should rely on numpy only at import time.
    import netguard.ml.gbm as gbm

    assert "numpy" in sys.modules
    # Sanity: the GBM module references numpy.
    assert hasattr(gbm, "np")
