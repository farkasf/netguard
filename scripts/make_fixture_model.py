#!/usr/bin/env python3
"""Train a tiny, committable fixture model from the fixture pcaps.

Extracts flows from ``benign_sample.pcap`` (label BENIGN) and
``attack_sample.pcap`` (label PORTSCAN), fits a small GBM, and saves it to
``data/fixtures/fixture_model/``. Integration tests load this model so the
end-to-end pcap-replay path is exercised without training in the test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netguard.features.extractor import FEATURE_NAMES, extract
from netguard.training.dataset import flows_from_pcap
from netguard.training.train import evaluate, train_model

REPO_ROOT = Path(__file__).resolve().parent.parent
FIX = REPO_ROOT / "data" / "fixtures"


def main() -> int:
    benign = flows_from_pcap(FIX / "benign_sample.pcap")
    attack = flows_from_pcap(FIX / "attack_sample.pcap")
    if not benign or not attack:
        raise SystemExit("Run scripts/make_fixture_pcap.py first.")

    X = np.vstack([extract(fr) for fr in benign + attack])
    y = np.asarray(["BENIGN"] * len(benign) + ["PORTSCAN"] * len(attack), dtype=object)

    # Small, fast model — the classes are highly separable.
    model = train_model(
        X, y,
        feature_names=FEATURE_NAMES,
        gbm_kwargs=dict(n_estimators=40, learning_rate=0.3, max_depth=3,
                        min_samples_leaf=1, min_child_weight=1e-3, reg_lambda=1.0,
                        subsample=1.0, colsample=1.0, random_state=0),
        version="vfixture",
    )
    report = evaluate(model, X, y)
    model.macro_f1 = report["macro_f1"]

    out = FIX / "fixture_model"
    model.save(out)
    import json
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Fixture model saved -> {out} (train macro_f1={model.macro_f1:.4f})")
    print(f"  benign flows={len(benign)} attack flows={len(attack)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
