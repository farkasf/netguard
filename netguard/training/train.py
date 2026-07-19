"""Training CLI and the reusable training/evaluation routines.

scikit-learn is imported **here** (training/evaluation code) only for metrics —
``precision_recall_fscore_support`` and ``confusion_matrix``. It is never
imported by anything under ``netguard/ml/`` (enforced by the purity test).

Usage:
    python -m netguard.training.train --synthetic --out models/ --register
    python -m netguard.training.train --csv data.csv --evaluate
    python -m netguard.training.train --cic data/cic.csv --out models/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from netguard.config import get_settings
from netguard.features.extractor import FEATURE_NAMES
from netguard.ml.encoders import LabelEncoder, StandardScaler
from netguard.ml.gbm import GradientBoostingClassifier
from netguard.ml.persistence import NetGuardModel, make_version
from netguard.training import dataset as ds


def evaluate(model: NetGuardModel, X: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    """Compute a metrics report using sklearn (allowed in evaluation code)."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    classes = model.classes
    y_pred = np.asarray(model.predict_labels(X), dtype=object)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    accuracy = float(np.mean(y_pred == y_true))
    return {
        "classes": list(classes),
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(classes)
        },
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
    }


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    gbm_kwargs: dict[str, Any] | None = None,
    version: str | None = None,
) -> NetGuardModel:
    """Fit scaler + label encoder + GBM and wrap them in a NetGuardModel."""
    settings = get_settings()
    feature_names = feature_names or FEATURE_NAMES
    if gbm_kwargs is None:
        gbm_kwargs = settings.gbm.model_dump()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    le = LabelEncoder()
    yi = le.fit_transform(y)

    gbm = GradientBoostingClassifier(**gbm_kwargs)
    gbm.fit(Xs, yi)

    return NetGuardModel(
        gbm=gbm,
        scaler=scaler,
        label_encoder=le,
        feature_names=feature_names,
        version=version or make_version(),
    )


def _load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.synthetic:
        return ds.make_synthetic(n_per_class=args.n_per_class, random_state=42)
    if args.csv:
        return ds.load_csv(args.csv)
    if args.cic:
        return ds.load_cic_ids2017(args.cic, label_col=args.label_col)
    if args.pcap:
        return ds.load_pcap(args.pcap, label=args.pcap_label)
    raise SystemExit("Provide one of --synthetic / --csv / --cic / --pcap")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train a NetGuard GBM model.")
    src = parser.add_argument_group("data source")
    src.add_argument("--synthetic", action="store_true", help="use built-in synthetic data")
    src.add_argument("--csv", help="generic labeled CSV (last column = label)")
    src.add_argument("--cic", help="CIC-IDS2017 CSV")
    src.add_argument("--pcap", help="pcap file (single label)")
    parser.add_argument("--label-col", default="label", help="CIC label column name")
    parser.add_argument("--pcap-label", default="BENIGN", help="label for --pcap flows")
    parser.add_argument("--n-per-class", type=int, default=300, help="synthetic samples/class")
    parser.add_argument("--out", default=str(settings.models_dir), help="output models dir")
    parser.add_argument("--evaluate", action="store_true", help="print held-out metrics report")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--register", action="store_true", help="register + activate in the DB")
    args = parser.parse_args(argv)

    X, y = _load_dataset(args)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, "
          f"classes={sorted(set(y.tolist()))}")

    report: dict[str, Any] | None = None
    if args.evaluate:
        X_tr, X_te, y_tr, y_te = ds.train_test_split(X, y, test_size=args.test_size)
        model = train_model(X_tr, y_tr)
        report = evaluate(model, X_te, y_te)
        model.macro_f1 = report["macro_f1"]
        print(_format_report(report))
    else:
        model = train_model(X, y)
        # Self-eval (train) just to populate macro_f1 on the artifact.
        report = evaluate(model, X, y)
        model.macro_f1 = report["macro_f1"]

    out_dir = Path(args.out) / model.version
    model.save(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Saved model {model.version} -> {out_dir} (macro_f1={model.macro_f1:.4f})")

    if args.register:
        from netguard.store.repository import Repository

        repo = Repository()
        repo.register_model(
            version=model.version,
            path=str(out_dir),
            macro_f1=model.macro_f1,
            metrics=report,
            activate=True,
        )
        repo.close()
        print(f"Registered and activated {model.version}")

    return 0


def _format_report(report: dict[str, Any]) -> str:
    lines = ["", "=== Evaluation report ===",
             f"accuracy      : {report['accuracy']:.4f}",
             f"macro F1      : {report['macro_f1']:.4f}",
             f"macro precision: {report['macro_precision']:.4f}",
             f"macro recall  : {report['macro_recall']:.4f}",
             "", "per-class  precision  recall      f1  support"]
    for cls, m in report["per_class"].items():
        lines.append(
            f"  {cls:<10} {m['precision']:.3f}    {m['recall']:.3f}   {m['f1']:.3f}  {m['support']:>6}"
        )
    lines.append("")
    lines.append("confusion matrix (rows=true, cols=pred):")
    lines.append("  " + " ".join(f"{c[:8]:>9}" for c in report["classes"]))
    for cls, row in zip(report["classes"], report["confusion_matrix"], strict=False):
        lines.append(f"  {cls[:8]:<9}" + " ".join(f"{v:>9}" for v in row))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
