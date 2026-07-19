"""Dataset loading and splitting. NumPy/stdlib only — no sklearn.

Three paths to ``(X, y)``:
  * :func:`load_csv` — generic labeled CSV where the last column is the label
    and the rest are numeric features (used for synthetic / exported datasets).
  * :func:`load_cic_ids2017` — maps CIC-IDS2017 column names onto our frozen
    :data:`FEATURE_NAMES`, so the public CSVs bootstrap training.
  * :func:`load_pcap` — runs a pcap through the assembler + extractor; labels
    are taken from a provided mapping or a single ``label`` for the whole file.

:func:`train_test_split` is a deterministic, stratified-ish split using a seeded
NumPy permutation (no sklearn).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from netguard.capture.flow_assembler import FlowAssembler, FlowRecord
from netguard.capture.source import PcapPacketSource
from netguard.features.extractor import FEATURE_NAMES, N_FEATURES, extract

# CIC-IDS2017 column name -> our feature name. The public CSVs use these
# (sometimes space-prefixed) headers; we strip/normalize before matching.
CIC_COLUMN_MAP: dict[str, str] = {
    "flow duration": "duration",
    "total fwd packets": "total_fwd_packets",
    "total backward packets": "total_bwd_packets",
    "total length of fwd packets": "total_fwd_bytes",
    "total length of bwd packets": "total_bwd_bytes",
    "fwd packet length mean": "fwd_pkt_size_mean",
    "fwd packet length std": "fwd_pkt_size_std",
    "fwd packet length min": "fwd_pkt_size_min",
    "fwd packet length max": "fwd_pkt_size_max",
    "bwd packet length mean": "bwd_pkt_size_mean",
    "bwd packet length std": "bwd_pkt_size_std",
    "bwd packet length min": "bwd_pkt_size_min",
    "bwd packet length max": "bwd_pkt_size_max",
    "flow bytes/s": "flow_bytes_per_s",
    "flow packets/s": "flow_packets_per_s",
    "fwd iat mean": "fwd_iat_mean",
    "fwd iat std": "fwd_iat_std",
    "bwd iat mean": "bwd_iat_mean",
    "bwd iat std": "bwd_iat_std",
    "down/up ratio": "down_up_ratio",
    "syn flag count": "syn_count",
    "ack flag count": "ack_count",
    "fin flag count": "fin_count",
    "rst flag count": "rst_count",
    "psh flag count": "psh_count",
    "average packet size": "avg_packet_size",
}


def _clean_float(s: str) -> float:
    try:
        v = float(s)
    except (ValueError, TypeError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a generic CSV: all columns but the last are features; last is label."""
    rows: list[list[float]] = []
    labels: list[str] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # noqa: F841 (header skipped)
        for row in reader:
            if not row:
                continue
            labels.append(str(row[-1]).strip())
            rows.append([_clean_float(c) for c in row[:-1]])
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=object)
    return X, y


def load_cic_ids2017(path: str | Path, label_col: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """Load a CIC-IDS2017 CSV, projecting onto the frozen feature order.

    Missing columns are filled with 0.0 so partial exports still load. The label
    column (default ``Label``) supplies ``y``.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        norm_fields = {(fn or "").strip().lower(): fn for fn in (reader.fieldnames or [])}
        label_field = norm_fields.get(label_col.lower())
        # Resolve each frozen feature to a source column (or None -> zeros).
        feat_source: list[str | None] = []
        inv = {v: k for k, v in CIC_COLUMN_MAP.items()}
        for fname in FEATURE_NAMES:
            cic_name = inv.get(fname)
            src = norm_fields.get(cic_name) if cic_name else None
            feat_source.append(src)

        rows: list[list[float]] = []
        labels: list[str] = []
        for row in reader:
            vec = [
                _clean_float(row[src]) if src is not None else 0.0
                for src in feat_source
            ]
            rows.append(vec)
            labels.append(str(row[label_field]).strip() if label_field else "BENIGN")
    X = np.asarray(rows, dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(labels, dtype=object)
    return X, y


def flows_from_pcap(path: str | Path) -> list[FlowRecord]:
    """Replay a pcap through the assembler and return all closed flows."""
    closed: list[FlowRecord] = []
    assembler = FlowAssembler(on_close=closed.append)
    for pkt in PcapPacketSource(str(path)).packets():
        assembler.add_packet(pkt)
    assembler.flush_all()
    return closed


def load_pcap(path: str | Path, label: str = "BENIGN") -> tuple[np.ndarray, np.ndarray]:
    """Extract features from every flow in a pcap, all tagged with ``label``."""
    flows = flows_from_pcap(path)
    if not flows:
        return np.empty((0, N_FEATURES)), np.empty((0,), dtype=object)
    X = np.vstack([extract(fr) for fr in flows])
    y = np.asarray([label] * len(flows), dtype=object)
    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stratified split (per-class proportional holdout)."""
    rng = np.random.default_rng(random_state)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size))) if len(idx) > 1 else 0
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx_arr = np.array(sorted(train_idx))
    test_idx_arr = np.array(sorted(test_idx))
    return X[train_idx_arr], X[test_idx_arr], y[train_idx_arr], y[test_idx_arr]


def make_synthetic(
    n_per_class: int = 200, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a separable 3-class synthetic dataset in feature space.

    Classes: BENIGN, PORTSCAN (many SYN, tiny packets), DOS (huge byte rates).
    Used by the demo trainer and tests so the project is runnable without
    downloading CIC-IDS2017.
    """
    rng = np.random.default_rng(random_state)
    d = N_FEATURES
    blocks = []
    labels = []

    # BENIGN: moderate everything.
    b = rng.normal(loc=1.0, scale=0.3, size=(n_per_class, d))
    b[:, FEATURE_NAMES.index("syn_count")] = rng.poisson(1, n_per_class)
    blocks.append(b)
    labels += ["BENIGN"] * n_per_class

    # PORTSCAN: high SYN count, small packets, short duration.
    p = rng.normal(loc=0.5, scale=0.2, size=(n_per_class, d))
    p[:, FEATURE_NAMES.index("syn_count")] = rng.poisson(20, n_per_class) + 10
    p[:, FEATURE_NAMES.index("fwd_pkt_size_mean")] = rng.normal(40, 5, n_per_class)
    p[:, FEATURE_NAMES.index("duration")] = rng.normal(0.05, 0.01, n_per_class)
    blocks.append(p)
    labels += ["PORTSCAN"] * n_per_class

    # DOS: very high byte/packet rates.
    o = rng.normal(loc=1.0, scale=0.3, size=(n_per_class, d))
    o[:, FEATURE_NAMES.index("flow_bytes_per_s")] = rng.normal(1e6, 1e5, n_per_class)
    o[:, FEATURE_NAMES.index("flow_packets_per_s")] = rng.normal(5000, 500, n_per_class)
    o[:, FEATURE_NAMES.index("total_fwd_packets")] = rng.normal(2000, 200, n_per_class)
    blocks.append(o)
    labels += ["DOS"] * n_per_class

    X = np.vstack(blocks)
    np.nan_to_num(X, copy=False)
    y = np.asarray(labels, dtype=object)
    return X, y
