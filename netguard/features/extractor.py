"""Feature extraction: FlowRecord -> fixed-length float64 vector.

The feature order is **frozen** in :data:`FEATURE_NAMES` and defined in exactly
one place so that training and inference can never drift. Every division is
guarded against zero and NaN/Inf are scrubbed to 0.0.

The chosen features deliberately align with CIC-IDS2017 column semantics so its
labeled CSVs can bootstrap training (see ``netguard/training/dataset.py``).
"""

from __future__ import annotations

import numpy as np

from netguard.capture.flow_assembler import FlowRecord

FEATURE_NAMES: list[str] = [
    "duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_pkt_size_mean",
    "fwd_pkt_size_std",
    "fwd_pkt_size_min",
    "fwd_pkt_size_max",
    "bwd_pkt_size_mean",
    "bwd_pkt_size_std",
    "bwd_pkt_size_min",
    "bwd_pkt_size_max",
    "flow_bytes_per_s",
    "flow_packets_per_s",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_mean",
    "bwd_iat_std",
    "down_up_ratio",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "avg_packet_size",
]

N_FEATURES = len(FEATURE_NAMES)


def _stats(values: list[int]) -> tuple[float, float, float, float]:
    """mean, std, min, max of a list; all zero when empty.

    Non-finite inputs are dropped first so intermediate reductions (e.g. std on
    an ``inf``) don't emit numpy warnings; the final vector is scrubbed anyway.
    """
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def _iat_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def extract(record: FlowRecord) -> np.ndarray:
    """Return the frozen-order feature vector for a closed flow."""
    duration = max(record.last_ts - record.start_ts, 0.0)
    total_packets = record.total_packets
    total_bytes = record.fwd_bytes + record.bwd_bytes

    fwd_mean, fwd_std, fwd_min, fwd_max = _stats(record.fwd_pkt_sizes)
    bwd_mean, bwd_std, bwd_min, bwd_max = _stats(record.bwd_pkt_sizes)
    fwd_iat_mean, fwd_iat_std = _iat_stats(record.fwd_iats)
    bwd_iat_mean, bwd_iat_std = _iat_stats(record.bwd_iats)

    denom_dur = duration if duration > 0 else 1.0
    flow_bytes_per_s = total_bytes / denom_dur
    flow_packets_per_s = total_packets / denom_dur
    down_up_ratio = record.bwd_bytes / max(record.fwd_bytes, 1)
    avg_packet_size = total_bytes / max(total_packets, 1)

    values = [
        duration,
        float(record.fwd_packets),
        float(record.bwd_packets),
        float(record.fwd_bytes),
        float(record.bwd_bytes),
        fwd_mean,
        fwd_std,
        fwd_min,
        fwd_max,
        bwd_mean,
        bwd_std,
        bwd_min,
        bwd_max,
        flow_bytes_per_s,
        flow_packets_per_s,
        fwd_iat_mean,
        fwd_iat_std,
        bwd_iat_mean,
        bwd_iat_std,
        down_up_ratio,
        float(record.flags.get("SYN", 0)),
        float(record.flags.get("ACK", 0)),
        float(record.flags.get("FIN", 0)),
        float(record.flags.get("RST", 0)),
        float(record.flags.get("PSH", 0)),
        avg_packet_size,
    ]
    vec = np.asarray(values, dtype=np.float64)
    # Scrub NaN/Inf defensively (e.g. from upstream bad data).
    np.nan_to_num(vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return vec
