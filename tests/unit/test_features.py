"""Unit tests for feature extraction."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from netguard.capture.flow_assembler import FlowKey, FlowRecord
from netguard.features.extractor import FEATURE_NAMES, N_FEATURES, extract


def _record(**kw) -> FlowRecord:
    key = FlowKey("10.0.0.1", 5000, "10.0.0.2", 80, "TCP")
    rec = FlowRecord(key=key, start_ts=0.0, last_ts=0.0, init_ip="10.0.0.1", init_port=5000)
    for k, v in kw.items():
        setattr(rec, k, v)
    return rec


def test_feature_vector_shape_and_order():
    assert len(FEATURE_NAMES) == N_FEATURES
    rec = _record()
    vec = extract(rec)
    assert vec.shape == (N_FEATURES,)
    assert vec.dtype == np.float64


def test_hand_computed_values():
    rec = _record(
        last_ts=2.0,
        fwd_packets=3, bwd_packets=2,
        fwd_bytes=300, bwd_bytes=100,
        fwd_pkt_sizes=[100, 100, 100],
        bwd_pkt_sizes=[40, 60],
        fwd_iats=[0.5, 0.5],
        bwd_iats=[1.0],
        flags=Counter({"SYN": 1, "ACK": 4, "PSH": 2}),
    )
    vec = extract(rec)
    f = dict(zip(FEATURE_NAMES, vec, strict=True))
    assert f["duration"] == 2.0
    assert f["total_fwd_packets"] == 3
    assert f["total_bwd_packets"] == 2
    assert f["total_fwd_bytes"] == 300
    assert f["fwd_pkt_size_mean"] == 100.0
    assert f["fwd_pkt_size_std"] == 0.0
    assert f["bwd_pkt_size_mean"] == 50.0
    assert f["flow_bytes_per_s"] == (400 / 2.0)
    assert f["flow_packets_per_s"] == (5 / 2.0)
    assert math.isclose(f["down_up_ratio"], 100 / 300)
    assert f["syn_count"] == 1
    assert f["ack_count"] == 4
    assert f["psh_count"] == 2
    assert f["avg_packet_size"] == 400 / 5


def test_zero_division_guards():
    # Empty flow: zero duration, zero packets -> no NaN/Inf, all finite.
    rec = _record(last_ts=0.0, fwd_bytes=0, bwd_bytes=0)
    vec = extract(rec)
    assert np.all(np.isfinite(vec))
    f = dict(zip(FEATURE_NAMES, vec, strict=True))
    assert f["flow_bytes_per_s"] == 0.0
    assert f["down_up_ratio"] == 0.0
    assert f["avg_packet_size"] == 0.0


def test_nan_inf_scrubbed():
    rec = _record(last_ts=1.0, fwd_pkt_sizes=[float("inf")], fwd_packets=1)
    vec = extract(rec)
    assert np.all(np.isfinite(vec))
