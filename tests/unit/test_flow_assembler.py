"""Unit tests for the flow assembler."""

from __future__ import annotations

from netguard.capture.flow_assembler import FlowAssembler, FlowState
from netguard.capture.source import ACK, FIN, RST, SYN
from tests.conftest import pkt


def collect():
    closed = []
    return closed, FlowAssembler(on_close=closed.append, inactive_timeout=15.0, active_timeout=120.0)


def test_bidirectional_counts_and_normalization():
    closed, asm = collect()
    # client -> server then server -> client map to ONE flow.
    asm.add_packet(pkt(0.0, "10.0.0.1", 5000, "10.0.0.2", 80, length=100, flags=SYN))
    asm.add_packet(pkt(0.1, "10.0.0.2", 80, "10.0.0.1", 5000, length=200, flags=SYN | ACK))
    asm.add_packet(pkt(0.2, "10.0.0.1", 5000, "10.0.0.2", 80, length=60, flags=ACK))
    assert asm.active_count == 1
    asm.flush_all()

    assert len(closed) == 1
    rec = closed[0]
    assert rec.fwd_packets == 2  # two from the initiator
    assert rec.bwd_packets == 1
    assert rec.fwd_bytes == 160
    assert rec.bwd_bytes == 200
    assert rec.flags["SYN"] == 2
    assert rec.flags["ACK"] == 2
    assert rec.src_ip == "10.0.0.1" and rec.dst_ip == "10.0.0.2"
    assert rec.state is FlowState.CLOSED


def test_fin_from_both_sides_closes():
    closed, asm = collect()
    asm.add_packet(pkt(0.0, "a", 1, "b", 2, flags=SYN))
    asm.add_packet(pkt(1.0, "a", 1, "b", 2, flags=FIN | ACK))
    assert asm.active_count == 1  # only one side has FIN'd
    asm.add_packet(pkt(2.0, "b", 2, "a", 1, flags=FIN | ACK))
    assert asm.active_count == 0  # both sides -> closed
    assert len(closed) == 1


def test_rst_closes_immediately():
    closed, asm = collect()
    asm.add_packet(pkt(0.0, "a", 1, "b", 2, flags=SYN))
    asm.add_packet(pkt(0.5, "b", 2, "a", 1, flags=RST))
    assert asm.active_count == 0
    assert len(closed) == 1


def test_inactivity_timeout_closes_with_injected_now():
    closed, asm = collect()
    asm.add_packet(pkt(100.0, "a", 1, "b", 2, flags=ACK))
    asm.flush_expired(now=110.0)  # within timeout
    assert asm.active_count == 1
    asm.flush_expired(now=120.0)  # 20s > 15s inactive timeout
    assert asm.active_count == 0
    assert len(closed) == 1


def test_active_timeout_closes_long_flow():
    closed, asm = collect()
    asm.add_packet(pkt(0.0, "a", 1, "b", 2, flags=ACK))
    # Keep it active with periodic packets but exceed the 120s active timeout.
    asm.add_packet(pkt(119.0, "a", 1, "b", 2, flags=ACK))
    asm.flush_expired(now=121.0)
    assert asm.active_count == 0
    assert len(closed) == 1


def test_single_packet_flow_flushed_at_end():
    closed, asm = collect()
    asm.add_packet(pkt(0.0, "a", 1, "b", 2, length=42, flags=SYN))
    assert asm.active_count == 1  # never closes on its own
    asm.flush_all()
    assert len(closed) == 1
    assert closed[0].total_packets == 1


def test_out_of_order_timestamps_keep_last_ts_max():
    closed, asm = collect()
    asm.add_packet(pkt(5.0, "a", 1, "b", 2))
    asm.add_packet(pkt(2.0, "a", 1, "b", 2))  # earlier ts
    asm.flush_all()
    assert closed[0].last_ts == 5.0
    assert closed[0].start_ts == 5.0


def test_determinism_same_sequence_same_record():
    seq = [
        pkt(0.0, "a", 1, "b", 2, length=100, flags=SYN),
        pkt(0.1, "b", 2, "a", 1, length=120, flags=SYN | ACK),
        pkt(0.2, "a", 1, "b", 2, length=80, flags=ACK),
    ]
    out = []
    for _ in range(2):
        closed, asm = collect()
        for p in seq:
            asm.add_packet(p)
        asm.flush_all()
        out.append((closed[0].fwd_bytes, closed[0].bwd_bytes, dict(closed[0].flags)))
    assert out[0] == out[1]
