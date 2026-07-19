"""Bidirectional flow assembly.

A :class:`FlowAssembler` consumes :class:`~netguard.capture.source.ParsedPacket`
objects and groups them into :class:`FlowRecord` bidirectional conversations,
keyed by a direction-normalized 5-tuple. Closed flows are emitted to a callback.

The assembler is pure and deterministic: the same packet sequence yields
identical FlowRecords. All time-based logic accepts an injected ``now`` so tests
never depend on the wall clock.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from netguard.capture.source import ACK, FIN, PSH, RST, SYN, URG, ParsedPacket


class FlowState(StrEnum):
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


@dataclass(frozen=True, slots=True)
class FlowKey:
    """Direction-normalized 5-tuple.

    The two endpoints are ordered so that A->B and B->A map to the same key.
    ``protocol`` is part of the key so TCP/UDP flows on the same ports differ.
    """

    ip_a: str
    port_a: int
    ip_b: str
    port_b: int
    protocol: str

    @staticmethod
    def from_packet(pkt: ParsedPacket) -> tuple[FlowKey, bool]:
        """Build a normalized key and report whether ``pkt`` is forward.

        ``forward`` is True when the packet's source is endpoint A (the lower
        ``(ip, port)``). The first packet seen for a flow defines the initiator
        as the forward direction, which matches CIC-IDS2017 conventions.
        """
        a = (pkt.src_ip, pkt.src_port)
        b = (pkt.dst_ip, pkt.dst_port)
        if a <= b:
            return FlowKey(pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, pkt.protocol), True
        return FlowKey(pkt.dst_ip, pkt.dst_port, pkt.src_ip, pkt.src_port, pkt.protocol), False


@dataclass
class FlowRecord:
    """An in-flight or closed bidirectional conversation."""

    key: FlowKey
    start_ts: float
    last_ts: float
    # The initiator's endpoint defines the forward direction for fwd/bwd stats.
    init_ip: str = ""
    init_port: int = 0
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    fwd_pkt_sizes: list[int] = field(default_factory=list)
    bwd_pkt_sizes: list[int] = field(default_factory=list)
    fwd_iats: list[float] = field(default_factory=list)
    bwd_iats: list[float] = field(default_factory=list)
    flags: Counter[str] = field(default_factory=Counter)
    state: FlowState = FlowState.ACTIVE
    # Internal bookkeeping for FIN-from-both-sides close logic.
    _last_fwd_ts: float | None = None
    _last_bwd_ts: float | None = None
    _fin_fwd: bool = False
    _fin_bwd: bool = False

    @property
    def src_ip(self) -> str:
        return self.init_ip

    @property
    def dst_ip(self) -> str:
        return self.key.ip_b if self.init_ip == self.key.ip_a else self.key.ip_a

    @property
    def src_port(self) -> int:
        return self.init_port

    @property
    def dst_port(self) -> int:
        return self.key.port_b if self.init_ip == self.key.ip_a else self.key.port_a

    @property
    def protocol(self) -> str:
        return self.key.protocol

    @property
    def total_packets(self) -> int:
        return self.fwd_packets + self.bwd_packets


_FLAG_NAMES: list[tuple[str, int]] = [
    ("SYN", SYN),
    ("ACK", ACK),
    ("FIN", FIN),
    ("RST", RST),
    ("PSH", PSH),
    ("URG", URG),
]


class FlowAssembler:
    """Maintains a flow table and emits closed flows via ``on_close``."""

    def __init__(
        self,
        on_close: Callable[[FlowRecord], None],
        inactive_timeout: float = 15.0,
        active_timeout: float = 120.0,
    ) -> None:
        self._flows: dict[FlowKey, FlowRecord] = {}
        self._on_close = on_close
        self.inactive_timeout = inactive_timeout
        self.active_timeout = active_timeout

    @property
    def active_count(self) -> int:
        return len(self._flows)

    def add_packet(self, pkt: ParsedPacket) -> None:
        """Add a packet, creating or updating its flow record."""
        key, _ = FlowKey.from_packet(pkt)
        rec = self._flows.get(key)
        if rec is None:
            rec = FlowRecord(key=key, start_ts=pkt.ts, last_ts=pkt.ts)
            # The first packet's source is the initiator (forward direction).
            rec.init_ip = pkt.src_ip
            rec.init_port = pkt.src_port
            self._flows[key] = rec

        # Direction is defined relative to the recorded initiator.
        is_forward = (pkt.src_ip, pkt.src_port) == (rec.init_ip, rec.init_port)

        # Inter-arrival times, per direction.
        if is_forward:
            if rec._last_fwd_ts is not None:
                rec.fwd_iats.append(pkt.ts - rec._last_fwd_ts)
            rec._last_fwd_ts = pkt.ts
            rec.fwd_packets += 1
            rec.fwd_bytes += pkt.length
            rec.fwd_pkt_sizes.append(pkt.length)
        else:
            if rec._last_bwd_ts is not None:
                rec.bwd_iats.append(pkt.ts - rec._last_bwd_ts)
            rec._last_bwd_ts = pkt.ts
            rec.bwd_packets += 1
            rec.bwd_bytes += pkt.length
            rec.bwd_pkt_sizes.append(pkt.length)

        rec.last_ts = max(rec.last_ts, pkt.ts)

        # Flag tallies.
        for name, mask in _FLAG_NAMES:
            if pkt.has_flag(mask):
                rec.flags[name] += 1

        # Connection-teardown tracking.
        if pkt.protocol == "TCP":
            if pkt.has_flag(RST):
                self._close(key, rec)
                return
            if pkt.has_flag(FIN):
                if is_forward:
                    rec._fin_fwd = True
                else:
                    rec._fin_bwd = True
                if rec._fin_fwd and rec._fin_bwd:
                    self._close(key, rec)
                    return

    def flush_expired(self, now: float) -> None:
        """Close flows idle past ``inactive_timeout`` or older than ``active_timeout``.

        ``now`` is injected so tests are deterministic.
        """
        to_close = [
            (key, rec)
            for key, rec in self._flows.items()
            if (now - rec.last_ts) > self.inactive_timeout
            or (now - rec.start_ts) > self.active_timeout
        ]
        for key, rec in to_close:
            self._close(key, rec)

    def flush_all(self) -> None:
        """Close every remaining flow (e.g. at end of a pcap)."""
        for key, rec in list(self._flows.items()):
            self._close(key, rec)

    def _close(self, key: FlowKey, rec: FlowRecord) -> None:
        rec.state = FlowState.CLOSED
        del self._flows[key]
        self._on_close(rec)
