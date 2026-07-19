"""Packet sources.

The whole pipeline downstream of this module operates on :class:`ParsedPacket`
dataclasses, never on Scapy types. That seam is what lets the bulk of the test
suite run without a live NIC or root: tests feed packets through
:class:`PcapPacketSource` (or hand-built ``ParsedPacket`` lists) instead of
:class:`LivePacketSource`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# TCP flag bit masks (matching Scapy's TCP.flags integer encoding).
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20


@dataclass(frozen=True, slots=True)
class ParsedPacket:
    """A protocol-agnostic view of a single captured packet."""

    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str  # "TCP" | "UDP" | "OTHER"
    length: int  # total frame length in bytes (headers + payload)
    tcp_flags: int = 0  # bitmask; 0 for non-TCP

    def has_flag(self, mask: int) -> bool:
        return bool(self.tcp_flags & mask)


@runtime_checkable
class PacketSource(Protocol):
    """A source of parsed packets."""

    def packets(self) -> Iterator[ParsedPacket]:  # pragma: no cover - protocol
        ...


def _parse_scapy(pkt: object, default_ts: float) -> ParsedPacket | None:
    """Convert a Scapy packet into a :class:`ParsedPacket`.

    Returns ``None`` for packets without an IP layer (ARP, etc.). Scapy is
    imported lazily so importing this module never requires Scapy to be present
    at parse time for callers that only use :func:`from_parts`.
    """
    # Importing l2 registers Ether<->DLT_EN10MB so rdpcap dissects link layer
    # correctly under scapy's lazy loading (otherwise frames decode as Raw).
    import scapy.layers.l2  # noqa: F401
    from scapy.layers.inet import IP, TCP, UDP

    if IP not in pkt:  # type: ignore[operator]
        return None

    ip = pkt[IP]  # type: ignore[index]
    ts = float(getattr(pkt, "time", default_ts))
    length = int(len(pkt))  # type: ignore[arg-type]

    if TCP in pkt:  # type: ignore[operator]
        tcp = pkt[TCP]  # type: ignore[index]
        return ParsedPacket(
            ts=ts,
            src_ip=str(ip.src),
            dst_ip=str(ip.dst),
            src_port=int(tcp.sport),
            dst_port=int(tcp.dport),
            protocol="TCP",
            length=length,
            tcp_flags=int(tcp.flags),
        )
    if UDP in pkt:  # type: ignore[operator]
        udp = pkt[UDP]  # type: ignore[index]
        return ParsedPacket(
            ts=ts,
            src_ip=str(ip.src),
            dst_ip=str(ip.dst),
            src_port=int(udp.sport),
            dst_port=int(udp.dport),
            protocol="UDP",
            length=length,
        )
    # IP without a transport layer we model (ICMP, etc.).
    return ParsedPacket(
        ts=ts,
        src_ip=str(ip.src),
        dst_ip=str(ip.dst),
        src_port=0,
        dst_port=0,
        protocol="OTHER",
        length=length,
    )


class PcapPacketSource:
    """Deterministic packet source backed by a pcap file (``rdpcap``).

    Requires no privileges; used by all integration tests and CI.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def packets(self) -> Iterator[ParsedPacket]:
        # Register link-layer bindings *before* rdpcap dissects, so Ethernet
        # frames are decoded as Ether/IP/... rather than Raw.
        import scapy.layers.inet  # noqa: F401
        import scapy.layers.l2  # noqa: F401
        from scapy.utils import rdpcap

        for idx, pkt in enumerate(rdpcap(self.path)):
            parsed = _parse_scapy(pkt, default_ts=float(idx))
            if parsed is not None:
                yield parsed


class LivePacketSource:
    """Live capture via Scapy ``AsyncSniffer``. Requires ``CAP_NET_RAW``.

    ``store=False`` keeps memory flat: the sniffer thread pushes packets into a
    bounded queue and this generator yields them as they arrive, so unbounded
    captures stream instead of buffering. ``count``/``timeout`` allow bounded
    captures (handy for smoke tests).
    """

    def __init__(self, iface: str, count: int = 0, timeout: float | None = None) -> None:
        self.iface = iface
        self.count = count
        self.timeout = timeout

    def packets(self) -> Iterator[ParsedPacket]:
        import queue
        import time

        from scapy.sendrecv import AsyncSniffer

        q: queue.Queue[object] = queue.Queue(maxsize=65536)

        def _enqueue(pkt: object) -> None:
            try:
                q.put_nowait(pkt)
            except queue.Full:
                pass  # drop under backpressure rather than stall the capture thread

        sniffer = AsyncSniffer(
            iface=self.iface,
            store=False,
            count=self.count,
            timeout=self.timeout,
            prn=_enqueue,
        )
        sniffer.start()
        try:
            while True:
                try:
                    pkt = q.get(timeout=0.5)
                except queue.Empty:
                    if sniffer.thread is None or not sniffer.thread.is_alive():
                        break  # capture finished (count/timeout hit) and queue drained
                    continue
                parsed = _parse_scapy(pkt, default_ts=time.time())
                if parsed is not None:
                    yield parsed
        finally:
            if sniffer.running:
                sniffer.stop()
