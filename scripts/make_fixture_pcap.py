#!/usr/bin/env python3
"""Build tiny, committable labeled pcaps for CI and integration tests.

Crafts two pcaps with Scapy:
  * ``benign_sample.pcap`` — a handful of well-formed short TCP/UDP exchanges
    (SYN/SYN-ACK/ACK ... data ... FIN/FIN) with modest packet sizes.
  * ``attack_sample.pcap`` — a SYN flood / port-scan pattern: one source firing
    SYNs at many destination ports, tiny packets, no handshakes completed.

These are deterministic and small so they can live in the repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Packet, Raw
from scapy.utils import wrpcap

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "fixtures"

# Fixed MACs so frames are well-formed Ethernet (DLT_EN10MB) — the most
# portable link type for rdpcap across platforms.
MAC_A = "02:00:00:00:00:01"
MAC_B = "02:00:00:00:00:02"


def _tcp(src, sport, dst, dport, flags, ts, payload=b"") -> Packet:
    pkt = (
        Ether(src=MAC_A, dst=MAC_B)
        / IP(src=src, dst=dst)
        / TCP(sport=sport, dport=dport, flags=flags)
    )
    if payload:
        pkt = pkt / Raw(load=payload)
    pkt.time = ts
    return pkt


def build_benign() -> list[Packet]:
    """A few normal client/server conversations."""
    pkts: list[Packet] = []
    t = 1000.0
    client, server = "10.0.0.5", "10.0.0.10"
    for conv in range(3):
        sport = 40000 + conv
        # Handshake.
        pkts.append(_tcp(client, sport, server, 80, "S", t))
        t += 0.01
        pkts.append(_tcp(server, 80, client, sport, "SA", t))
        t += 0.01
        pkts.append(_tcp(client, sport, server, 80, "A", t))
        t += 0.02
        # Data exchange with realistic-ish sizes.
        for _ in range(4):
            pkts.append(_tcp(client, sport, server, 80, "PA", t, payload=b"x" * 200))
            t += 0.05
            pkts.append(_tcp(server, 80, client, sport, "PA", t, payload=b"y" * 800))
            t += 0.05
        # Graceful close (FIN from both sides).
        pkts.append(_tcp(client, sport, server, 80, "FA", t))
        t += 0.01
        pkts.append(_tcp(server, 80, client, sport, "FA", t))
        t += 0.01
    # A benign UDP DNS-ish exchange.
    u = (Ether(src=MAC_A, dst=MAC_B) / IP(src=client, dst="10.0.0.1")
         / UDP(sport=51000, dport=53) / Raw(load=b"q" * 30))
    u.time = t
    pkts.append(u)
    t += 0.02
    u2 = (Ether(src=MAC_B, dst=MAC_A) / IP(src="10.0.0.1", dst=client)
          / UDP(sport=53, dport=51000) / Raw(load=b"r" * 90))
    u2.time = t
    pkts.append(u2)
    return pkts


def build_attack() -> list[Packet]:
    """SYN flood / port scan: one attacker, many ports, tiny SYN-only packets."""
    pkts: list[Packet] = []
    t = 2000.0
    attacker, target = "10.0.0.66", "10.0.0.10"
    for port in range(20, 90):
        # bare SYN, no payload, no completion
        pkts.append(_tcp(attacker, 55000, target, port, "S", t))
        t += 0.001
    return pkts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixture pcaps.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    benign = build_benign()
    attack = build_attack()
    wrpcap(str(out / "benign_sample.pcap"), benign)
    wrpcap(str(out / "attack_sample.pcap"), attack)
    print(f"Wrote {len(benign)} benign packets -> {out / 'benign_sample.pcap'}")
    print(f"Wrote {len(attack)} attack packets -> {out / 'attack_sample.pcap'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
