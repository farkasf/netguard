"""Long-lived capture -> assembler -> scorer process.

This is the privileged process on the Pi (needs ``CAP_NET_RAW`` for live
capture). It wires a :class:`PacketSource` into a :class:`FlowAssembler` whose
closed-flow callback feeds the :class:`Scorer`. A periodic sweep flushes idle
flows so they get scored even when no teardown packet is seen.
"""

from __future__ import annotations

import argparse
import time

from netguard.capture.flow_assembler import FlowAssembler, FlowRecord
from netguard.capture.source import LivePacketSource, PacketSource, PcapPacketSource
from netguard.config import get_settings
from netguard.pipeline.scorer import Scorer
from netguard.store.repository import Repository


class Runner:
    """Owns the assembler + scorer and drives a packet source to completion."""

    def __init__(self, source: PacketSource, repo: Repository, scorer: Scorer | None = None):
        self.settings = get_settings()
        self.source = source
        self.repo = repo
        self.scorer = scorer or Scorer(repo)
        self.assembler = FlowAssembler(
            on_close=self._on_flow_closed,
            inactive_timeout=self.settings.inactive_timeout,
            active_timeout=self.settings.active_timeout,
        )
        self.flows_scored = 0
        self.anomalies_found = 0

    def _on_flow_closed(self, record: FlowRecord) -> None:
        result = self.scorer.score_flow(record)
        self.flows_scored += 1
        if result["anomaly"]:
            self.anomalies_found += 1

    def run(self, sweep: bool = True) -> None:
        """Process every packet from the source, sweeping expired flows.

        For a finite pcap source this returns when the source is exhausted; for
        a live source it runs until the source stops yielding.
        """
        last_sweep = time.time()
        last_ts = 0.0
        for pkt in self.source.packets():
            self.assembler.add_packet(pkt)
            last_ts = max(last_ts, pkt.ts)
            now = time.time()
            if sweep and (now - last_sweep) >= self.settings.sweep_interval:
                # Judge expiry on the packet clock: correct for historic pcap
                # timestamps, and equal to the wall clock for live capture.
                self.assembler.flush_expired(last_ts)
                last_sweep = now
        # Drain everything still open at end of stream.
        self.assembler.flush_all()


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="NetGuard capture + scoring runner.")
    parser.add_argument("--pcap", help="replay a pcap instead of live capture")
    parser.add_argument("--iface", default=settings.iface, help="interface for live capture")
    parser.add_argument("--count", type=int, default=0, help="live: stop after N packets")
    parser.add_argument("--timeout", type=float, default=None, help="live: stop after N seconds")
    args = parser.parse_args(argv)

    repo = Repository()
    source: PacketSource
    if args.pcap:
        source = PcapPacketSource(args.pcap)
    else:
        source = LivePacketSource(args.iface, count=args.count, timeout=args.timeout)

    runner = Runner(source, repo)
    print(f"Runner starting (model={runner.scorer.model_version}).")
    runner.run()
    print(f"Done. flows_scored={runner.flows_scored} anomalies={runner.anomalies_found}")
    repo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
