#!/usr/bin/env bash
# Replay a labeled pcap at a live interface for an on-device smoke test.
#
# Usage:
#   sudo ./scripts/replay.sh <pcap> [iface]
#
# Requires tcpreplay (apt install tcpreplay). The NetGuard runner must be
# sniffing the same interface. On a Pi, the loopback won't carry replayed
# frames, so use a real iface (e.g. eth0) and point the runner at it.
set -euo pipefail

PCAP="${1:-data/fixtures/attack_sample.pcap}"
IFACE="${2:-eth0}"

if ! command -v tcpreplay >/dev/null 2>&1; then
  echo "tcpreplay not found. Install it: sudo apt install tcpreplay" >&2
  exit 1
fi

if [[ ! -f "$PCAP" ]]; then
  echo "pcap not found: $PCAP" >&2
  exit 1
fi

echo "Replaying $PCAP on $IFACE ..."
tcpreplay --intf1="$IFACE" --mbps=1 "$PCAP"
echo "Done. Check the NetGuard dashboard / /api/anomalies."
