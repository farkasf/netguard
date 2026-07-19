"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from netguard.capture.source import ACK, FIN, PSH, SYN, ParsedPacket
from netguard.store.repository import Repository

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "data" / "fixtures"


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Repository:
    """A Repository backed by a throwaway SQLite file."""
    db = tmp_path / "test.db"
    repo = Repository(db_path=db, schema_path=REPO_ROOT / "netguard" / "store" / "schema.sql")
    yield repo
    repo.close()


@pytest.fixture
def fixture_model_dir() -> Path:
    """Path to the committed fixture model (built by make_fixture_model.py)."""
    return FIXTURE_DIR / "fixture_model"


def pkt(
    ts: float,
    src_ip: str = "10.0.0.1",
    src_port: int = 1234,
    dst_ip: str = "10.0.0.2",
    dst_port: int = 80,
    protocol: str = "TCP",
    length: int = 100,
    flags: int = 0,
) -> ParsedPacket:
    """Convenience constructor for hand-built packets in tests."""
    return ParsedPacket(
        ts=ts, src_ip=src_ip, dst_ip=dst_ip, src_port=src_port,
        dst_port=dst_port, protocol=protocol, length=length, tcp_flags=flags,
    )


@pytest.fixture
def toy_3class():
    """A separable 3-class problem in 4-D feature space."""
    rng = np.random.default_rng(7)
    n = 150
    centers = np.array([[0, 0, 0, 0], [5, 5, 0, 0], [0, 0, 5, 5]], dtype=float)
    X = np.vstack([rng.normal(c, 0.4, size=(n, 4)) for c in centers])
    y = np.array([0] * n + [1] * n + [2] * n)
    return X, y


# Re-export flag constants for tests.
__all__ = ["pkt", "SYN", "ACK", "FIN", "PSH"]
