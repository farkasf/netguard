"""End-to-end pcap replay: PcapPacketSource -> assembler -> scorer.

Uses the committed fixture model (data/fixtures/fixture_model). If it is
missing the test is skipped with a hint to build it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from netguard.capture.source import PcapPacketSource
from netguard.ml.persistence import NetGuardModel
from netguard.pipeline.runner import Runner

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIX = REPO_ROOT / "data" / "fixtures"


@pytest.fixture
def fixture_model():
    path = FIX / "fixture_model"
    if not (path / "meta.json").exists():
        pytest.skip("fixture model missing; run scripts/make_fixture_model.py")
    return NetGuardModel.load(path)


def _run(pcap: Path, repo, model):
    from netguard.pipeline.scorer import Scorer

    scorer = Scorer(repo, model=model)
    runner = Runner(PcapPacketSource(str(pcap)), repo, scorer=scorer)
    runner.run()
    return runner


def test_attack_pcap_flags_anomalies(tmp_repo, fixture_model):
    pcap = FIX / "attack_sample.pcap"
    assert pcap.exists(), "run scripts/make_fixture_pcap.py"
    runner = _run(pcap, tmp_repo, fixture_model)

    assert runner.flows_scored > 0
    assert runner.anomalies_found >= 1
    anomalies = tmp_repo.recent_anomalies(limit=200)
    assert len(anomalies) >= 1
    # The port-scan flows should be classified PORTSCAN (non-benign).
    classes = {a["predicted_class"] for a in anomalies}
    assert "PORTSCAN" in classes
    # Persisted anomalies carry the exact feature vector and model version.
    a = anomalies[0]
    assert len(a["features"]) > 0
    assert a["model_version"] == fixture_model.version


def test_benign_pcap_produces_no_anomalies(tmp_repo, fixture_model):
    pcap = FIX / "benign_sample.pcap"
    assert pcap.exists(), "run scripts/make_fixture_pcap.py"
    runner = _run(pcap, tmp_repo, fixture_model)

    assert runner.flows_scored > 0
    # Benign flows should be classified BENIGN with high confidence -> no
    # anomalies (or at most sub-threshold ones).
    assert runner.anomalies_found == 0
    flows = tmp_repo.recent_flows(limit=100)
    assert all(f["predicted_class"] == "BENIGN" for f in flows)
