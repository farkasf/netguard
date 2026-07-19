# NetGuard - From-Scratch Network Anomaly Detector

![Python](https://img.shields.io/badge/python-3.11%2B-blue)

NetGuard is a self-contained network anomaly-detection system designed for the
Raspberry Pi 5 but happy on any Linux/macOS host. It sniffs live traffic,
assembles bidirectional flows, extracts statistical features, and classifies
each flow with a **Gradient Boosting Machine written entirely from scratch in
NumPy** - no scikit-learn estimator, no XGBoost/LightGBM. Anomalies are
persisted to SQLite and surfaced on a dependency-free web dashboard. A
retraining loop trains candidate models and promotes them only when their
macro F1 beats the incumbent.

> The centerpiece is the ML engine in [`netguard/ml/`](netguard/ml/): a
> hessian-aware CART regression tree, softmax cross-entropy gradients/hessians,
> and a multiclass boosting driver - all in NumPy. A test asserts that importing
> anything under `netguard.ml` never pulls a machine-learning framework into
> `sys.modules`.

## Architecture

```
                 ┌──────────────┐     ParsedPacket      ┌─────────────────┐
  NIC / pcap ──► │ PacketSource │ ────────────────────► │  FlowAssembler  │
                 └──────────────┘                       └────────┬────────┘
                                                                  │ closed FlowRecord
                                                                  ▼
   ┌──────────────┐   features    ┌──────────────┐  predict_proba  ┌────────┐
   │   Extractor  │ ◄──────────── │    Scorer    │ ◄────────────── │  GBM   │ (from scratch)
   └──────────────┘               └──────┬───────┘                 └────────┘
                                          │ flows + anomalies
                                          ▼
                                   ┌──────────────┐      reads      ┌──────────────┐
                                   │  SQLite (WAL)│ ◄────────────── │  FastAPI/API │
                                   └──────────────┘                 └──────┬───────┘
                                          ▲ promote                        │ serves
                                   ┌──────┴───────┐                        ▼
                                   │ retrain_job  │                 ┌──────────────┐
                                   │  + F1 gate   │                 │ Web dashboard│
                                   └──────────────┘                 └──────────────┘
```

Two long-lived processes share one SQLite file (WAL mode handles concurrent
read/write):

- **runner** (`netguard.pipeline.runner`): privileged; captures packets, builds
  flows, scores them, writes results. Needs `CAP_NET_RAW` for live capture.
- **api** (`netguard.api.app`): unprivileged; serves the JSON API and the static
  dashboard, and runs retraining (on demand and, optionally, on a cron schedule).

The `ParsedPacket` dataclass is the seam between Scapy and everything else:
downstream code and the whole test suite operate on `ParsedPacket`, never on
Scapy types - which is what lets the suite run with no live NIC and no root.

## The ML engine (from scratch)

Everything under `netguard/ml/` depends on **NumPy only**:

- **CART regression tree** ([`ml/tree.py`](netguard/ml/tree.py)) - fits the
  Newton step of a second-order objective. Split quality uses the modern
  gradient-boosting gain

  ```
  gain = 0.5 * ( G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_p²/(H_p+λ) ) − γ
  ```

  and each leaf is the Newton step `leaf = −G / (H + λ)`. The threshold scan is
  vectorized with prefix sums over the sorted order - O(n) per feature, no
  Python double loop, so it's affordable on a Pi. Stopping criteria:
  `max_depth`, `min_samples_leaf`, `min_child_weight`, `min_split_gain`.

- **Losses** ([`ml/losses.py`](netguard/ml/losses.py)) - numerically stable
  `softmax`/`sigmoid`; per-class gradient `g = p − y`, hessian `h = p(1 − p)`.
  Unit-tested against finite differences.

- **Boosting driver** ([`ml/gbm.py`](netguard/ml/gbm.py)) - multiclass
  `GradientBoostingClassifier` with an sklearn-like surface (`fit`, `predict`,
  `predict_proba`): raw scores start at log class priors; each round fits one
  tree per class to the softmax gradients/hessians and adds
  `learning_rate · tree(X)`. Row/feature subsampling via NumPy masks; optional
  early stopping on validation macro F1; split-gain feature importance.

Supporting modules: [`ml/encoders.py`](netguard/ml/encoders.py) (from-scratch
`LabelEncoder` + `StandardScaler`) and
[`ml/persistence.py`](netguard/ml/persistence.py) (JSON serialization of the
whole model; `load()` reproduces `predict_proba` **bit-for-bit**).

## Features extracted per flow

Feature order is **frozen** in one place
([`features/extractor.py`](netguard/features/extractor.py)) so training and
inference can't drift. The 26 features align with CIC-IDS2017 column semantics:

`duration`, `total_fwd_packets`, `total_bwd_packets`, `total_fwd_bytes`,
`total_bwd_bytes`, `fwd_pkt_size_{mean,std,min,max}`,
`bwd_pkt_size_{mean,std,min,max}`, `flow_bytes_per_s`, `flow_packets_per_s`,
`fwd_iat_{mean,std}`, `bwd_iat_{mean,std}`, `down_up_ratio`,
`syn_count`, `ack_count`, `fin_count`, `rst_count`, `psh_count`,
`avg_packet_size`.

Every division is guarded against zero and NaN/Inf are scrubbed to `0.0`.

## Quick start

Requires **Python 3.11+**.

```bash
# 1. Create a venv and install (editable, with dev extras for tests/lint).
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Build the committable fixtures: tiny labeled pcaps + a fixture model.
python scripts/make_fixture_pcap.py
python scripts/make_fixture_model.py

# 3. Train a model on the built-in synthetic dataset and register it.
python -m netguard.training.train --synthetic --evaluate --register

# 4. Start the API + dashboard.
python -m netguard.api.app
# open http://localhost:8000

# 5. (Optional) Replay a pcap through the scoring pipeline to populate the UI.
python -m netguard.pipeline.runner --pcap data/fixtures/attack_sample.pcap
```

No CIC-IDS2017 download is required to run end-to-end: the synthetic dataset
generates three separable classes (`BENIGN`, `PORTSCAN`, `DOS`) in feature
space. It exists to demonstrate the pipeline - expect false positives if you
score real traffic with a synthetic-trained model. For real use, train on your
own captured baseline (`--pcap`) or on CIC-IDS2017 CSVs (`--cic`).

## Training a model

`python -m netguard.training.train` accepts one data source:

| Flag | Source |
|---|---|
| `--synthetic` | built-in separable 3-class dataset (no download) |
| `--csv FILE` | generic CSV, last column is the label |
| `--cic FILE` | CIC-IDS2017 CSV (columns mapped onto the frozen feature order) |
| `--pcap FILE` | a pcap whose flows all share `--pcap-label` |

Useful options: `--evaluate` (held-out metrics report), `--test-size`,
`--register` (register + activate the artifact in the DB), `--out DIR`.
scikit-learn is used **only** in evaluation/test code (never imported by
`netguard.ml`); on the separable synthetic dataset the from-scratch GBM reaches
macro F1 = 1.0 on the held-out split - a clean check that the boosting and
trees are correct.

## Running the live pipeline

```bash
# Live capture on eth0 (needs CAP_NET_RAW; see deployment).
python -m netguard.pipeline.runner --iface eth0

# Deterministic pcap replay (no privileges) - used by integration tests.
python -m netguard.pipeline.runner --pcap data/fixtures/attack_sample.pcap
```

Flows close on FIN-from-both-sides, on RST, after `inactive_timeout` of
silence, or after `active_timeout` for long-lived flows. Each closed flow is
scored and written to the rolling `flows` table; an `anomalies` row is added
when the predicted class is non-benign **or** confidence falls below
`low_confidence_threshold`.

## The API

FastAPI app factory in [`api/app.py`](netguard/api/app.py); routes under
`/api`, dashboard served at `/`.

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{status, model_version, uptime_s}` |
| GET | `/api/flows?limit=100` | recent flows from the rolling table |
| GET | `/api/anomalies?limit=100&since=<ts>` | recent anomalies, newest first |
| GET | `/api/metrics` | active model P/R/F1, confusion matrix, feature importances, counts |
| POST | `/api/retrain` | starts a background retrain; returns a job id (202) |
| GET | `/api/retrain/last` | status/result of the most recent retrain job |
| GET | `/api/model/registry` | all models with F1 and active flag |

The dashboard ([`netguard/web/`](netguard/web/)) is a single static page -
vanilla HTML/CSS/JS, no build step, no CDN - polling the API every few seconds:
live flows table, anomaly feed with expandable per-flow feature vectors,
per-class F1 bars, confusion matrix, top feature importances, and a retrain
button.

## Retraining and the F1 gate

[`training/retrain_job.py`](netguard/training/retrain_job.py) trains a candidate
on a fresh train/test split, evaluates it, and **promotes only if its macro F1
strictly beats the incumbent's**. On promotion the candidate is registered and
activated, and the live scorer is hot-reloaded - no process restart. A rejected
candidate is still recorded (inactive) for the audit trail.

The dashboard's button triggers a retrain on demand; setting
`NETGUARD_RETRAIN_ENABLED=true` additionally runs it on a cron schedule
(`NETGUARD_RETRAIN_CRON`, default daily at 03:00) via APScheduler inside the
API process.

## Configuration

All tunables live in [`netguard/config.py`](netguard/config.py)
(pydantic-settings). Override any value with an environment variable prefixed
`NETGUARD_`, or a `.env` file. Selected settings:

| Env var | Default | Meaning |
|---|---|---|
| `NETGUARD_IFACE` | `eth0` | capture interface |
| `NETGUARD_INACTIVE_TIMEOUT` | `15` | seconds of silence that closes a flow |
| `NETGUARD_ACTIVE_TIMEOUT` | `120` | max lifetime of a long-lived flow |
| `NETGUARD_LOW_CONFIDENCE_THRESHOLD` | `0.5` | below this, flag even benign flows |
| `NETGUARD_FLOWS_TABLE_CAP` | `5000` | rolling `flows` rows kept for the UI |
| `NETGUARD_API_HOST` / `NETGUARD_API_PORT` | `0.0.0.0` / `8000` | API bind |
| `NETGUARD_DB_PATH` | `data/netguard.db` | SQLite file |
| `NETGUARD_MODELS_DIR` | `models/` | promoted artifacts |
| `NETGUARD_RETRAIN_ENABLED` | `false` | run scheduled retrains in the API |
| `NETGUARD_RETRAIN_CRON` | `0 3 * * *` | retrain schedule (crontab syntax) |
| `NETGUARD_GBM_*` | - | GBM hyperparameters (`N_ESTIMATORS`, `LEARNING_RATE`, `MAX_DEPTH`, `REG_LAMBDA`, `SUBSAMPLE`, `COLSAMPLE`, …) |

Keep `n_estimators × max_depth` modest on the Pi.

## Testing

The suite runs in CI on x86 **without root and without a NIC**: unit tests
cover the assembler, extractor, tree/losses/GBM (including gradients vs finite
differences and an ML-framework purity guard), encoders, persistence
round-trips, and the F1 promotion gate; integration tests replay committed
fixture pcaps end-to-end (`attack_sample.pcap` must flag PORTSCAN anomalies,
`benign_sample.pcap` none); API tests assert every endpoint against a seeded
DB.

```bash
pytest                       # unit + integration + api
pytest --cov=netguard        # with coverage
ruff check netguard tests    # lint
mypy netguard                # type-check
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff + mypy +
pytest with coverage on freshly built fixtures.

## Deployment on a Raspberry Pi 4

```bash
sudo useradd -r -s /usr/sbin/nologin netguard
sudo mkdir -p /opt/netguard && sudo chown netguard:netguard /opt/netguard
# copy the repo to /opt/netguard, then as the netguard user:
python -m venv /opt/netguard/.venv
/opt/netguard/.venv/bin/pip install -e /opt/netguard

# Grant raw-capture capability to the venv interpreter (no full root needed):
sudo setcap cap_net_raw,cap_net_admin+eip \
    $(readlink -f /opt/netguard/.venv/bin/python3)

# Install the two systemd units (see deploy/).
sudo cp deploy/netguard-runner.service deploy/netguard-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now netguard-runner netguard-api
```

Browse from a laptop at `http://<pi-ip>:8000`. For an on-device smoke test,
`sudo ./scripts/replay.sh data/fixtures/attack_sample.pcap eth0` pushes a
labeled pcap at the live interface with `tcpreplay`.

## Repository layout

```
netguard/
├── capture/        # ParsedPacket, PacketSource (live + pcap), FlowAssembler
├── features/       # extractor.py with frozen FEATURE_NAMES
├── ml/             # ZERO ML-framework imports: tree, losses, gbm, encoders, persistence
├── pipeline/       # scorer.py (load model, score, persist), runner.py (long-lived)
├── store/          # schema.sql + repository.py (SQLite DAO, WAL)
├── training/       # dataset.py, train.py (CLI + eval), retrain_job.py (F1 gate)
├── api/            # FastAPI app.py, routes.py, schemas.py
├── web/            # index.html, app.js, style.css (no build step)
└── config.py       # pydantic-settings; all tunables
scripts/            # make_fixture_pcap.py, make_fixture_model.py, replay.sh
deploy/             # systemd units
data/fixtures/      # tiny committable pcaps + fixture model
tests/              # unit / integration / api
```

## Design notes

**Why from scratch?** The point is to show that gradient boosting (trees,
gradients, hessians, the additive update) is comprehensible and implementable
in a few hundred lines of NumPy. `netguard/ml/` has no ML-framework dependency,
and a test enforces it.

**Why are flows direction-normalized?** A flow is keyed by an ordered 5-tuple so
A→B and B→A map to one bidirectional conversation; the first packet's source is
recorded as the initiator to keep forward/backward statistics meaningful - the
same convention CIC-IDS2017 uses.

**Scope.** NetGuard is a detector, not an IPS - no inline blocking, no deep
learning, single node, no third-party threat feeds.
