"""SQLite data-access layer.

A thin DAO over the schema in :mod:`schema.sql`. WAL mode is enabled so the
capture/scoring runner can write concurrently with API reads. The connection
uses ``check_same_thread=False`` and a short busy timeout so the FastAPI
threadpool and APScheduler jobs can share one repository.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from netguard.config import get_settings


class Repository:
    """Data-access object for anomalies, flows, and the model registry."""

    def __init__(self, db_path: str | Path | None = None, schema_path: str | Path | None = None):
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else settings.db_path
        self.schema_path = Path(schema_path) if schema_path else settings.schema_path
        self.flows_cap = settings.flows_table_cap
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000")
        self.init_schema()

    # ------------------------------------------------------------- lifecycle
    def init_schema(self) -> None:
        sql = Path(self.schema_path).read_text()
        self._conn.executescript(sql)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------- anomalies
    def insert_anomaly(
        self,
        ts: float,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        predicted_class: str,
        confidence: float,
        features: list[float],
        model_version: str,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO anomalies
               (ts, src_ip, dst_ip, src_port, dst_port, protocol,
                predicted_class, confidence, features_json, model_version)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                ts, src_ip, dst_ip, src_port, dst_port, protocol,
                predicted_class, confidence, json.dumps(features), model_version,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def recent_anomalies(self, limit: int = 100, since: float | None = None) -> list[dict[str, Any]]:
        if since is not None:
            cur = self._conn.execute(
                "SELECT * FROM anomalies WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (since, limit),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM anomalies ORDER BY ts DESC LIMIT ?", (limit,)
            )
        return [self._anomaly_row(r) for r in cur.fetchall()]

    @staticmethod
    def _anomaly_row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["features"] = json.loads(d.pop("features_json"))
        return d

    # ------------------------------------------------------------------ flows
    def insert_flow(
        self,
        last_ts: float,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        predicted_class: str,
        confidence: float,
        duration: float,
        total_packets: int,
        total_bytes: int,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO flows
               (last_ts, src_ip, dst_ip, src_port, dst_port, protocol,
                predicted_class, confidence, duration, total_packets, total_bytes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                last_ts, src_ip, dst_ip, src_port, dst_port, protocol,
                predicted_class, confidence, duration, total_packets, total_bytes,
            ),
        )
        self._conn.commit()
        self.prune_flows()
        return int(cur.lastrowid or 0)

    def recent_flows(self, limit: int = 100) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM flows ORDER BY last_ts DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]

    def prune_flows(self) -> None:
        """Keep only the most recent ``flows_cap`` rows."""
        self._conn.execute(
            """DELETE FROM flows WHERE id NOT IN (
                   SELECT id FROM flows ORDER BY last_ts DESC LIMIT ?
               )""",
            (self.flows_cap,),
        )
        self._conn.commit()

    def count_flows(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM flows").fetchone()[0])

    def count_anomalies(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0])

    # -------------------------------------------------------------- registry
    def register_model(
        self,
        version: str,
        path: str,
        macro_f1: float,
        metrics: dict[str, Any] | None = None,
        activate: bool = False,
        promoted_at: float | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO model_registry
               (version, path, macro_f1, promoted_at, is_active, metrics_json)
               VALUES (?,?,?,?,?,?)""",
            (
                version, path, macro_f1, promoted_at or time.time(),
                1 if activate else 0,
                json.dumps(metrics) if metrics is not None else None,
            ),
        )
        self._conn.commit()
        if activate:
            self.set_active(version)
        return int(cur.lastrowid or 0)

    def set_active(self, version: str) -> None:
        """Atomically make ``version`` the only active model."""
        self._conn.execute("UPDATE model_registry SET is_active=0")
        self._conn.execute(
            "UPDATE model_registry SET is_active=1 WHERE version=?", (version,)
        )
        self._conn.commit()

    def active_model(self) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT * FROM model_registry WHERE is_active=1 ORDER BY promoted_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        return self._registry_row(row) if row else None

    def list_models(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM model_registry ORDER BY promoted_at DESC"
        )
        return [self._registry_row(r) for r in cur.fetchall()]

    @staticmethod
    def _registry_row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["is_active"] = bool(d["is_active"])
        if d.get("metrics_json"):
            d["metrics"] = json.loads(d["metrics_json"])
        else:
            d["metrics"] = None
        d.pop("metrics_json", None)
        return d
