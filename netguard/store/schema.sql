-- NetGuard SQLite schema. WAL mode lets the runner write while the API reads.
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Persisted anomalies (flagged flows).
CREATE TABLE IF NOT EXISTS anomalies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    src_ip          TEXT    NOT NULL,
    dst_ip          TEXT    NOT NULL,
    src_port        INTEGER NOT NULL,
    dst_port        INTEGER NOT NULL,
    protocol        TEXT    NOT NULL,
    predicted_class TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    features_json   TEXT    NOT NULL,
    model_version   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_anomalies_ts ON anomalies(ts);

-- Rolling recent flows for the UI live table (pruned by repository).
CREATE TABLE IF NOT EXISTS flows (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    last_ts         REAL    NOT NULL,
    src_ip          TEXT    NOT NULL,
    dst_ip          TEXT    NOT NULL,
    src_port        INTEGER NOT NULL,
    dst_port        INTEGER NOT NULL,
    protocol        TEXT    NOT NULL,
    predicted_class TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    duration        REAL    NOT NULL,
    total_packets   INTEGER NOT NULL,
    total_bytes     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_flows_last_ts ON flows(last_ts);

-- Registry of trained models; exactly one row may be active at a time.
CREATE TABLE IF NOT EXISTS model_registry (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    version      TEXT    NOT NULL UNIQUE,
    path         TEXT    NOT NULL,
    macro_f1     REAL    NOT NULL,
    promoted_at  REAL    NOT NULL,
    is_active    INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_registry_active ON model_registry(is_active);
