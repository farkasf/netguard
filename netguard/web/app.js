"use strict";

const API = "/api";
const POLL_MS = 3000;
const BENIGN = "BENIGN";

async function getJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(path + " -> " + res.status);
  return res.json();
}

function timeAgo(ts) {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 60) return s + "s ago";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  return Math.floor(s / 86400) + "d ago";
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function isBenign(cls) { return (cls || "").toUpperCase() === BENIGN; }

// ----------------------------------------------------------------- health
function fmtUptime(s) {
  s = Math.floor(s);
  if (s < 60) return s + "s";
  if (s < 3600) return Math.floor(s / 60) + "m";
  if (s < 86400) return Math.floor(s / 3600) + "h";
  return Math.floor(s / 86400) + "d";
}

async function refreshHealth() {
  const state = document.getElementById("health-dot");
  const text = document.getElementById("health-text");
  try {
    const h = await getJSON("/health");
    state.className = "State State--open";
    text.textContent = "Online";
    document.getElementById("model-version").textContent = h.model_version;
    document.getElementById("uptime").textContent = fmtUptime(h.uptime_s);
  } catch (e) {
    state.className = "State State--closed";
    text.textContent = "Offline";
  }
}

// ------------------------------------------------------------------ flows
async function refreshFlows() {
  const flows = await getJSON("/flows?limit=100");
  const tbody = document.querySelector("#flows-table tbody");
  document.getElementById("flows-counter").textContent = flows.length;
  tbody.innerHTML = "";
  for (const f of flows) {
    const tr = document.createElement("tr");
    tr.className = isBenign(f.predicted_class) ? "benign" : "flagged";
    const badgeCls = isBenign(f.predicted_class) ? "cls-benign" : "cls-attack";
    tr.innerHTML = `
      <td>${fmtTime(f.last_ts)}</td>
      <td>${f.src_ip}:${f.src_port}</td>
      <td>${f.dst_ip}:${f.dst_port}</td>
      <td>${f.protocol}</td>
      <td><span class="cls-badge ${badgeCls}">${f.predicted_class}</span></td>
      <td>${(f.confidence * 100).toFixed(1)}%</td>
      <td>${f.total_packets}</td>
      <td>${f.total_bytes}</td>`;
    tbody.appendChild(tr);
  }
}

// -------------------------------------------------------------- anomalies
let FEATURE_NAMES = [];
async function refreshAnomalies() {
  const items = await getJSON("/anomalies?limit=100");
  const feed = document.getElementById("anomaly-feed");
  document.getElementById("anomaly-counter").textContent = items.length;
  feed.innerHTML = "";
  for (const a of items) {
    const div = document.createElement("div");
    div.className = "anomaly";
    const featRows = a.features.map((v, i) =>
      `<span>${(FEATURE_NAMES[i] || ("f" + i))}: ${v.toFixed(3)}</span>`).join("");
    div.innerHTML = `
      <div class="row1">
        <span class="tuple">${a.src_ip}:${a.src_port} → ${a.dst_ip}:${a.dst_port} <span class="cls-badge cls-attack">${a.predicted_class}</span></span>
        <span class="time">${(a.confidence*100).toFixed(1)}% · ${timeAgo(a.ts)}</span>
      </div>
      <div class="features">${featRows}</div>`;
    div.addEventListener("click", () => div.classList.toggle("open"));
    feed.appendChild(div);
  }
}

// ---------------------------------------------------------------- metrics
function bar(name, value, max) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return `<div class="bar-row">
      <span class="name" title="${name}">${name}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${pct.toFixed(1)}%"></span></span>
      <span class="num">${value.toFixed(3)}</span>
    </div>`;
}

async function refreshMetrics() {
  const m = await getJSON("/metrics");
  if (m.feature_names && m.feature_names.length) FEATURE_NAMES = m.feature_names;

  const summary = document.getElementById("metrics-summary");
  summary.innerHTML = `
    <div class="metric"><span class="label">macro F1</span><span class="value">${m.macro_f1 != null ? m.macro_f1.toFixed(3) : "—"}</span></div>
    <div class="metric"><span class="label">accuracy</span><span class="value">${m.accuracy != null ? m.accuracy.toFixed(3) : "—"}</span></div>
    <div class="metric"><span class="label">flows</span><span class="value">${m.flow_count}</span></div>
    <div class="metric"><span class="label">anomalies</span><span class="value">${m.anomaly_count}</span></div>`;

  // Per-class F1 bars.
  const pc = document.getElementById("perclass-bars");
  pc.innerHTML = "";
  for (const cls of (m.classes || [])) {
    const met = m.per_class[cls];
    if (!met) continue;
    pc.innerHTML += bar(cls + " F1", met.f1, 1);
  }

  // Confusion matrix.
  const conf = document.getElementById("confusion");
  conf.innerHTML = "";
  const classes = m.classes || [];
  const cm = m.confusion_matrix || [];
  if (classes.length && cm.length) {
    conf.style.gridTemplateColumns = `repeat(${classes.length + 1}, auto)`;
    conf.innerHTML += `<div class="cell corner"></div>`;
    for (const c of classes) conf.innerHTML += `<div class="cell head">${c.slice(0,6)}</div>`;
    cm.forEach((row, i) => {
      conf.innerHTML += `<div class="cell head">${classes[i].slice(0,6)}</div>`;
      row.forEach((v, j) => {
        conf.innerHTML += `<div class="cell ${i===j ? "diag" : ""}">${v}</div>`;
      });
    });
  }

  // Feature importance (top 12).
  const fi = document.getElementById("feature-importance");
  fi.innerHTML = "";
  const imp = m.feature_importances || [];
  if (imp.length) {
    const pairs = imp.map((v, i) => [FEATURE_NAMES[i] || ("f"+i), v])
      .sort((a, b) => b[1] - a[1]).slice(0, 12);
    const max = Math.max(...pairs.map(p => p[1]), 1e-9);
    for (const [name, v] of pairs) fi.innerHTML += bar(name, v, max);
  }
}

// ---------------------------------------------------------------- retrain
async function triggerRetrain() {
  const btn = document.getElementById("retrain-btn");
  const status = document.getElementById("retrain-status");
  btn.disabled = true;
  status.textContent = "submitting…";
  try {
    const res = await fetch(API + "/retrain", { method: "POST" });
    const body = await res.json();
    status.textContent = "job " + body.job_id + " running…";
    pollRetrain(body.job_id);
  } catch (e) {
    status.textContent = "failed";
    btn.disabled = false;
  }
}

async function pollRetrain(jobId) {
  const status = document.getElementById("retrain-status");
  const btn = document.getElementById("retrain-btn");
  const tick = async () => {
    const last = await getJSON("/retrain/last");
    if (last && last.job_id === jobId && last.status !== "none") {
      status.textContent = `${last.status} (F1 ${Number(last.candidate_f1).toFixed(3)})`;
      btn.disabled = false;
      refreshMetrics();
      return;
    }
    setTimeout(tick, 1500);
  };
  setTimeout(tick, 1500);
}

// ------------------------------------------------------------------- loop
async function refreshAll() {
  await Promise.allSettled([
    refreshHealth(), refreshFlows(), refreshAnomalies(), refreshMetrics(),
  ]);
}

document.getElementById("retrain-btn").addEventListener("click", triggerRetrain);
document.getElementById("poll-interval").textContent = (POLL_MS / 1000).toString();
refreshAll();
setInterval(refreshAll, POLL_MS);
