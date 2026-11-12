"""Digital twin core: persistent patient state + ML inference + what-if."""
import os, json, sqlite3, time, joblib, numpy as np, pandas as pd
from edge import edge_alerts
from crypto import enc, dec

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "twin.db")
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]
# Static "EHR" fields; used when a patient has no record on file
DEFAULT_EHR = {"cp": 0, "chol": 240, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}
VITALS = ["hr", "spo2", "sys_bp", "dia_bp", "temp", "glucose"]
_models = {}

def models():
    if not _models:
        d = os.path.join(BASE, "ml", "models")
        _models["rf"] = joblib.load(os.path.join(d, "heart_model.joblib"))
        _models["iso"] = joblib.load(os.path.join(d, "anomaly_iso.joblib"))
    return _models

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    return c

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS patients(id TEXT PRIMARY KEY, name TEXT, age INT, sex INT, bmi REAL, ehr TEXT);
        CREATE TABLE IF NOT EXISTS readings(id INTEGER PRIMARY KEY AUTOINCREMENT, pid TEXT, ts REAL,
            hr REAL, spo2 REAL, sys_bp REAL, dia_bp REAL, temp REAL, glucose REAL,
            risk REAL, anomaly INT);
        CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, pid TEXT, ts REAL,
            level TEXT, msg TEXT, source TEXT);
        """)
        try: c.execute("ALTER TABLE patients ADD COLUMN ehr TEXT")   # upgrade older twin.db
        except sqlite3.OperationalError: pass

def upsert_patient(p):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO patients VALUES(?,?,?,?,?,?)",
                  (p["id"], p.get("name", p["id"]), p["age"], p["sex"], p.get("bmi", 0),
                   enc(p.get("ehr", {}))))

def model_input(p, v: dict, chol_delta=0.0) -> dict:
    """EHR (static) + live IoT vitals -> model features. BP and blood sugar come live from the wearable."""
    ehr = {**DEFAULT_EHR, **dec(p["ehr"])}
    ehr["chol"] = ehr["chol"] + chol_delta
    return {"age": p["age"], "sex": p["sex"], **ehr, "trestbps": v["sys_bp"], "fbs": int(v["glucose"] >= 120)}

def risk_of(profile: dict) -> float:
    x = pd.DataFrame([[profile[f] for f in FEATURES]], columns=FEATURES)
    return float(models()["rf"].predict_proba(x)[0, 1])

def ingest(pid: str, v: dict):
    with db() as c:
        p = c.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    if not p:
        raise KeyError("unknown patient")
    risk = risk_of(model_input(p, v))
    with db() as c:
        prev = c.execute("SELECT risk FROM readings WHERE pid=? ORDER BY ts DESC LIMIT 1", (pid,)).fetchone()
    anomaly = int(models()["iso"].predict(pd.DataFrame([[v[k] for k in VITALS]], columns=VITALS))[0] == -1)
    ts = time.time()
    alerts = [{**a, "source": "edge"} for a in edge_alerts(v)]
    if anomaly: alerts.append({"level": "warning", "msg": "ML anomaly in vitals pattern", "source": "cloud"})
    if risk > 0.7 and (prev is None or prev["risk"] <= 0.7):   # alert when it crosses, not on every reading
        alerts.append({"level": "critical", "msg": f"High heart-disease likelihood ({risk:.0%})", "source": "cloud"})
    with db() as c:
        c.execute("INSERT INTO readings(pid,ts,hr,spo2,sys_bp,dia_bp,temp,glucose,risk,anomaly) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (pid, ts, *(v[k] for k in VITALS), risk, anomaly))
        for a in alerts:
            c.execute("INSERT INTO alerts(pid,ts,level,msg,source) VALUES(?,?,?,?,?)",
                      (pid, ts, a["level"], a["msg"], a["source"]))
    return {"risk": risk, "anomaly": bool(anomaly), "alerts": alerts}

def twin_state(pid: str):
    with db() as c:
        p = c.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        if not p: return None
        last = c.execute("SELECT * FROM readings WHERE pid=? ORDER BY ts DESC LIMIT 1", (pid,)).fetchone()
        recent = c.execute("SELECT risk FROM readings WHERE pid=? ORDER BY ts DESC LIMIT 10", (pid,)).fetchall()
        alerts = c.execute("SELECT * FROM alerts WHERE pid=? ORDER BY ts DESC LIMIT 10", (pid,)).fetchall()
    risks = [r["risk"] for r in recent][::-1]
    trend = "stable"
    if len(risks) >= 4:
        d = np.mean(risks[-2:]) - np.mean(risks[:2])
        trend = "rising" if d > 0.05 else "falling" if d < -0.05 else "stable"
    return {"patient": {k: p[k] for k in ("id", "name", "age", "sex", "bmi")}, "latest": dict(last) if last else None,
            "risk_trend": trend, "alerts": [dict(a) for a in alerts]}

def history(pid: str, n=100):
    with db() as c:
        rows = c.execute("SELECT * FROM readings WHERE pid=? ORDER BY ts DESC LIMIT ?", (pid, n)).fetchall()
    return [dict(r) for r in rows][::-1]

def whatif(pid: str, changes: dict):
    """changes = deltas on the latest reading, e.g. {"glucose": -40, "sys_bp": -20, "chol": -30}."""
    s = twin_state(pid)
    if not s or not s["latest"]: return None
    with db() as c:
        p = c.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    base_v = {k: s["latest"][k] for k in VITALS}
    new_v = {k: base_v[k] + changes.get(k, 0) for k in VITALS}
    r0 = risk_of(model_input(p, base_v))
    r1 = risk_of(model_input(p, new_v, chol_delta=changes.get("chol", 0)))
    return {"baseline_risk": r0, "scenario_risk": r1, "delta": r1 - r0, "scenario": new_v}


# ---------- Explainability: exact Shapley values ----------
_N = len(FEATURES)
_MASKS = ((np.arange(2 ** _N)[:, None] >> np.arange(_N)) & 1).astype(bool)      # 8192 x 13 coalitions
_SIZE = _MASKS.sum(1)
from math import factorial as _f
_W = np.array([_f(k) * _f(_N - k - 1) / _f(_N) for k in range(_N)])             # weight of a coalition of size k

def _baseline():
    if "base" not in _models:
        with open(os.path.join(BASE, "ml", "models", "baseline.json")) as f:
            _models["base"] = json.load(f)
    return _models["base"]

def explain(pid: str):
    """Exact Shapley values: all 2^13 coalitions are scored in one batch; features outside a
    coalition are set to the median training patient. Sum(shap) == risk - baseline_risk."""
    s = twin_state(pid)
    if not s or not s["latest"]: return None
    with db() as c:
        p = c.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    x = model_input(p, {k: s["latest"][k] for k in VITALS})
    xv = np.array([float(x[f]) for f in FEATURES])
    base = _baseline(); bv = np.array([float(base[f]) for f in FEATURES])
    data = np.where(_MASKS, xv, bv)
    v = models()["rf"].predict_proba(pd.DataFrame(data, columns=FEATURES))[:, 1]   # value of every coalition
    phi = np.zeros(_N)
    for i in range(_N):
        without = np.where(~_MASKS[:, i])[0]
        with_i = without | (1 << i)
        phi[i] = np.sum(_W[_SIZE[without]] * (v[with_i] - v[without]))
    base_risk, risk = float(v[0]), float(v[-1])
    rows = sorted(({"feature": f, "value": float(xv[k]), "baseline": float(bv[k]), "shap": float(phi[k])}
                   for k, f in enumerate(FEATURES)), key=lambda r: -abs(r["shap"]))
    return {"baseline_risk": base_risk, "risk": risk, "check": float(phi.sum() - (risk - base_risk)), "contributions": rows}
