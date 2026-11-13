"""Fake IoT wearables. Run: python simulator/simulator.py  (backend must be running)"""
import os, time, random, requests
API = os.environ.get("API_URL", "https://dt-health.onrender.com") + "/api"
sess = requests.Session()
sess.headers["X-Device-Key"] = os.environ.get("DEVICE_KEY", "dev-device-key")   # device credential
# EHR fields below are real rows taken from the UCI heart dataset (static record), vitals are simulated live.
PATIENTS = [
    {"id": "P001", "name": "Ramesh Kumar", "age": 67, "sex": 1, "bmi": 29.5,
     "ehr": {"cp": 0, "chol": 286, "restecg": 0, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 1, "ca": 3, "thal": 2}},
    {"id": "P002", "name": "Sunita Devi", "age": 54, "sex": 0, "bmi": 31.0,
     "ehr": {"cp": 2, "chol": 267, "restecg": 0, "thalach": 167, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2}},
    {"id": "P003", "name": "Aman Verma", "age": 29, "sex": 1, "bmi": 23.0,
     "ehr": {"cp": 1, "chol": 204, "restecg": 0, "thalach": 202, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2}},
    {"id": "P004", "name": "Kavita Singh", "age": 60, "sex": 0, "bmi": 27.0,
     "ehr": {"cp": 0, "chol": 258, "restecg": 0, "thalach": 157, "exang": 0, "oldpeak": 2.6, "slope": 1, "ca": 2, "thal": 3}},
]
for p in PATIENTS: sess.post(f"{API}/patients", json=p, timeout=60)
state = {p["id"]: {"hr": 76, "spo2": 97, "sys_bp": 122, "dia_bp": 80, "temp": 36.8, "glucose": 105} for p in PATIENTS}
episode = {}   # pid -> ticks left in a deterioration episode

def step(pid):
    s = state[pid]
    if pid not in episode and random.random() < 0.03: episode[pid] = 12
    sick = pid in episode
    if sick:
        episode[pid] -= 1
        if episode[pid] <= 0: del episode[pid]
    t = {"hr": 118 if sick else 76, "spo2": 90 if sick else 97, "sys_bp": 158 if sick else 122,
         "dia_bp": 98 if sick else 80, "temp": 38.2 if sick else 36.8, "glucose": 190 if sick else 105}
    for k in s:
        s[k] += (t[k] - s[k]) * 0.25 + random.gauss(0, {"temp": .08, "spo2": .4}.get(k, 2.0))
    s["spo2"] = min(100, s["spo2"])
    return {k: round(v, 1) for k, v in s.items()}

while True:
    for p in PATIENTS:
        try:
            r = sess.post(f"{API}/vitals/{p['id']}", json=step(p["id"]), timeout=60)
            r.raise_for_status()
            r = r.json()
            print(p["id"], f"risk={r['risk']:.0%}", [a["msg"] for a in r["alerts"]])
        except Exception as e:
            print(p["id"], "failed:", e)
    time.sleep(2)