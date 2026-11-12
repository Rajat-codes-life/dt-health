from flask import Flask, jsonify, request, g
from flask_cors import CORS
import twin, auth

app = Flask(__name__)
CORS(app)
twin.init_db()
auth.init_users()
@app.get("/")
def home():
    return app.send_static_file("index.html")

@app.post("/api/login")
def login():
    b = request.get_json() or {}
    r, err = auth.login(b.get("username", ""), b.get("password", ""))
    if err == "locked": return jsonify(error="too many attempts, wait 60 seconds"), 429
    if err: return jsonify(error="wrong username or password"), 401
    return jsonify(r)

@app.post("/api/patients")                       # admin, or provisioning via device key
@auth.require(roles=["admin"], device_ok=True)
def add_patient():
    twin.upsert_patient(request.get_json())
    return jsonify(ok=True)

@app.get("/api/patients")
@auth.require()
def patients():
    with twin.db() as c:
        rows = c.execute("SELECT * FROM patients").fetchall()
    out = []
    for r in rows:
        if not auth.can_see(r["id"]): continue
        s = twin.twin_state(r["id"])
        out.append({"id": r["id"], "name": r["name"], "age": r["age"], "sex": r["sex"], "bmi": r["bmi"],
                    "latest": s["latest"], "risk_trend": s["risk_trend"]})   # EHR is never sent to the browser
    return jsonify(out)

@app.post("/api/vitals/<pid>")                   # IoT devices only
@auth.require(roles=[], device_ok=True)
def vitals(pid):
    try:
        return jsonify(twin.ingest(pid, request.get_json()))
    except KeyError:
        return jsonify(error="unknown patient"), 404

@app.get("/api/patients/<pid>/twin")
@auth.require()
def get_twin(pid):
    if not auth.can_see(pid): return jsonify(error="not allowed"), 403
    s = twin.twin_state(pid)
    return (jsonify(s), 200) if s else (jsonify(error="not found"), 404)

@app.get("/api/patients/<pid>/history")
@auth.require()
def get_history(pid):
    if not auth.can_see(pid): return jsonify(error="not allowed"), 403
    return jsonify(twin.history(pid, int(request.args.get("n", 100))))

@app.get("/api/patients/<pid>/explain")
@auth.require()
def get_explain(pid):
    if not auth.can_see(pid): return jsonify(error="not allowed"), 403
    r = twin.explain(pid)
    return (jsonify(r), 200) if r else (jsonify(error="no data yet"), 404)

@app.post("/api/patients/<pid>/whatif")          # clinical tool: doctors/admins only
@auth.require(roles=["doctor", "admin"])
def get_whatif(pid):
    r = twin.whatif(pid, request.get_json() or {})
    return (jsonify(r), 200) if r else (jsonify(error="no data yet"), 404)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
