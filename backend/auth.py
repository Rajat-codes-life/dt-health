"""Authentication + role-based access control.
 - Users log in with username/password -> JWT (8h). Roles: admin, doctor, patient.
 - IoT devices/simulator authenticate with an API key (header X-Device-Key, env DEVICE_KEY).
 - Passwords stored as salted hashes; 5 wrong logins -> 60 s lockout.
"""
import os, time, hmac, hashlib, functools, jwt
from flask import request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash
import twin
from crypto import KEY

JWT_SECRET = hashlib.sha256(KEY + b"jwt").hexdigest()
DEVICE_KEY = os.environ.get("DEVICE_KEY", "dev-device-key")
_fails = {}   # username -> (count, locked_until)

DEMO_USERS = [  # CHANGE THESE for anything beyond a demo
    ("admin",  "admin123",   "admin",   None),
    ("drsingh", "doctor123", "doctor",  None),
    ("p001", "patient123", "patient", "P001"),
    ("p002", "patient123", "patient", "P002"),
    ("p003", "patient123", "patient", "P003"),
    ("p004", "patient123", "patient", "P004"),
]

def init_users():
    with twin.db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, pw TEXT, role TEXT, pid TEXT)")
        if not c.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            for u, pw, role, pid in DEMO_USERS:
                c.execute("INSERT INTO users VALUES(?,?,?,?)", (u, generate_password_hash(pw), role, pid))

def login(username, password):
    n, until = _fails.get(username, (0, 0))
    if time.time() < until: return None, "locked"
    with twin.db() as c:
        u = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not u or not check_password_hash(u["pw"], password or ""):
        n += 1; _fails[username] = (n, time.time() + 60 if n >= 5 else 0)
        return None, "bad"
    _fails.pop(username, None)
    tok = jwt.encode({"sub": u["username"], "role": u["role"], "pid": u["pid"],
                      "exp": int(time.time()) + 8 * 3600}, JWT_SECRET, algorithm="HS256")
    return {"token": tok, "user": {"username": u["username"], "role": u["role"], "pid": u["pid"]}}, None

def require(roles=None, device_ok=False):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            if device_ok and hmac.compare_digest(request.headers.get("X-Device-Key", ""), DEVICE_KEY):
                g.user = {"role": "device", "pid": None}
                return fn(*a, **k)
            h = request.headers.get("Authorization", "")
            if not h.startswith("Bearer "):
                return jsonify(error="login required"), 401
            try:
                p = jwt.decode(h[7:], JWT_SECRET, algorithms=["HS256"])
            except jwt.PyJWTError:
                return jsonify(error="invalid or expired token"), 401
            g.user = {"username": p["sub"], "role": p["role"], "pid": p["pid"]}
            if roles is not None and g.user["role"] not in roles:
                return jsonify(error="not allowed for your role"), 403
            return fn(*a, **k)
        return wrap
    return deco

def can_see(pid) -> bool:
    return g.user["role"] in ("admin", "doctor", "device") or g.user.get("pid") == pid
