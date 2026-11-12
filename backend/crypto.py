"""Encryption at rest for sensitive patient (EHR) fields. Key lives in backend/.secret_key (never commit it)."""
import os, json
from cryptography.fernet import Fernet

KEYFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".secret_key")

def _key() -> bytes:
    if os.path.exists(KEYFILE):
        return open(KEYFILE, "rb").read().strip()
    k = Fernet.generate_key()
    open(KEYFILE, "wb").write(k)
    return k

KEY = _key()
_f = Fernet(KEY)

def enc(d: dict) -> str:
    return _f.encrypt(json.dumps(d).encode()).decode()

def dec(s) -> dict:
    if not s: return {}
    try: return json.loads(_f.decrypt(s.encode()))
    except Exception:
        try: return json.loads(s)        # legacy plaintext rows
        except Exception: return {}
