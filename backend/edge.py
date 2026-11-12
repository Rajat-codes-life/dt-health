"""Simulated EDGE layer: fast rule-based alerts, no ML, runs on every reading."""
RULES = [
    ("spo2",   lambda v: v < 92,  "critical", "SpO2 low"),
    ("hr",     lambda v: v > 120, "warning",  "Tachycardia"),
    ("hr",     lambda v: v < 50,  "warning",  "Bradycardia"),
    ("sys_bp", lambda v: v > 160, "critical", "Hypertensive reading"),
    ("temp",   lambda v: v > 38.5,"warning",  "Fever"),
    ("glucose",lambda v: v > 250, "critical", "Hyperglycemia"),
    ("glucose",lambda v: v < 65,  "critical", "Hypoglycemia"),
]
def edge_alerts(v: dict):
    return [{"level": lvl, "msg": msg, "field": f, "value": v[f]}
            for f, cond, lvl, msg in RULES if f in v and cond(v[f])]
