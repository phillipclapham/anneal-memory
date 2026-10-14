"""Row: audit chain under concurrent writers (KL-24).

Three processes write 200 audited episodes each to one store. Does the chain still verify, and
does status() admit any dropped audit write?"""
import multiprocessing as mp
import os

from anneal_memory import Store
from anneal_memory.audit import AuditTrail

NAME = "audit_multiwriter"
# Direction: the KL-24 lock keeps one valid chain.
EXPECTED = [{"metric": "episodes", "op": "==", "value": 600}, {"metric": "verify_valid", "op": "==", "value": True}, {"metric": "status_write_failures", "op": "==", "value": 0},
            {"metric": "audit_entries", "op": ">=", "value": 600}, {"metric": "writer_failures", "op": "==", "value": 0}]
_WRITERS, _EACH = 3, 200


def _writer(db: str, n: int) -> None:
    with Store(db, audit=True) as st:
        for i in range(_EACH):
            st.record(f"writer {n} episode {i} about the deployment pipeline", "observation")


def run(tmpdir: str) -> dict:
    db = os.path.join(tmpdir, "multiwriter.db")
    Store(db, audit=True).close()
    ps = [mp.Process(target=_writer, args=(db, k)) for k in range(_WRITERS)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=300)
    hung = [p for p in ps if p.is_alive()]
    for p in hung:
        p.terminate()
        p.join()
    with Store(db, audit=True) as st:
        episodes = st.recall(limit=0).total_matching
        failures = getattr(st.status(), "audit_write_failures", None)
    r = AuditTrail.verify(db)
    out = {
        "episodes": episodes,
        "verify_valid": bool(getattr(r, "valid", False)),
        "status_write_failures": failures,
        "audit_entries": getattr(r, "total_entries", None),
        "writer_failures": sum(1 for p in ps if p.exitcode != 0),
    }
    for k in ("first_invalid_seq", "invalid_seq", "broken_at"):
        if hasattr(r, k):
            out[k] = getattr(r, k)
    return out
