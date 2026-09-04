"""Quick single-writer tamper probe — verify the audit chain works as
designed when used in its supported configuration."""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "Documents" / "anneal-memory"))
from anneal_memory.store import Store, EpisodeType
from anneal_memory.audit import AuditTrail

ROOT = Path("/tmp/bold-stand-phase1/store_attack3b")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
db = ROOT / "single.db"

# Single-writer happy path
store = Store(path=db, project_name="single_writer")
for i in range(3):
    store.record(f"single writer episode {i}", EpisodeType.OBSERVATION, source="single")

baseline = AuditTrail.verify(db)
print(f"Single-writer clean chain valid: {baseline.valid}, entries: {baseline.total_entries}")

# Now tamper
audit_file = next(ROOT.glob("*.audit.jsonl"))
original = audit_file.read_text()
lines = original.strip().split("\n")
entry = json.loads(lines[1])
old = entry.get("actor", "single")
entry["actor"] = "TAMPERED"
lines[1] = json.dumps(entry)
audit_file.write_text("\n".join(lines) + "\n")

tampered = AuditTrail.verify(db)
print(f"After tampering (line 1 actor {old!r}->'TAMPERED'): valid={tampered.valid}")
if not tampered.valid:
    print(f"   detected at seq {tampered.chain_break_at}, error: {tampered.error[:80]}")

# Restore
audit_file.write_text(original)
restored = AuditTrail.verify(db)
print(f"After restore: valid={restored.valid}")
