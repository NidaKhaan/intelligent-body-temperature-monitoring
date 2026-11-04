import json, os, time
from pathlib import Path

def atomic_write_json(path, data):
    """Temp file mein likho, phir replace: crash par purani file safe rehti hai."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

class StageTracker:
    """Har stage ka status Drive par save karta hai; reconnect par done stages skip."""
    def __init__(self, path):
        self.path = Path(path)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    def is_done(self, stage):
        return self.state.get(stage, {}).get("status") == "done"

    def mark_done(self, stage, **meta):
        self.state[stage] = {"status": "done",
                             "time": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
        atomic_write_json(self.path, self.state)

    def run(self, stage, fn, *args, **kwargs):
        if self.is_done(stage):
            print(f"[skip] {stage} (already done)")
            return
        print(f"[run]  {stage}")
        fn(*args, **kwargs)
        self.mark_done(stage)
