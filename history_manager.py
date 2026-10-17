import os
import json
import uuid
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)
MAX_SESSIONS = 20


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]


def save_session(session_id: str, topic: str, history: list) -> None:
    if not history:
        return
    data = {
        "session_id": session_id,
        "topic": topic,
        "saved_at": datetime.now().isoformat(),
        "preview": history[-1]["user_text"][:60] if history else "",
        "turn_count": len(history),
        "history": history,
    }
    filepath = HISTORY_DIR / f"{session_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_all_sessions() -> list:
    sessions = []
    files = sorted(HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    for fp in files[:MAX_SESSIONS]:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append(data)
        except Exception:
            pass
    return sessions


def delete_session(session_id: str) -> None:
    fp = HISTORY_DIR / f"{session_id}.json"
    if fp.exists():
        fp.unlink()
