from datetime import datetime, timezone

from app.config import get_settings
from app.vector_store import get_collection


def get_history_collection():
    settings = get_settings()
    chunks_collection = get_collection()
    db = chunks_collection.database
    return db["chat_history"]


def save_message(session_id: str, role: str, content: str) -> None:
    collection = get_history_collection()

    collection.insert_one(
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc),
        }
    )


def get_recent_history(session_id: str = "default", limit: int = 6) -> list[dict]:
    collection = get_history_collection()

    messages = list(
        collection.find(
            {"session_id": session_id},
            {"_id": 0, "role": 1, "content": 1, "created_at": 1},
        )
        .sort("created_at", -1)
        .limit(limit)
    )

    return list(reversed(messages))


def format_history_for_prompt(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."

    lines = []
    for message in history:
        role = message["role"].upper()
        content = message["content"]
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def clear_history(session_id: str = "default") -> int:
    collection = get_history_collection()
    result = collection.delete_many({"session_id": session_id})
    return result.deleted_count