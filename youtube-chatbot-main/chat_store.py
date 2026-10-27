"""PostgreSQL persistence for conversations, so they survive server restarts.

Only plain data is stored (metadata, transcript, messages). Search indexes are
not stored; app_yt rebuilds them from the saved transcript on first use.
"""
import json
import secrets

import psycopg
from psycopg.types.json import Jsonb


# Keys that are never saved: the search index and per-request debug state.
UNSAVED_KEYS = {"retriever"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id     TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    data        JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS chats_session_idx ON chats (session_id, created_at);
CREATE TABLE IF NOT EXISTS app_settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


class ChatStore:
    """Saves and restores chats. Every call opens a short-lived connection."""

    def __init__(self, database_url, retention_days=30):
        self.database_url = database_url
        with self._connect() as conn:
            conn.execute(SCHEMA)
            # Old conversations are removed so the table does not grow forever.
            conn.execute(
                "DELETE FROM chats WHERE updated_at < now() - make_interval(days => %s)",
                (retention_days,),
            )

    def _connect(self):
        return psycopg.connect(self.database_url, autocommit=True, connect_timeout=5)

    @staticmethod
    def _data(chat):
        data = {
            key: value for key, value in chat.items()
            if key not in UNSAVED_KEYS and not key.startswith("_")
        }
        # Round-trip through JSON so unexpected types fail here, not inside psycopg.
        return json.loads(json.dumps(data, ensure_ascii=False, default=str))

    def browser_secret(self):
        """A key that stays the same across restarts, for the browser's saved session id."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO app_settings (key, value) VALUES ('browser_state_secret', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (secrets.token_urlsafe(32),),
            )
            return conn.execute(
                "SELECT value FROM app_settings WHERE key = 'browser_state_secret'"
            ).fetchone()[0]

    def save_chat(self, session_id, chat):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chats (chat_id, session_id, data)
                VALUES (%s, %s, %s)
                ON CONFLICT (chat_id) DO UPDATE
                SET data = EXCLUDED.data, updated_at = now()
                WHERE chats.session_id = EXCLUDED.session_id
                """,
                (chat["id"], session_id, Jsonb(self._data(chat))),
            )

    def delete_chat(self, session_id, chat_id):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chats WHERE chat_id = %s AND session_id = %s",
                (chat_id, session_id),
            )

    def load_chats(self, session_id):
        """Return this session's chats, oldest first, without search indexes."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM chats WHERE session_id = %s ORDER BY created_at, chat_id",
                (session_id,),
            ).fetchall()
        return [dict(row[0], retriever=None) for row in rows]
