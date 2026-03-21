import json
import sqlite3
from typing import Dict, List, Optional

from app.config import settings
from app.models import ChunkRecord, UserContext
from app.utils import ensure_dir


class MetadataStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.sqlite_path
        ensure_dir(settings.storage_dir)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                page INTEGER,
                section TEXT,
                extra_metadata TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)"
        )
        self.conn.commit()

    def reset(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    def insert_chunks(self, chunks: List[ChunkRecord]) -> None:
        rows = [
            (
                c.chunk_id,
                c.source,
                c.title,
                c.text,
                c.page,
                c.section,
                json.dumps(c.extra_metadata),
            )
            for c in chunks
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, source, title, text, page, section, extra_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            source=row["source"],
            title=row["title"],
            text=row["text"],
            page=row["page"],
            section=row["section"],
            extra_metadata=json.loads(row["extra_metadata"] or "{}"),
        )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkRecord]:
        cur = self.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        return self._row_to_chunk(row) if row else None

    def get_all_chunks(self) -> List[ChunkRecord]:
        cur = self.conn.execute("SELECT * FROM chunks")
        return [self._row_to_chunk(row) for row in cur.fetchall()]

    def filter_chunk(self, chunk: ChunkRecord, filters: Dict) -> bool:
        if not filters:
            return True

        for key, value in filters.items():
            if key == "source" and chunk.source != value:
                return False
            elif key == "title" and chunk.title != value:
                return False
            elif key == "page" and chunk.page != value:
                return False
            elif key not in {"source", "title", "page"}:
                if chunk.extra_metadata.get(key) != value:
                    return False
        return True

    def user_can_access_chunk(self, chunk: ChunkRecord, user: UserContext) -> bool:
        if user.is_admin:
            return True

        allowed_users = chunk.extra_metadata.get("allowed_users", [])
        allowed_roles = chunk.extra_metadata.get("allowed_roles", [])
        visibility = chunk.extra_metadata.get("visibility", "private")

        if visibility == "public":
            return True
        if allowed_users and user.user_id in allowed_users:
            return True
        if allowed_roles and set(user.roles).intersection(set(allowed_roles)):
            return True
        return not allowed_users and not allowed_roles and visibility == "internal"