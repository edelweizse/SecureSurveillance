from __future__ import annotations

import json
import re
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

from .gallery_models import GalleryEmbedding, GalleryIdentity


SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,126}$")


def now_ms() -> int:
    return int(time.time() * 1000)


def slugify_identity(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-_")
    if not slug:
        slug = f"identity-{now_ms()}"
    return slug[:127]


def validate_identity_key(identity_key: str) -> str:
    if not SLUG_RE.fullmatch(identity_key):
        raise ValueError("identity_key must be slug-safe: lowercase letters, digits, '-' or '_'")
    return identity_key


def encode_embedding(values: list[float]) -> bytes:
    if len(values) != 128:
        raise ValueError("MobileFaceNet embeddings must have exactly 128 dimensions")
    return struct.pack("<128f", *(float(v) for v in values))


class GalleryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def migrate(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS identities (
                  identity_key TEXT PRIMARY KEY,
                  display_name TEXT,
                  active INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  identity_key TEXT NOT NULL,
                  model TEXT NOT NULL DEFAULT 'mobilefacenet',
                  dim INTEGER NOT NULL DEFAULT 128,
                  embedding BLOB NOT NULL,
                  active INTEGER NOT NULL DEFAULT 1,
                  FOREIGN KEY(identity_key) REFERENCES identities(identity_key)
                );
                """
            )
            self._add_column(conn, "identities", "created_at_ms", "INTEGER")
            self._add_column(conn, "identities", "updated_at_ms", "INTEGER")
            self._add_column(conn, "face_embeddings", "created_at_ms", "INTEGER")
            self._add_column(conn, "face_embeddings", "source_type", "TEXT")
            self._add_column(conn, "face_embeddings", "source_ref", "TEXT")
            self._add_column(conn, "face_embeddings", "quality_json", "TEXT")
            self._add_column(conn, "face_embeddings", "face_bbox_json", "TEXT")
            ts = now_ms()
            conn.execute("UPDATE identities SET created_at_ms = COALESCE(created_at_ms, ?)", (ts,))
            conn.execute("UPDATE identities SET updated_at_ms = COALESCE(updated_at_ms, created_at_ms, ?)", (ts,))
            conn.execute("UPDATE face_embeddings SET created_at_ms = COALESCE(created_at_ms, ?)", (ts,))
            conn.commit()

    @staticmethod
    def _add_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def list_identities(self) -> list[GalleryIdentity]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT i.identity_key, i.display_name, i.active, i.created_at_ms, i.updated_at_ms,
                       COALESCE(SUM(CASE WHEN fe.active = 1 THEN 1 ELSE 0 END), 0) AS embedding_count
                FROM identities i
                LEFT JOIN face_embeddings fe
                  ON fe.identity_key = i.identity_key
                 AND fe.model = 'mobilefacenet'
                 AND fe.dim = 128
                GROUP BY i.identity_key
                ORDER BY lower(COALESCE(i.display_name, i.identity_key)), i.identity_key
                """
            ).fetchall()
        return [self._identity(row) for row in rows]

    def create_identity(self, display_name: str, identity_key: str | None, active: bool = True) -> GalleryIdentity:
        key = validate_identity_key(identity_key or slugify_identity(display_name))
        ts = now_ms()
        with self.connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO identities(identity_key, display_name, active, created_at_ms, updated_at_ms)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, display_name, int(active), ts, ts),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError("identity_key already exists") from exc
        found = self.get_identity(key)
        if found is None:
            raise RuntimeError("created identity could not be loaded")
        return found

    def get_identity(self, identity_key: str) -> GalleryIdentity | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT i.identity_key, i.display_name, i.active, i.created_at_ms, i.updated_at_ms,
                       COALESCE(SUM(CASE WHEN fe.active = 1 THEN 1 ELSE 0 END), 0) AS embedding_count
                FROM identities i
                LEFT JOIN face_embeddings fe
                  ON fe.identity_key = i.identity_key
                 AND fe.model = 'mobilefacenet'
                 AND fe.dim = 128
                WHERE i.identity_key = ?
                GROUP BY i.identity_key
                """,
                (identity_key,),
            ).fetchone()
        return self._identity(row) if row else None

    def update_identity(
        self,
        identity_key: str,
        *,
        display_name: str | None = None,
        active: bool | None = None,
    ) -> GalleryIdentity | None:
        validate_identity_key(identity_key)
        fields: list[str] = []
        values: list[Any] = []
        if display_name is not None:
            fields.append("display_name = ?")
            values.append(display_name)
        if active is not None:
            fields.append("active = ?")
            values.append(int(active))
        if fields:
            fields.append("updated_at_ms = ?")
            values.append(now_ms())
            values.append(identity_key)
            with self.connect() as conn:
                conn.execute(f"UPDATE identities SET {', '.join(fields)} WHERE identity_key = ?", values)
                conn.commit()
        return self.get_identity(identity_key)

    def soft_delete_identity(self, identity_key: str) -> bool:
        updated = self.update_identity(identity_key, active=False)
        return updated is not None

    def delete_identity(self, identity_key: str) -> bool:
        validate_identity_key(identity_key)
        with self.connect() as conn:
            if not conn.execute("SELECT 1 FROM identities WHERE identity_key = ?", (identity_key,)).fetchone():
                return False
            conn.execute("DELETE FROM face_embeddings WHERE identity_key = ?", (identity_key,))
            conn.execute("DELETE FROM identities WHERE identity_key = ?", (identity_key,))
            conn.commit()
        return True

    def ensure_identity_active(self, identity_key: str) -> GalleryIdentity | None:
        validate_identity_key(identity_key)
        identity = self.get_identity(identity_key)
        if identity is None:
            return None
        if not identity.active:
            with self.connect() as conn:
                conn.execute(
                    "UPDATE identities SET active = 1, updated_at_ms = ? WHERE identity_key = ?",
                    (now_ms(), identity_key),
                )
                conn.commit()
            identity = self.get_identity(identity_key)
        return identity

    def list_embeddings(self, identity_key: str) -> list[GalleryEmbedding]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, identity_key, model, dim, active, source_type, source_ref,
                       quality_json, face_bbox_json, created_at_ms
                FROM face_embeddings
                WHERE identity_key = ?
                ORDER BY id DESC
                """,
                (identity_key,),
            ).fetchall()
        return [self._embedding(row) for row in rows]

    def add_embedding(
        self,
        *,
        identity_key: str,
        embedding: list[float],
        source_type: str,
        source_ref: str | None,
        quality: dict[str, Any],
        face_bbox: dict[str, Any],
    ) -> GalleryEmbedding:
        validate_identity_key(identity_key)
        blob = encode_embedding(embedding)
        ts = now_ms()
        with self.connect() as conn:
            if not conn.execute("SELECT 1 FROM identities WHERE identity_key = ?", (identity_key,)).fetchone():
                raise KeyError(identity_key)
            cur = conn.execute(
                """
                INSERT INTO face_embeddings(
                  identity_key, model, dim, embedding, active, created_at_ms,
                  source_type, source_ref, quality_json, face_bbox_json
                )
                VALUES (?, 'mobilefacenet', 128, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    identity_key,
                    blob,
                    ts,
                    source_type,
                    source_ref,
                    json.dumps(quality, separators=(",", ":")),
                    json.dumps(face_bbox, separators=(",", ":")),
                ),
            )
            conn.execute("UPDATE identities SET updated_at_ms = ? WHERE identity_key = ?", (ts, identity_key))
            conn.commit()
            embedding_id = int(cur.lastrowid)
            row = conn.execute(
                """
                SELECT id, identity_key, model, dim, active, source_type, source_ref,
                       quality_json, face_bbox_json, created_at_ms
                FROM face_embeddings
                WHERE id = ?
                """,
                (embedding_id,),
            ).fetchone()
        return self._embedding(row)

    def set_embedding_active(self, embedding_id: int, active: bool) -> GalleryEmbedding | None:
        with self.connect() as conn:
            conn.execute("UPDATE face_embeddings SET active = ? WHERE id = ?", (int(active), embedding_id))
            row = conn.execute(
                """
                SELECT id, identity_key, model, dim, active, source_type, source_ref,
                       quality_json, face_bbox_json, created_at_ms
                FROM face_embeddings
                WHERE id = ?
                """,
                (embedding_id,),
            ).fetchone()
            conn.commit()
        return self._embedding(row) if row else None

    @staticmethod
    def _identity(row: sqlite3.Row) -> GalleryIdentity:
        return GalleryIdentity(
            identity_key=row["identity_key"],
            display_name=row["display_name"],
            active=bool(row["active"]),
            embedding_count=int(row["embedding_count"] or 0),
            created_at_ms=row["created_at_ms"],
            updated_at_ms=row["updated_at_ms"],
        )

    @staticmethod
    def _embedding(row: sqlite3.Row) -> GalleryEmbedding:
        quality = json.loads(row["quality_json"]) if row["quality_json"] else None
        face_bbox = json.loads(row["face_bbox_json"]) if row["face_bbox_json"] else None
        return GalleryEmbedding(
            id=int(row["id"]),
            identity_key=row["identity_key"],
            model=row["model"],
            dim=int(row["dim"]),
            active=bool(row["active"]),
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            quality=quality,
            face_bbox=face_bbox,
            created_at_ms=row["created_at_ms"],
        )
