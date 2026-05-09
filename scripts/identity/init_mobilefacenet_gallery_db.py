#!/usr/bin/env python3
"""Create a MobileFaceNet gallery DB and optionally import JSON embeddings."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Iterable


SCHEMA = """
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path, help="SQLite gallery database path to create/update")
    parser.add_argument(
        "--json",
        type=Path,
        help=(
            "Optional JSON embeddings. Accepts a list of objects with identity_key, "
            "display_name, embedding, or a mapping of identity_key to embedding/list."
        ),
    )
    parser.add_argument("--replace", action="store_true", help="Delete the existing DB before creating it")
    return parser.parse_args()


def as_embedding(values: object) -> list[float]:
    if not isinstance(values, list) or len(values) != 128:
        raise ValueError("each embedding must be a 128-value JSON array")
    out = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in out))
    if norm <= 0.0:
        raise ValueError("embedding has zero norm")
    return [v / norm for v in out]


def iter_records(payload: object) -> Iterable[tuple[str, str | None, list[float]]]:
    if isinstance(payload, dict) and isinstance(payload.get("identities"), list):
        payload = payload["identities"]

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("list entries must be objects")
            key = str(item["identity_key"])
            display_name = item.get("display_name")
            yield key, str(display_name) if display_name is not None else None, as_embedding(item["embedding"])
        return

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                display_name = value.get("display_name")
                embeddings = value.get("embeddings", value.get("embedding"))
            else:
                display_name = None
                embeddings = value

            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                for embedding in embeddings:
                    yield str(key), str(display_name) if display_name is not None else None, as_embedding(embedding)
            else:
                yield str(key), str(display_name) if display_name is not None else None, as_embedding(embeddings)
        return

    raise ValueError("unsupported JSON shape")


def main() -> int:
    args = parse_args()
    if args.replace and args.db_path.exists():
        args.db_path.unlink()
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(args.db_path) as db:
        db.executescript(SCHEMA)
        if args.json:
            payload = json.loads(args.json.read_text())
            for identity_key, display_name, embedding in iter_records(payload):
                db.execute(
                    "INSERT OR IGNORE INTO identities(identity_key, display_name, active) VALUES (?, ?, 1)",
                    (identity_key, display_name),
                )
                if display_name is not None:
                    db.execute(
                        "UPDATE identities SET display_name = ?, active = 1 WHERE identity_key = ?",
                        (display_name, identity_key),
                    )
                db.execute(
                    "INSERT INTO face_embeddings(identity_key, model, dim, embedding, active) VALUES (?, ?, ?, ?, 1)",
                    (identity_key, "mobilefacenet", 128, struct.pack("<128f", *embedding)),
                )
        db.commit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
