from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from controller.veilsight_controller.api import RunnerClientRegistry, router
from controller.veilsight_controller.gallery_service import GalleryService
from controller.veilsight_controller.gallery_store import GalleryStore


class GalleryTestSettings:
    auto_refresh = True
    enrollment_ttl_s = 1
    max_image_bytes = 1024 * 1024

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path


class FakeRunner:
    def __init__(self) -> None:
        self.reloads = 0

    def add_telemetry_callback(self, callback: Any) -> None:
        return None

    async def analyze_enrollment_image(self, image_bytes: bytes, mime_type: str) -> dict[str, Any]:
        return {
            "ok": True,
            "image_width": 100,
            "image_height": 100,
            "candidates": [
                {
                    "candidate_id": "face-0",
                    "bbox": {"x": 10, "y": 10, "w": 20, "h": 20},
                    "score": 0.95,
                    "image_width": 100,
                    "image_height": 100,
                    "usable": True,
                    "reject_reasons": [],
                    "embedding": [0.01] * 128,
                }
            ],
        }

    async def reload_gallery(self) -> dict[str, Any]:
        self.reloads += 1
        return {"accepted": True, "message": "gallery reloaded"}

    async def streams(self) -> dict[str, Any]:
        return {"public_base_url": "http://runner", "streams": [{"stream_id": "cam0", "profile": "ui"}]}


def test_gallery_db_migrates_minimal_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "gallery.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE identities (identity_key TEXT PRIMARY KEY, display_name TEXT, active INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE face_embeddings (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              identity_key TEXT NOT NULL,
              model TEXT NOT NULL DEFAULT 'mobilefacenet',
              dim INTEGER NOT NULL DEFAULT 128,
              embedding BLOB NOT NULL,
              active INTEGER NOT NULL DEFAULT 1
            );
            """
        )
    GalleryStore(db_path)
    with sqlite3.connect(db_path) as conn:
        identity_cols = {row[1] for row in conn.execute("PRAGMA table_info(identities)")}
        embedding_cols = {row[1] for row in conn.execute("PRAGMA table_info(face_embeddings)")}
    assert {"created_at_ms", "updated_at_ms"} <= identity_cols
    assert {"source_type", "source_ref", "quality_json", "face_bbox_json", "created_at_ms"} <= embedding_cols


def test_gallery_store_identity_and_embedding_lifecycle(tmp_path: Path) -> None:
    store = GalleryStore(tmp_path / "gallery.sqlite3")
    identity = store.create_identity("Alice Person", None)
    assert identity.identity_key == "alice-person"
    updated = store.update_identity(identity.identity_key, display_name="Alice", active=False)
    assert updated is not None
    assert updated.display_name == "Alice"
    assert updated.active is False
    store.update_identity(identity.identity_key, active=True)
    embedding = store.add_embedding(
        identity_key=identity.identity_key,
        embedding=[0.01] * 128,
        source_type="upload",
        source_ref="alice.jpg",
        quality={"usable": True},
        face_bbox={"x": 1, "y": 2, "w": 3, "h": 4},
    )
    assert store.list_embeddings(identity.identity_key)[0].id == embedding.id
    assert store.set_embedding_active(embedding.id, False).active is False  # type: ignore[union-attr]


def test_candidate_cache_expiry(tmp_path: Path) -> None:
    service = GalleryService(GalleryTestSettings(tmp_path / "gallery.sqlite3"))
    runner = FakeRunner()

    async def run() -> None:
        response = await service.analyze_image(
            image_bytes=b"fake",
            mime_type="image/jpeg",
            source_type="upload",
            source_ref=None,
            runner=runner,  # type: ignore[arg-type]
        )
        cached = service._enrollments[response.enrollment_id]
        cached.expires_at_ms = 1
        service.prune_expired()
        assert response.enrollment_id not in service._enrollments

    asyncio.run(run())


def test_api_commit_flow_with_fake_runner(tmp_path: Path) -> None:
    service = GalleryService(GalleryTestSettings(tmp_path / "gallery.sqlite3"))
    service.store.create_identity("Alice", "alice")
    fake = FakeRunner()
    previous_gallery = RunnerClientRegistry.gallery_service
    previous_runner = RunnerClientRegistry.client
    RunnerClientRegistry.gallery_service = service
    RunnerClientRegistry.client = fake  # type: ignore[assignment]
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    try:
        upload = client.post(
            "/api/gallery/enrollment-candidates",
            files={"file": ("alice.jpg", b"fake", "image/jpeg")},
        )
        assert upload.status_code == 200
        enrollment_id = upload.json()["enrollment_id"]
        commit = client.post(
            "/api/gallery/identities/alice/embeddings",
            json={"enrollment_id": enrollment_id, "candidate_ids": ["face-0"]},
        )
        assert commit.status_code == 200
        assert commit.json()[0]["identity_key"] == "alice"
        assert fake.reloads == 1
    finally:
        RunnerClientRegistry.gallery_service = previous_gallery
        RunnerClientRegistry.client = previous_runner


def test_stream_snapshot_rejects_unknown_stream(tmp_path: Path) -> None:
    service = GalleryService(GalleryTestSettings(tmp_path / "gallery.sqlite3"))
    fake = FakeRunner()
    previous_gallery = RunnerClientRegistry.gallery_service
    previous_runner = RunnerClientRegistry.client
    RunnerClientRegistry.gallery_service = service
    RunnerClientRegistry.client = fake  # type: ignore[assignment]
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    try:
        response = client.post(
            "/api/gallery/enrollment-candidates/from-stream",
            json={"stream_id": "missing", "profile": "ui"},
        )
        assert response.status_code == 404
    finally:
        RunnerClientRegistry.gallery_service = previous_gallery
        RunnerClientRegistry.client = previous_runner
