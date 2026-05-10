from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException

from .gallery_models import (
    EnrollmentCandidateInternal,
    EnrollmentCandidatePublic,
    EnrollmentCandidateResponse,
    GalleryEmbedding,
)
from .gallery_store import GalleryStore, now_ms
from .runner_client import RunnerClient
from .settings import GallerySettings


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@dataclass
class CachedEnrollment:
    enrollment_id: str
    created_at_ms: int
    expires_at_ms: int
    mime_type: str
    source_type: str
    source_ref: str | None
    images: dict[str, bytes]
    candidates: dict[str, EnrollmentCandidateInternal] = field(default_factory=dict)


class GalleryService:
    def __init__(self, gallery_settings: GallerySettings, store: GalleryStore | None = None) -> None:
        self.settings = gallery_settings
        self.store = store or GalleryStore(gallery_settings.db_path)
        self._enrollments: dict[str, CachedEnrollment] = {}

    def prune_expired(self) -> None:
        ts = now_ms()
        expired = [key for key, value in self._enrollments.items() if value.expires_at_ms <= ts]
        for key in expired:
            self._enrollments.pop(key, None)

    async def analyze_image(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        source_type: str,
        source_ref: str | None,
        runner: RunnerClient,
    ) -> EnrollmentCandidateResponse:
        self._validate_image(image_bytes, mime_type)
        raw = await runner.analyze_enrollment_image(image_bytes=image_bytes, mime_type=mime_type)
        return self._cache_runner_candidates(raw, image_bytes, mime_type, source_type, source_ref)

    async def analyze_stream_snapshot(
        self,
        *,
        stream_id: str,
        profile: str,
        runner: RunnerClient,
        snapshot_url: str,
    ) -> EnrollmentCandidateResponse:
        image_bytes, mime_type = await asyncio.to_thread(self._fetch_snapshot, snapshot_url)
        return await self.analyze_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            source_type="stream_snapshot",
            source_ref=f"{stream_id}/{profile}",
            runner=runner,
        )

    def get_image(self, enrollment_id: str, image_id: str) -> tuple[bytes, str]:
        self.prune_expired()
        cached = self._enrollments.get(enrollment_id)
        if cached is None:
            raise HTTPException(status_code=404, detail={"error": "enrollment_not_found"})
        image = cached.images.get(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail={"error": "enrollment_image_not_found"})
        return image, cached.mime_type

    async def commit_candidates(
        self,
        *,
        identity_key: str,
        enrollment_id: str,
        candidate_ids: list[str],
        source_ref: str | None,
        runner: RunnerClient,
    ) -> list[GalleryEmbedding]:
        self.prune_expired()
        cached = self._enrollments.get(enrollment_id)
        if cached is None:
            raise HTTPException(status_code=410, detail={"error": "enrollment_candidate_expired"})
        if not candidate_ids:
            raise HTTPException(status_code=400, detail={"error": "candidate_ids_required"})

        created: list[GalleryEmbedding] = []
        for candidate_id in candidate_ids:
            candidate = cached.candidates.get(candidate_id)
            if candidate is None:
                raise HTTPException(status_code=404, detail={"error": "candidate_not_found", "candidate_id": candidate_id})
            if not candidate.usable:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "candidate_low_quality", "candidate_id": candidate_id, "reject_reasons": candidate.reject_reasons},
                )
            try:
                created.append(
                    self.store.add_embedding(
                        identity_key=identity_key,
                        embedding=candidate.embedding,
                        source_type=cached.source_type,
                        source_ref=source_ref or cached.source_ref,
                        quality={
                            "usable": candidate.usable,
                            "reject_reasons": candidate.reject_reasons,
                            "score": candidate.score,
                        },
                        face_bbox=candidate.bbox,
                    )
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail={"error": "identity_not_found"}) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail={"error": "invalid_embedding", "detail": str(exc)}) from exc

        self._enrollments.pop(enrollment_id, None)
        if self.settings.auto_refresh:
            await runner.reload_gallery()
        return created

    def _validate_image(self, image_bytes: bytes, mime_type: str) -> None:
        clean_type = mime_type.split(";")[0].strip().lower()
        if clean_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail={"error": "unsupported_image_type"})
        if len(image_bytes) > self.settings.max_image_bytes:
            raise HTTPException(status_code=413, detail={"error": "image_too_large"})
        if not image_bytes:
            raise HTTPException(status_code=400, detail={"error": "empty_image"})

    def _cache_runner_candidates(
        self,
        raw: dict[str, Any],
        image_bytes: bytes,
        mime_type: str,
        source_type: str,
        source_ref: str | None,
    ) -> EnrollmentCandidateResponse:
        self.prune_expired()
        enrollment_id = uuid.uuid4().hex
        image_id = "source"
        created_at = now_ms()
        expires_at = created_at + self.settings.enrollment_ttl_s * 1000
        cached = CachedEnrollment(
            enrollment_id=enrollment_id,
            created_at_ms=created_at,
            expires_at_ms=expires_at,
            mime_type=mime_type.split(";")[0].strip().lower(),
            source_type=source_type,
            source_ref=source_ref,
            images={image_id: image_bytes},
        )

        width = int(raw.get("image_width") or raw.get("imageWidth") or 0)
        height = int(raw.get("image_height") or raw.get("imageHeight") or 0)
        public_candidates: list[EnrollmentCandidatePublic] = []
        for index, item in enumerate(raw.get("candidates", [])):
            candidate_id = str(item.get("candidate_id") or item.get("candidateId") or f"face-{index}")
            bbox = self._rect(item.get("bbox") or {})
            landmarks = [self._point(point) for point in item.get("landmarks", [])]
            internal = EnrollmentCandidateInternal(
                candidate_id=candidate_id,
                image_id=image_id,
                bbox=bbox,
                landmarks=landmarks,
                score=float(item.get("score", 0.0)),
                image_width=int(item.get("image_width") or item.get("imageWidth") or width),
                image_height=int(item.get("image_height") or item.get("imageHeight") or height),
                usable=bool(item.get("usable", False)),
                reject_reasons=[str(reason) for reason in item.get("reject_reasons", item.get("rejectReasons", []))],
                embedding=[float(v) for v in item.get("embedding", [])],
            )
            cached.candidates[candidate_id] = internal
            public_candidates.append(EnrollmentCandidatePublic(**internal.model_dump(exclude={"embedding"})))

        self._enrollments[enrollment_id] = cached
        return EnrollmentCandidateResponse(
            enrollment_id=enrollment_id,
            expires_at_ms=expires_at,
            image_id=image_id,
            candidates=public_candidates,
        )

    @staticmethod
    def _fetch_snapshot(snapshot_url: str) -> tuple[bytes, str]:
        try:
            request = Request(snapshot_url, headers={"Accept": "image/jpeg,image/png,image/webp"})
            with urlopen(request, timeout=5.0) as response:
                mime_type = response.headers.get_content_type() or "image/jpeg"
                return response.read(), mime_type
        except URLError as exc:
            raise HTTPException(status_code=502, detail={"error": "snapshot_fetch_failed", "detail": str(exc)}) from exc

    @staticmethod
    def _rect(value: dict[str, Any]) -> dict[str, float]:
        return {
            "x": float(value.get("x", 0.0)),
            "y": float(value.get("y", 0.0)),
            "w": float(value.get("w", value.get("width", 0.0))),
            "h": float(value.get("h", value.get("height", 0.0))),
        }

    @staticmethod
    def _point(value: dict[str, Any]) -> dict[str, float]:
        return {"x": float(value.get("x", 0.0)), "y": float(value.get("y", 0.0))}
