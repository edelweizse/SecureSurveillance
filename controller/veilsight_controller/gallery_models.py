from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GalleryIdentity(BaseModel):
    identity_key: str
    display_name: str | None = None
    active: bool = True
    embedding_count: int = 0
    created_at_ms: int | None = None
    updated_at_ms: int | None = None


class GalleryIdentityCreate(BaseModel):
    identity_key: str | None = None
    display_name: str
    active: bool = True


class GalleryIdentityUpdate(BaseModel):
    display_name: str | None = None
    active: bool | None = None


class GalleryEmbedding(BaseModel):
    id: int
    identity_key: str
    model: str = "mobilefacenet"
    dim: int = 128
    active: bool = True
    source_type: str | None = None
    source_ref: str | None = None
    quality: dict[str, Any] | None = None
    face_bbox: dict[str, Any] | None = None
    created_at_ms: int | None = None


class GalleryEmbeddingCreate(BaseModel):
    enrollment_id: str
    candidate_ids: list[str] = Field(default_factory=list)
    source_ref: str | None = None


class EnrollmentStreamCapture(BaseModel):
    stream_id: str
    profile: str = "ui"


class EnrollmentCandidatePublic(BaseModel):
    candidate_id: str
    image_id: str
    bbox: dict[str, float]
    landmarks: list[dict[str, float]] = Field(default_factory=list)
    score: float = 0.0
    image_width: int
    image_height: int
    usable: bool
    reject_reasons: list[str] = Field(default_factory=list)


class EnrollmentCandidateInternal(EnrollmentCandidatePublic):
    embedding: list[float] = Field(default_factory=list)


class EnrollmentCandidateResponse(BaseModel):
    enrollment_id: str
    expires_at_ms: int
    image_id: str
    candidates: list[EnrollmentCandidatePublic] = Field(default_factory=list)


class GalleryRefreshResponse(BaseModel):
    accepted: bool
    message: str
