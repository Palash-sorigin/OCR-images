from typing import Any

from pydantic import BaseModel, Field


class FieldResult(BaseModel):
    value: Any = None
    confidence: float | None = None
    status: str = "NOT_FOUND"
    source: str = "SYSTEM"
    bbox: list[float] | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class ImageResult(BaseModel):
    filename: str
    status: str
    screen_type: str = "UNKNOWN"
    screen_confidence: float = 0.0
    # Populated only by the document routes (EIR, Form 13, Visit Ticket,
    # vehicle display, container seal); None for the portal/container
    # pipeline. Kept separate from screen_type rather than overloading it,
    # since screen_type has an established, specific meaning there
    # (PIN_GENERATION_PSA, etc.).
    document_type: str | None = None
    document_subtype: str | None = None
    quality: dict[str, Any] = Field(default_factory=dict)
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    manual_review: list[dict[str, Any]] = Field(default_factory=list)
    raw_ocr: list[dict[str, Any]] = Field(default_factory=list)
    container_ocr: dict[str, Any] | None = None
    operation_status: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    status: str
    images: list[ImageResult]
    data: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    manual_review: dict[str, Any] = Field(default_factory=dict)
