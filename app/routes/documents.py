"""Routes for the fixed-schema document types: EIR, Form 13, Visit Ticket,
vehicle dashboard display, and container seal. One route per document type,
each backed by its own DocumentExtractionPipeline instance (same VLM + OCR
engines, different field schema) -- see app/services/document_pipeline.py
and app/config/document_schemas.py.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.dependencies import get_document_pipeline
from app.routes._shared import pipeline_lock, save_upload

router = APIRouter(tags=["Document OCR / Extraction"])

MAX_IMAGES = 4


async def _handle_upload(files: list[UploadFile], document_type: str) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if len(files) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_IMAGES} images are allowed per request.")

    saved: list[tuple[str, Path]] = []
    try:
        for file in files:
            saved.append(await save_upload(file))

        # Same process-wide lock as /ocr and /analyze: all routes share the
        # underlying PaddleOCR/VLM model instances, so inference across
        # every route must serialize through one lock, not a route-local one.
        async with pipeline_lock:
            pipeline = get_document_pipeline(document_type)
            image_results = pipeline.process_images(saved)

        manual_fields = [
            {"filename": image.filename, **item}
            for image in image_results
            for item in image.manual_review
        ]
        status = "NEEDS_REVIEW" if manual_fields else "READY"

        return {
            "status": status,
            "document_type": document_type,
            "images": [image.model_dump() for image in image_results],
            "manual_review": {"required": bool(manual_fields), "fields": manual_fields},
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{document_type} analysis failed: {exc}") from exc
    finally:
        for _, path in saved:
            path.unlink(missing_ok=True)
        for file in files:
            try:
                await file.close()
            except Exception:
                pass


@router.post("/ocr/eir")
async def analyze_eir(
    files: list[UploadFile] = File(..., description="One or more EIR (Equipment Interchange Receipt) photos."),
):
    """Equipment Interchange Report/Receipt extraction (NSFT and DP World layouts)."""
    return await _handle_upload(files, "eir")


@router.post("/ocr/form13")
async def analyze_form13(
    files: list[UploadFile] = File(..., description="One or more Form 13 / E-Gate Pass photos."),
):
    """Form 13 (E-Gate Pass / Form 6 / SEZ 4) extraction."""
    return await _handle_upload(files, "form13")


@router.post("/ocr/visit-ticket")
async def analyze_visit_ticket(
    files: list[UploadFile] = File(..., description="One or more visit ticket / drop-off / pick-up pass photos."),
):
    """Terminal visit ticket extraction (Gateway Terminals, DP World NSICT, PSA BMCT)."""
    return await _handle_upload(files, "visit_ticket")


@router.post("/ocr/vehicle-display")
async def analyze_vehicle_display(
    files: list[UploadFile] = File(..., description="One or more vehicle dashboard (KM/battery) photos."),
):
    """EV dashboard reading extraction (SOC %, voltage, odometer, range). Best-effort: see README."""
    return await _handle_upload(files, "vehicle_display")


@router.post("/ocr/container-seal")
async def analyze_container_seal(
    files: list[UploadFile] = File(..., description="One or more container seal close-up photos."),
):
    """Container seal number extraction. Relies primarily on VLM due to photo rotation."""
    return await _handle_upload(files, "container_seal")
