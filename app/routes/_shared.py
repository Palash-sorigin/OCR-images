"""Upload handling and inference serialization shared by every OCR route.

IMPORTANT: pipeline_lock is a single, process-wide asyncio.Lock. All routes
(the original /ocr and /analyze, and the document routes in documents.py)
share the same PaddleOCR and VLM model instances (see app/dependencies.py),
so inference across ALL routes must serialize through this one lock -- a
route-local lock would only prevent concurrency within that route and still
let two different routes run model inference at the same time.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import HTTPException, UploadFile

from app.config import MAX_IMAGE_BYTES

pipeline_lock = asyncio.Lock()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff",
}


def validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format: {extension or 'unknown'}. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )
    return extension


def validate_content_type(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type for '{file.filename or 'image'}': {content_type}. "
                f"Expected an image file."
            ),
        )


async def save_upload(file: UploadFile) -> tuple[str, Path]:
    """Read an uploaded image into a temporary file.

    Returns (original_filename, temporary_file_path).
    """
    filename = Path(file.filename or "image").name
    extension = validate_extension(filename)
    validate_content_type(file)

    # Read one byte more than the allowed maximum so oversized files are
    # detected without loading an unbounded amount of data.
    data = await file.read(MAX_IMAGE_BYTES + 1)

    if not data:
        raise HTTPException(status_code=400, detail=f"Empty image upload: {filename}")

    if len(data) > MAX_IMAGE_BYTES:
        max_mb = MAX_IMAGE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the maximum allowed size of {max_mb:.1f} MB: {filename}",
        )

    # delete=False: the OCR service needs to reopen this file after we return.
    with NamedTemporaryFile(delete=False, suffix=extension) as temp:
        temp.write(data)
        return filename, Path(temp.name)
