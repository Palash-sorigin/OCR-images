from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import MAX_IMAGE_BYTES
from app.dependencies import get_ocr_service, get_pipeline


router = APIRouter(tags=["OCR / Extraction"])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

MIN_IMAGES = 1
MAX_IMAGES = 4


# ---------------------------------------------------------------------------
# IMPORTANT:
# Explicitly serialize OCR/model inference.
#
# Even if multiple HTTP requests arrive at the same time, only one OCR
# pipeline is allowed to execute at a time.
#
# Inside /analyze, images are also processed one-by-one.
# ---------------------------------------------------------------------------

_pipeline_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def _validate_extension(filename: str) -> str:
    """
    Validate the file extension and return it.
    """

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format: "
                f"{extension or 'unknown'}. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    return extension


def _validate_content_type(file: UploadFile) -> None:
    """
    Validate MIME type when the client provides one.

    Some clients may omit Content-Type, so an empty content type is allowed
    and the extension check remains the fallback.
    """

    content_type = (file.content_type or "").lower()

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type for "
                f"'{file.filename or 'image'}': {content_type}. "
                f"Expected an image file."
            ),
        )


# ---------------------------------------------------------------------------
# Save uploaded image temporarily
# ---------------------------------------------------------------------------

async def _save_upload(file: UploadFile) -> tuple[str, Path]:
    """
    Read an uploaded image into a temporary file.

    Returns:
        (original_filename, temporary_file_path)
    """

    filename = Path(file.filename or "image").name

    # Validate extension
    extension = _validate_extension(filename)

    # Validate MIME type
    _validate_content_type(file)

    # Read one byte more than the allowed maximum so that we can detect
    # oversized files without loading an unbounded amount of data.
    data = await file.read(MAX_IMAGE_BYTES + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"Empty image upload: {filename}",
        )

    if len(data) > MAX_IMAGE_BYTES:
        max_mb = MAX_IMAGE_BYTES / (1024 * 1024)

        raise HTTPException(
            status_code=413,
            detail=(
                f"Image exceeds the maximum allowed size "
                f"of {max_mb:.1f} MB: {filename}"
            ),
        )

    # Create a temporary file.
    #
    # delete=False is required because the OCR service needs to open the
    # file after this function returns.
    with NamedTemporaryFile(
        delete=False,
        suffix=extension,
    ) as temp:

        temp.write(data)

        return filename, Path(temp.name)


# ---------------------------------------------------------------------------
# Existing specialized container-number OCR
# ---------------------------------------------------------------------------

@router.post("/ocr")
async def detect_container_number(
    file: UploadFile = File(
        ...,
        description=(
            "Container image. The existing specialized container-number "
            "OCR pipeline will process this image."
        ),
    )
):
    """
    Backward-compatible specialized container-number OCR endpoint.

    This endpoint is intentionally kept separate from /analyze.

    /ocr
        Container image
            ↓
        Existing container OCR
            ↓
        ISO 6346 validation
            ↓
        Container-number result
    """

    filename, temp_path = await _save_upload(file)

    try:

        # Never run OCR/model inference concurrently.
        async with _pipeline_lock:

            result = get_ocr_service().process(temp_path)

        return result

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed for '{filename}': {exc}",
        ) from exc

    finally:

        # Always remove temporary file.
        temp_path.unlink(missing_ok=True)

        # Close UploadFile resources.
        await file.close()


# ---------------------------------------------------------------------------
# Complete sequential extraction pipeline
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_images(
    image_1: UploadFile | None = File(
        default=None,
        description="Portal screenshot/image 1"
    ),
    image_2: UploadFile | None = File(
        default=None,
        description="Portal screenshot/image 2"
    ),
    image_3: UploadFile | None = File(
        default=None,
        description="Portal screenshot/image 3"
    ),
    image_4: UploadFile | None = File(
        default=None,
        description="Portal screenshot/image 4"
    ),
):
    """
    Run the complete sequential:

        Extract → Normalize → Validate → Review

    pipeline.

    Up to 4 images can be uploaded.

    Images are processed strictly one at a time.
    """

    # ---------------------------------------------------------------
    # Collect only the files that were actually uploaded.
    # ---------------------------------------------------------------

    files = [
        file
        for file in (
            image_1,
            image_2,
            image_3,
            image_4,
        )
        if file is not None
    ]

    # ---------------------------------------------------------------
    # At least one image is required.
    # ---------------------------------------------------------------

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required.",
        )

    # ---------------------------------------------------------------
    # Save uploaded images.
    # ---------------------------------------------------------------

    saved: list[tuple[str, Path]] = []

    try:

        for file in files:

            saved.append(
                await _save_upload(file)
            )

        # -----------------------------------------------------------
        # IMPORTANT:
        #
        # Only ONE request can run the OCR pipeline at a time.
        # -----------------------------------------------------------

        async with _pipeline_lock:

            pipeline = get_pipeline()

            # -------------------------------------------------------
            # IMPORTANT:
            #
            # process_images() must process these sequentially.
            #
            # image 1
            #    ↓
            # complete extraction
            #    ↓
            # validation
            #    ↓
            # review
            #    ↓
            # image 2
            #    ↓
            # ...
            #
            # NO asyncio.gather()
            # NO parallel inference
            # -------------------------------------------------------

            image_results = pipeline.process_images(saved)

        # -----------------------------------------------------------
        # Reconcile the results from all screenshots.
        # -----------------------------------------------------------

        final = _reconcile(image_results)

        return final

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    finally:

        # -----------------------------------------------------------
        # Delete temporary files.
        # -----------------------------------------------------------

        for _, path in saved:
            path.unlink(missing_ok=True)

        # -----------------------------------------------------------
        # Close uploaded files.
        # -----------------------------------------------------------

        for file in files:

            try:
                await file.close()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Cross-image reconciliation
# ---------------------------------------------------------------------------

def _reconcile(image_results):
    """
    Compare extracted fields across all processed images.

    IMPORTANT:
    We never silently choose one value when two images disagree.

    Example:

        Image 1 → TT No = MH03FC0372
        Image 2 → TT No = MH03FC0370

    Result:

        CROSS_IMAGE_CONFLICT

    The user must manually review the field.
    """

    values_by_field: dict[str, list[dict]] = {}

    # -----------------------------------------------------------------------
    # Collect observations
    # -----------------------------------------------------------------------

    for image in image_results:

        for name, field in image.fields.items():

            value = field.value

            # Do not reconcile missing/empty values.
            if value is None or value == "":
                continue

            values_by_field.setdefault(name, []).append(
                {
                    "value": value,
                    "filename": image.filename,
                    "confidence": field.confidence,
                    "status": field.status,
                }
            )

    # -----------------------------------------------------------------------
    # Determine canonical values and conflicts
    # -----------------------------------------------------------------------

    conflicts = []
    canonical_data = {}

    for field_name, observations in values_by_field.items():

        unique_values = list(
            dict.fromkeys(
                str(observation["value"])
                for observation in observations
            )
        )

        # ---------------------------------------------------------------
        # Different images disagree.
        # ---------------------------------------------------------------

        if len(unique_values) > 1:

            conflicts.append(
                {
                    "type": "CROSS_IMAGE_CONFLICT",
                    "field": field_name,
                    "values": unique_values,
                    "observations": observations,
                    "reason": (
                        "Different images produced different values; "
                        "no value was silently selected."
                    ),
                }
            )

            continue

        # ---------------------------------------------------------------
        # All observations agree.
        # ---------------------------------------------------------------

        canonical_data[field_name] = observations[0]

    # -----------------------------------------------------------------------
    # Collect fields requiring manual review.
    # -----------------------------------------------------------------------

    manual_fields = []

    for image in image_results:

        for item in image.manual_review:

            manual_fields.append(
                {
                    "filename": image.filename,
                    **item,
                }
            )

    # Cross-image conflicts automatically require review.
    if conflicts:

        for conflict in conflicts:

            manual_fields.append(
                {
                    "field": conflict["field"],
                    "status": "CONFLICT",
                    "reason": conflict["reason"],
                }
            )

    # -----------------------------------------------------------------------
    # Overall status
    # -----------------------------------------------------------------------

    status = (
        "NEEDS_REVIEW"
        if manual_fields
        else "READY"
    )

    # -----------------------------------------------------------------------
    # Final response
    # -----------------------------------------------------------------------

    return {
        "status": status,

        "images": [
            image.model_dump()
            for image in image_results
        ],

        "data": _build_canonical_data(
            canonical_data
        ),

        "validation": {
            "status": (
                "NEEDS_REVIEW"
                if conflicts
                else "OK"
            ),
            "cross_image_conflicts": conflicts,
        },

        "manual_review": {
            "required": bool(manual_fields),
            "fields": manual_fields,
        },
    }


# ---------------------------------------------------------------------------
# Build final canonical structure
# ---------------------------------------------------------------------------

def _build_canonical_data(values: dict) -> dict:
    """
    Convert the flat extracted field dictionary into the structure that
    the future portal-filling layer can consume.
    """

    def get(name: str):
        return values.get(name)

    # -----------------------------------------------------------------------
    # Containers
    # -----------------------------------------------------------------------

    containers = []

    for suffix in ("1", "2"):

        number = (
            get(f"export_container_no_{suffix}")
            or get(f"dpd_container_no_{suffix}")
        )

        if number:

            containers.append(
                {
                    "number": number,
                    "origin": get(
                        f"container_origin_{suffix}"
                    ),
                    "booking_no": get(
                        f"booking_no_{suffix}"
                    ),
                    "via_no": get(
                        f"via_no_{suffix}"
                    ),
                    "genset": get(
                        f"genset_{suffix}"
                    ),
                }
            )

    # -----------------------------------------------------------------------
    # Final canonical object
    # -----------------------------------------------------------------------

    return {

        "pin": {
            "reference_no": get(
                "pin_reference_no"
            ),
            "number": get(
                "pin_no"
            ),
        },

        "truck": {
            "registration_no": get(
                "tt_no"
            ),
            "company": get(
                "truck_company"
            ),
        },

        "driver": {
            "name": get(
                "driver_name"
            ),
            "license_no": get(
                "driving_license_no"
            ),
            "mobile": get(
                "driver_mobile"
            ),
            "sms_mobile": get(
                "driver_sms_mobile"
            ),
        },

        "transaction": {
            "type": get(
                "transaction_type"
            ),
            "container_count": get(
                "container_count"
            ),
            "size": (
                get("size")
                or get("cont_size")
            ),
            "container_type": get(
                "container_type"
            ),
        },

        "containers": containers,

        "booking": {
            "booking_no": (
                get("booking_no")
                or get("booking_no_1")
            ),
            "via_no": get(
                "via_no_1"
            ),
            "import_pin_no": get(
                "import_pin_no"
            ),
            "dpd_container_no_1": get(
                "dpd_container_no_1"
            ),
            "dpd_container_no_2": get(
                "dpd_container_no_2"
            ),
        },

        "terminal": {
            "group_code": get(
                "group_code"
            ),
            "cfs_code": get(
                "cfs_code"
            ),
            "window": get(
                "window"
            ),
        },

        "operation": {
            "available_count": get(
                "available_count"
            ),
            "used_count": get(
                "used_count"
            ),
            "gate_start_time": get(
                "gate_start_time"
            ),
        },
    }