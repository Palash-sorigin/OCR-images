# Container & Portal Screenshot OCR API

FastAPI service for a sequential **Extract → Validate → Review → Fill** workflow.

The original project was a container-number OCR service. It now keeps that specialized container OCR as a reusable component and adds general portal screenshot extraction for PIN Generation and Truck Booking screens.

## Core design

```text
                         Uploaded images
                               |
                               v
                    +---------------------+
                    | Image 1             |
                    +----------+----------+
                               |
                               v
                    Image quality check
                               |
                               v
                    Screen classification
                               |
                               v
                       General PaddleOCR
                               |
                               v
                       Field extraction
                               |
                               v
                   Specialized container OCR
                               |
                               v
                         Normalization
                               |
                               v
                     Field validation
                               |
                               v
                   Cross-field validation
                               |
                               v
                    Confidence assessment
                               |
                         +-----+-----+
                         |           |
                       clear     uncertain
                         |           |
                         |           v
                         |       Optional VLM
                         |           |
                         +-----+-----+
                               |
                               v
                         Manual review
                         only if needed
                               |
                               v
                        Finish image 1
                               |
                               v
                    +---------------------+
                    | Image 2             |
                    +---------------------+
                               |
                              ...
                               |
                               v
                       All images finished
                               |
                               v
                   Cross-image reconciliation
                               |
                               v
                          Final JSON
```

### Sequential processing is intentional

The service does **not** use parallel image processing. There is no `asyncio.gather`, worker pool, multiprocessing, Celery, queue, or background OCR job.

For four images:

```text
Image 1 → complete
Image 2 → complete
Image 3 → complete
Image 4 → complete
                  ↓
          reconcile all images
```

HTTP requests are also serialized around the OCR pipeline so two requests cannot run model inference concurrently in the same process.

## What changed

### 1. General OCR was separated from container OCR

`app/services/general_ocr.py` handles ordinary portal text.

`app/services/ocr_service.py` remains the specialized ISO 6346 container-number detector.

`app/services/paddle_engine.py` owns the shared Standard PaddleOCR instance so the application does not unnecessarily load the same OCR model twice.

### 2. Container OCR is OCR-first

The original implementation was VLM-first. The new design is:

```text
Standard PaddleOCR
       |
       v
ISO candidate extraction
       |
       v
ISO 6346 validation
       |
       +---- valid → return
       |
       +---- no valid result → optional VLM fallback
```

VLM is disabled by default.

### 3. Screen classification

`app/services/screen_classifier.py` identifies:

- `PIN_GENERATION_PSA`
- `PIN_GENERATION_APM`
- `PIN_GENERATION_NSFT`
- `TRUCK_BOOKING_NSFT`
- `UNKNOWN`

Initial classification is deterministic and based on OCR-visible labels. This avoids adding a classifier model before real data demonstrates the need for one.

### 4. Field extraction

`app/services/field_extractor.py` maps OCR text to canonical fields using:

- field labels
- OCR bounding boxes
- same-row relationships
- nearby values
- screen type

It does **not** depend on fixed screenshot coordinates.

### 5. Canonical field rules

`app/config/field_rules.py` contains aliases, field types, risk levels, and screen-specific field lists.

The field list is screen-specific so an unrelated field does not automatically become a manual-review error.

### 6. Normalization

`app/services/normalizer.py` handles safe normalization such as:

```text
+91 9594894803 → 9594894803
RLTU 304903 4  → RLTU3049034
ula            → ULA
```

Ambiguous OCR characters are not silently guessed.

### 7. Validation

`app/services/validator.py` validates fields according to their configured type.

Examples:

- phone number
- numeric fields
- container size
- name
- alphanumeric fields
- container number / ISO 6346

### 8. Cross-field validation

`app/services/cross_field_validator.py` checks relationships such as:

```text
Declared containers = 1
Detected populated container fields = 2
                         ↓
                    CONFLICT
```

and:

```text
Transaction = ONLY IMPORT
Export container populated
                         ↓
                    CONFLICT
```

The system does not silently choose one value when evidence conflicts.

### 9. Optional VLM fallback

`app/services/vlm_verifier.py` provides a replaceable VLM layer.

It is intended for uncertain/high-risk fields rather than every image.

If VLM is unavailable, OCR + validation + manual review still works.

The current VLM implementation is deliberately conservative: it can provide visual evidence, but it is not allowed to override a deterministic validation result merely because the VLM produced a plausible value.

### 10. Manual review

The API distinguishes:

```text
EMPTY
NOT_FOUND
INVALID
NEEDS_REVIEW
CONFLICT
VERIFIED
VLM_VERIFIED
MANUAL
```

This allows a frontend to ask the user only for fields that need intervention.

## Canonical output

A field looks like:

```json
{
  "value": "9594894803",
  "confidence": 0.99,
  "status": "VERIFIED",
  "source": "OCR",
  "bbox": [250, 260, 530, 290],
  "validation": {
    "valid": true,
    "status": "VALID"
  }
}
```

An empty field looks like:

```json
{
  "value": null,
  "confidence": null,
  "status": "EMPTY",
  "source": "SYSTEM"
}
```

An unresolved field looks like:

```json
{
  "value": "MH03FCO372",
  "confidence": 0.58,
  "status": "NEEDS_REVIEW",
  "source": "OCR",
  "reason": "Ambiguous O/0 character"
}
```

## API

### Health

```text
GET /health
```

### Existing container OCR endpoint

```text
POST /ocr
```

Upload one image.

This remains compatible with the original specialized container-number use case.

### New complete analysis endpoint

```text
POST /analyze
```

Upload one or more images using the `files` multipart field.

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Configuration

Copy `.env.example` to `.env`.

Important defaults:

```text
ENABLE_VLM=false
OCR_HIGH_CONFIDENCE_THRESHOLD=0.90
OCR_REVIEW_THRESHOLD=0.70
VLM_CONFIDENCE_THRESHOLD=0.85
SAVE_DEBUG_OUTPUT=true
```

The thresholds are initial values and should be calibrated against a labeled set of real screenshots.

## Running

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Do not use `--reload` when benchmarking OCR/model startup. `--reload` is fine during ordinary development.

## Tests

```powershell
pytest -q
```

Tests cover:

- ISO 6346 check digit
- invalid container numbers
- phone normalization/validation
- container-count conflicts
- transaction/export conflicts

## Important production limitation

The field extractor is intentionally generic and label/spatial based. It is not yet a learned document-layout model and it does not know every portal-specific rule.

Before using the output to automatically submit transactions, collect a labeled dataset of real screenshots and calibrate:

1. field association accuracy
2. OCR confidence thresholds
3. portal-specific field formats
4. cross-field business rules
5. VLM fallback behavior
6. manual-review thresholds

The system must prefer `null + manual review` over an invented or weakly supported value.
