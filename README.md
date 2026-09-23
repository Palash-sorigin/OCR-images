# Container & Portal Screenshot OCR API

FastAPI service for a sequential **Extract → Validate → Review → Fill** workflow.

The original project was a container-number OCR service. It now keeps that specialized container OCR as a reusable component, adds general portal screenshot extraction for PIN Generation and Truck Booking screens, and adds a second, VLM+OCR hybrid pipeline for fixed-schema logistics documents: EIR, Form 13, Visit Ticket, vehicle dashboard display, and container seal (see [Fixed-schema document pipeline](#fixed-schema-document-pipeline-eir-form-13-visit-ticket-vehicle-display-container-seal) below).

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

## Fixed-schema document pipeline (EIR, Form 13, Visit Ticket, vehicle display, container seal)

Beyond the container/portal pipeline above, the service has a second family of
pipelines for photographed logistics documents: Equipment Interchange
Reports, Form 13 / E-Gate passes, terminal visit tickets, EV dashboard
displays, and container seal close-ups. Each has its own route, its own
fixed field schema, and its own `DocumentExtractionPipeline` instance, all
sharing the same underlying PaddleOCR and PaddleOCR-VL model instances so
nothing is loaded twice.

```text
                    Uploaded document photo
                               |
                               v
                    Image quality check
                               |
                               v
                    Image preprocessing
                               |
                    +----------+----------+
                    |                     |
              Standard OCR            VLM extraction
                    |                     |
                    +----------+----------+
                               |
                               v
                    Sub-type classification
                     (e.g. EIR: NSFT vs DP World)
                               |
                               v
              Subtype-scoped field list selected
                               |
                               v
        Label-anchored regex extraction, per field,
           over BOTH the VLM text and the OCR text
                               |
                        +------+------+
                        |             |
                  both agree     only one found / disagree
                        |             |
                  higher confidence   lower confidence /
                                       flagged for review
                               |
                               v
                    Normalization -> Validation
                               |
                               v
                 Document-specific cross-field checks
                   (e.g. gate-in must not be after gate-out)
                               |
                               v
                         Manual review
```

### Why this pipeline differs from the portal/container one

The container/portal pipeline associates a label to a value using OCR
bounding-box geometry (same row / directly below), which works well for
clean, consistently-laid-out web screenshots. These documents are
photographed receipts and multi-column forms with skew, rotation, and
inconsistent reading order, so spatial heuristics are unreliable here.
Instead, each field has a **label-anchored regex** (`app/config/document_schemas.py`):
find the label text, then search a window of text right after it for a
value matching the field's pattern. A handful of fields with no reliable
printed label (e.g. an EV dashboard's unlabeled voltage reading) are matched
by a standalone pattern over the whole text instead.

Because VLM and OCR are run as two independent passes, every field is
**cross-checked between them** (`app/services/document_text_extractor.py`):

- Found by both, same value → `source: "VLM+OCR"`, high confidence.
- Found by only one → lower confidence, still returned, but flagged.
- Found by both with **different** values → the disagreement is reported in
  `reason` rather than silently picking one -- the same "never invent a
  value" principle the container/portal pipeline already applies.

### Document types and routes

| Route | Document | Sub-types | Notes |
|---|---|---|---|
| `POST /ocr/eir` | Equipment Interchange Report/Receipt | `NSFT`, `DPWORLD` | Two structurally different printed layouts; field list is scoped per subtype after classification. |
| `POST /ocr/form13` | Form 13 / E-Gate Pass (Form 13/Form 6/SEZ 4) | single layout | Multi-column CARGOES e-gate form. |
| `POST /ocr/visit-ticket` | Terminal visit ticket | `GTI_DROPOFF_EXPORT`, `GTI_PICKUP_IMPORT`, `DPWORLD_NSICT`, `PSA_BMCT_PICKUP` | Four distinct terminal formats; field list scoped per subtype. Unrecognized layouts fall back to the full field set rather than extracting nothing. |
| `POST /ocr/vehicle-display` | EV dashboard (SOC %, voltage, odometer) | `EV_DASHBOARD` | See limitation below -- these readings are largely unlabeled. |
| `POST /ocr/container-seal` | Container seal close-up photo | `SEAL_PHOTO` | Relies primarily on VLM; see limitation below. |

Each route accepts one or more images via the `files` multipart field, the
same convention as `/analyze`:

```bash
curl -X POST http://127.0.0.1:8000/ocr/eir \
  -F "files=@eir1.jpg" \
  -F "files=@eir2.jpg"
```

All routes -- `/ocr`, `/analyze`, and the five document routes -- serialize
through the **same** process-wide lock (`app/routes/_shared.py`). They all
call into the same shared PaddleOCR/VLM model instances, so a route-local
lock would only prevent concurrency within one route while still letting
two different routes run model inference at the same time; that would defeat
the point.

### Known limitations (read before trusting the output)

- **Vehicle dashboard fields are inherently a heuristic, not a labeled
  extraction.** Most gauges (SOC, voltage, temperature) are marked by icons,
  not words, and the dashboard shows two-to-three unlabeled `"...km"`
  numbers at once. `odometer_km` / `range_to_empty_km` are told apart by
  "the larger bare integer is the odometer" (true across every sample
  observed, but a heuristic, not a rule) -- `range_to_empty_km > odometer_km`
  is cross-validated and flagged, but the pair can still be swapped without
  tripping that check. Treat this route's output as best-effort telemetry.
- **Container seal photos are frequently rotated 90-180 degrees.** The
  shared standard-OCR engine has orientation classification switched off
  (see `app/services/paddle_engine.py` -- changing it would affect the
  already-tuned container/portal pipeline), so OCR corroboration is weak on
  this route. The VLM engine has `use_doc_orientation_classify=True` and
  `use_seal_recognition=True`, so it carries most of the weight here;
  without `ENABLE_VLM=true`, expect this route to under-perform.
  Seal formats also vary by manufacturer with no ISO-equivalent standard,
  so `seal_number` validation is a loose alphanumeric-with-a-digit check,
  not a format-specific one.
- **Visit Ticket has four distinct terminal formats** discovered by
  inspecting the sample set; a terminal format not seen yet will classify
  as the closest match or fall back to the full field list. Expect to
  extend `VISIT_TICKET_CLASSIFY_RULES` / `VISIT_TICKET_FIELDS_BY_SUBTYPE`
  as new formats show up in real traffic.
- **Not yet run against live PaddleOCR/PaddleOCR-VL output.** The
  extraction/normalization/validation/classification logic is tested
  against text transcribed directly from the real sample documents in
  `docs/docs/` (see `tests/test_document_extraction.py`), plus one
  end-to-end run of the real pipeline against a real sample image with
  stubbed OCR/VLM text. It has not been run against actual model inference
  output, since `paddleocr` requires a dependency install this environment
  couldn't reach. Before relying on this in production, run each route
  against a batch of the real sample images with `ENABLE_VLM=true` and
  `SAVE_DEBUG_OUTPUT=true`, and compare the extracted fields against the
  source photos.

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
- field-extraction exclusivity and the VLM run-once-per-image fix (`test_pipeline_fixes.py`)
- document field extraction for EIR/Form 13/Visit Ticket/vehicle display/container seal, against text transcribed from the real sample documents, plus the new `date`/`datetime`/`weight`/`percentage`/`voltage`/`temperature`/`seal_number`/strict `truck_registration` kinds (`test_document_extraction.py`)

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
