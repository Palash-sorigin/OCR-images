from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.config import ENABLE_VLM
from app.dependencies import initialize_services, shutdown_services
from app.routes.documents import router as documents_router
from app.routes.ocr import router as ocr_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 70)
    print("Starting Container / Portal Screenshot OCR API")
    print("=" * 70)

    initialize_services()

    yield

    shutdown_services()


app = FastAPI(
    title="Container & Portal Extraction API",
    version="2.0.0",
    description=(
        "Sequential screenshot extraction pipeline: general PaddleOCR, "
        "specialized container OCR, deterministic validation, optional VLM fallback, "
        "manual review, and cross-image reconciliation."
    ),
    lifespan=lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    def convert_file_schema(obj):
        if isinstance(obj, dict):

            # OpenAPI 3.1 representation:
            #
            # {
            #     "type": "string",
            #     "contentMediaType": "application/octet-stream"
            # }
            #
            # Convert to OpenAPI 3.0 / Swagger representation:
            #
            # {
            #     "type": "string",
            #     "format": "binary"
            # }

            if obj.get("contentMediaType") == "application/octet-stream":
                obj.pop("contentMediaType", None)
                obj["format"] = "binary"

            # Recursively process everything underneath this object.
            for value in obj.values():
                convert_file_schema(value)

        elif isinstance(obj, list):
            for item in obj:
                convert_file_schema(item)

    # The important part:
    # process the ENTIRE generated OpenAPI document, including
    # components/schemas where FastAPI puts the UploadFile definitions.
    convert_file_schema(openapi_schema)

    # Tell Swagger UI to treat this as OpenAPI 3.0.
    openapi_schema["openapi"] = "3.0.3"

    app.openapi_schema = openapi_schema

    return app.openapi_schema

app.openapi = custom_openapi


app.include_router(ocr_router)
app.include_router(documents_router)


@app.get("/health")
def health():
    from app.dependencies import get_ocr_service

    service = get_ocr_service()

    return {
        "status": "ok",
        "ocrLoaded": service.ocr is not None,
        "vlmLoaded": service.vlm is not None,
        "vlmEnabled": ENABLE_VLM,
    }