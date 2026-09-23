from contextlib import asynccontextmanager

from fastapi import FastAPI

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
