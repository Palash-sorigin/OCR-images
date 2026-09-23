from __future__ import annotations

from app.config.document_schemas import DOCUMENT_TYPES
from app.services.container_ocr_adapter import ContainerOCRAdapter
from app.services.container_ocr_service_factory import build_container_ocr_service
from app.services.document_pipeline import DocumentExtractionPipeline
from app.services.general_ocr import GeneralOCRService
from app.services.paddle_engine import PaddleOCREngine
from app.services.pipeline import SequentialExtractionPipeline
from app.services.vlm_engine import VLMEngine
from app.services.vlm_verifier import VLMVerifier


paddle_engine: PaddleOCREngine | None = None
vlm_engine: VLMEngine | None = None
container_ocr_service = None
general_ocr_service = None
container_ocr_adapter = None
vlm_verifier = None
pipeline = None
document_pipelines: dict[str, DocumentExtractionPipeline] = {}


def initialize_services() -> None:
    global paddle_engine, vlm_engine, container_ocr_service, general_ocr_service
    global container_ocr_adapter, vlm_verifier, pipeline, document_pipelines

    if pipeline is not None:
        return

    # One shared standard PaddleOCR model per application process.
    paddle_engine = PaddleOCREngine()
    # One shared VLM model, used by the container-OCR fallback, the general
    # field verifier, and every document pipeline below, instead of each
    # loading its own copy.
    vlm_engine = VLMEngine()
    general_ocr_service = GeneralOCRService(paddle_engine)
    container_ocr_service = build_container_ocr_service(paddle_engine, vlm_engine)
    container_ocr_adapter = ContainerOCRAdapter(container_ocr_service)
    vlm_verifier = VLMVerifier(vlm_engine)
    pipeline = SequentialExtractionPipeline(general_ocr_service, container_ocr_adapter, vlm_verifier)

    # One DocumentExtractionPipeline per fixed schema (EIR, Form 13, Visit
    # Ticket, vehicle display, container seal), all sharing the same OCR/VLM
    # engines so no model is loaded more than once.
    document_pipelines = {
        document_type: DocumentExtractionPipeline(document_type, general_ocr_service, vlm_verifier)
        for document_type in DOCUMENT_TYPES
    }


def shutdown_services() -> None:
    global paddle_engine, vlm_engine, container_ocr_service, general_ocr_service
    global container_ocr_adapter, vlm_verifier, pipeline, document_pipelines
    paddle_engine = None
    vlm_engine = None
    container_ocr_service = None
    general_ocr_service = None
    container_ocr_adapter = None
    vlm_verifier = None
    pipeline = None
    document_pipelines = {}


def get_ocr_service():
    if container_ocr_service is None:
        raise RuntimeError("OCR service has not been initialized.")
    return container_ocr_service


def get_pipeline() -> SequentialExtractionPipeline:
    if pipeline is None:
        raise RuntimeError("Extraction pipeline has not been initialized.")
    return pipeline


def get_document_pipeline(document_type: str) -> DocumentExtractionPipeline:
    if not document_pipelines:
        raise RuntimeError("Document pipelines have not been initialized.")
    if document_type not in document_pipelines:
        raise RuntimeError(f"Unknown document type: {document_type}")
    return document_pipelines[document_type]
