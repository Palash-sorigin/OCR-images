from __future__ import annotations

from app.services.container_ocr_adapter import ContainerOCRAdapter
from app.services.container_ocr_service_factory import build_container_ocr_service
from app.services.general_ocr import GeneralOCRService
from app.services.paddle_engine import PaddleOCREngine
from app.services.pipeline import SequentialExtractionPipeline
from app.services.vlm_verifier import VLMVerifier


paddle_engine: PaddleOCREngine | None = None
container_ocr_service = None
general_ocr_service = None
container_ocr_adapter = None
vlm_verifier = None
pipeline = None


def initialize_services() -> None:
    global paddle_engine, container_ocr_service, general_ocr_service
    global container_ocr_adapter, vlm_verifier, pipeline

    if pipeline is not None:
        return

    # One shared standard PaddleOCR model per application process.
    paddle_engine = PaddleOCREngine()
    general_ocr_service = GeneralOCRService(paddle_engine)
    container_ocr_service = build_container_ocr_service(paddle_engine)
    container_ocr_adapter = ContainerOCRAdapter(container_ocr_service)
    vlm_verifier = VLMVerifier()
    pipeline = SequentialExtractionPipeline(general_ocr_service, container_ocr_adapter, vlm_verifier)


def shutdown_services() -> None:
    global paddle_engine, container_ocr_service, general_ocr_service
    global container_ocr_adapter, vlm_verifier, pipeline
    paddle_engine = None
    container_ocr_service = None
    general_ocr_service = None
    container_ocr_adapter = None
    vlm_verifier = None
    pipeline = None


def get_ocr_service():
    if container_ocr_service is None:
        raise RuntimeError("OCR service has not been initialized.")
    return container_ocr_service


def get_pipeline() -> SequentialExtractionPipeline:
    if pipeline is None:
        raise RuntimeError("Extraction pipeline has not been initialized.")
    return pipeline
