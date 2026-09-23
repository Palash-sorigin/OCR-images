from app.services.ocr_service import ContainerOCRService
from app.services.paddle_engine import PaddleOCREngine
from app.services.vlm_engine import VLMEngine


def build_container_ocr_service(engine: PaddleOCREngine, vlm_engine: VLMEngine) -> ContainerOCRService:
    return ContainerOCRService(engine=engine, vlm_engine=vlm_engine)
