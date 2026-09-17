from app.services.ocr_service import ContainerOCRService
from app.services.paddle_engine import PaddleOCREngine


def build_container_ocr_service(engine: PaddleOCREngine) -> ContainerOCRService:
    return ContainerOCRService(engine=engine)
