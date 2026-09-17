from __future__ import annotations

from pathlib import Path

from app.services.ocr_service import ContainerOCRService


class ContainerOCRAdapter:
    """Compatibility wrapper around the existing container-number OCR service."""

    def __init__(self, service: ContainerOCRService) -> None:
        self.service = service

    def extract(self, image_path: Path) -> dict:
        return self.service.process(image_path)
