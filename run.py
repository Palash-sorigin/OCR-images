import argparse
import json
from pathlib import Path

from app.services.ocr_service import ContainerOCRService
from app.services.vlm_engine import VLMEngine


def main():
    parser = argparse.ArgumentParser(
        description="Detect a container number from an image."
    )
    parser.add_argument("image", help="Path to the input image.")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    service = ContainerOCRService(vlm_engine=VLMEngine())
    result = service.process(image_path)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
