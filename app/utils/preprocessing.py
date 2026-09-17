from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter


def inspect_image_quality(image_path: Path) -> dict:
    try:
        image = Image.open(image_path)
        width, height = image.size
        if width < 600 or height < 400:
            status = "WARNING"
            reason = "Image dimensions are small and may reduce OCR accuracy."
        else:
            status = "GOOD"
            reason = None
        return {"status": status, "width": width, "height": height, "reason": reason}
    except Exception as exc:
        return {"status": "UNREADABLE", "width": None, "height": None, "reason": str(exc)}


def preprocess_image(image_path: Path, output_dir: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    if image.width < 1600:
        scale = 1600 / image.width
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = image.filter(ImageFilter.SHARPEN)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_processed.jpg"
    image.save(output_path, quality=95)
    return output_path
