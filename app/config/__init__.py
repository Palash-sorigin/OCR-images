from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENABLE_VLM = os.getenv("ENABLE_VLM", "false").lower() == "true"
VLM_VERSION = os.getenv("VLM_VERSION", "v1.6")
OCR_CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "4"))
SAVE_DEBUG_OUTPUT = os.getenv("SAVE_DEBUG_OUTPUT", "true").lower() == "true"
OCR_HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_HIGH_CONFIDENCE_THRESHOLD", "0.90"))
OCR_REVIEW_THRESHOLD = float(os.getenv("OCR_REVIEW_THRESHOLD", "0.70"))
VLM_CONFIDENCE_THRESHOLD = float(os.getenv("VLM_CONFIDENCE_THRESHOLD", "0.85"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))

# Gemini Flash LLM — used for Visit Ticket OCR+LLM extraction instead of
# the heavier PaddleOCR-VL pipeline.  Set ENABLE_LLM_EXTRACTION=false to
# fall back to the VLM+OCR hybrid path.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
ENABLE_LLM_EXTRACTION = os.getenv("ENABLE_LLM_EXTRACTION", "true").lower() == "true"
