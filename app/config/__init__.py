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

# Gemini Flash-Lite LLM — used for Visit Ticket OCR+image+LLM extraction
# instead of the much heavier PaddleOCR-VL pipeline (which measured at
# 3-7 minutes/image on CPU; Gemini Flash-Lite is a hosted multimodal model
# sized for low-latency, high-throughput document parsing). Set
# ENABLE_LLM_EXTRACTION=false to disable and use fast OCR-only extraction.
#
# IMPORTANT: unlike every other model in this project, this one is NOT
# self-hosted. Enabling it sends the document image and its OCR text to
# Google's Gemini API. Confirm that is acceptable for your data before
# turning this on for real documents.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
ENABLE_LLM_EXTRACTION = os.getenv("ENABLE_LLM_EXTRACTION", "true").lower() == "true"
# Hard ceiling on the Gemini call itself. On timeout, the pipeline falls
# back to fast OCR-only extraction rather than blocking -- see
# GeminiLLMEngine.extract_fields and DocumentExtractionPipeline._process_with_llm.
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "12"))
