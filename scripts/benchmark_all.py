"""Run every image in the project through its pipeline, with per-stage timing.

Usage:
    python scripts/benchmark_all.py

Writes a JSON report to output/benchmark_report.json and prints a summary
table to stdout: per-image latency broken into OCR time / VLM time / rest,
status, and any errors, grouped by document category.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.document_schemas import DOCUMENT_TYPES
from app.services.container_ocr_service_factory import build_container_ocr_service
from app.services.document_pipeline import DocumentExtractionPipeline
from app.services.general_ocr import GeneralOCRService
from app.services.paddle_engine import PaddleOCREngine
from app.services.pipeline import SequentialExtractionPipeline
from app.services.container_ocr_adapter import ContainerOCRAdapter
from app.services.vlm_engine import VLMEngine
from app.services.vlm_verifier import VLMVerifier

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CATEGORIES: list[tuple[str, Path, str]] = [
    ("container", ROOT / "sample images", "container_ocr"),
    ("container", ROOT / "docs" / "docs" / "Container", "container_ocr"),
    ("container_seal", ROOT / "docs" / "docs" / "Container Seal", "document:container_seal"),
    ("eir", ROOT / "docs" / "docs" / "EIR", "document:eir"),
    ("form13", ROOT / "docs" / "docs" / "Form 13", "document:form13"),
    ("vehicle_display", ROOT / "docs" / "docs" / "KM and Battery percentage", "document:vehicle_display"),
    ("port_login", ROOT / "docs" / "docs" / "Port Login Screenshots", "portal"),
    ("visit_ticket", ROOT / "docs" / "docs" / "Visit Ticket", "document:visit_ticket"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def main() -> None:
    print("=" * 78)
    print("Loading models (PaddleOCR + PaddleOCR-VL)...")
    print("=" * 78)
    load_start = time.perf_counter()
    paddle_engine = PaddleOCREngine()
    vlm_engine = VLMEngine()
    load_elapsed = time.perf_counter() - load_start
    print(f"Model load time: {load_elapsed:.2f}s")
    print(f"OCR loaded: {paddle_engine.ocr is not None} (error: {paddle_engine.error})")
    print(f"VLM loaded: {vlm_engine.vlm is not None} (error: {vlm_engine.error})")

    general_ocr = GeneralOCRService(paddle_engine)
    vlm_verifier = VLMVerifier(vlm_engine)
    container_ocr_service = build_container_ocr_service(paddle_engine, vlm_engine)
    container_ocr_adapter = ContainerOCRAdapter(container_ocr_service)
    portal_pipeline = SequentialExtractionPipeline(general_ocr, container_ocr_adapter, vlm_verifier)
    document_pipelines = {
        doc_type: DocumentExtractionPipeline(doc_type, general_ocr, vlm_verifier)
        for doc_type in DOCUMENT_TYPES
    }

    report: list[dict] = []

    for category, folder, pipeline_kind in CATEGORIES:
        if not folder.exists():
            print(f"[skip] {folder} does not exist")
            continue
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            continue

        print()
        print("-" * 78)
        print(f"Category: {category} ({pipeline_kind})  [{len(images)} images]")
        print("-" * 78)

        for image_path in images:
            entry = {
                "category": category, "pipeline": pipeline_kind, "file": image_path.name,
            }
            try:
                # Isolate OCR-only and VLM-only timing directly, in addition
                # to the full pipeline, so slow stages are identifiable.
                ocr_result, ocr_time = timed(general_ocr.extract, image_path)
                entry["ocr_time_s"] = round(ocr_time, 3)
                entry["ocr_item_count"] = len(ocr_result.get("items", []))

                if vlm_engine.vlm is not None:
                    vlm_result, vlm_time = timed(vlm_verifier.extract_text, image_path)
                    entry["vlm_time_s"] = round(vlm_time, 3)
                    entry["vlm_text_len"] = len(vlm_result.get("text", ""))
                else:
                    entry["vlm_time_s"] = None

                if pipeline_kind == "container_ocr":
                    result, total_time = timed(container_ocr_service.process, image_path)
                    entry["total_time_s"] = round(total_time, 3)
                    entry["container_number"] = result.get("containerNumber")
                    entry["method"] = result.get("method")
                    entry["status"] = "OK" if result.get("containerNumber") else "NO_NUMBER_FOUND"
                elif pipeline_kind == "portal":
                    result, total_time = timed(portal_pipeline.process_one, image_path.name, image_path)
                    entry["total_time_s"] = round(total_time, 3)
                    entry["status"] = result.status
                    entry["screen_type"] = result.screen_type
                    entry["manual_review_count"] = len(result.manual_review)
                    entry["fields"] = {
                        name: f.value for name, f in result.fields.items() if f.value is not None
                    }
                elif pipeline_kind.startswith("document:"):
                    doc_type = pipeline_kind.split(":", 1)[1]
                    pipeline = document_pipelines[doc_type]
                    result, total_time = timed(pipeline.process_one, image_path.name, image_path)
                    entry["total_time_s"] = round(total_time, 3)
                    entry["status"] = result.status
                    entry["subtype"] = result.document_subtype
                    entry["manual_review_count"] = len(result.manual_review)
                    entry["fields"] = {
                        name: f.value for name, f in result.fields.items() if f.value is not None
                    }
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "EXCEPTION"
                entry["error"] = f"{type(exc).__name__}: {exc}"

            report.append(entry)
            status = entry.get("status", "?")
            total = entry.get("total_time_s", "?")
            print(f"  {image_path.name:45s} status={status:20s} total={total}s "
                  f"ocr={entry.get('ocr_time_s')}s vlm={entry.get('vlm_time_s')}s")

    report_path = OUTPUT_DIR / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    by_category: dict[str, list[dict]] = {}
    for entry in report:
        by_category.setdefault(entry["category"], []).append(entry)

    for category, entries in by_category.items():
        times = [e["total_time_s"] for e in entries if isinstance(e.get("total_time_s"), (int, float))]
        errors = [e for e in entries if e.get("status") == "EXCEPTION"]
        avg = sum(times) / len(times) if times else 0
        mx = max(times) if times else 0
        print(f"{category:20s} n={len(entries):3d}  avg={avg:6.2f}s  max={mx:6.2f}s  errors={len(errors)}")
        for e in errors:
            print(f"    ERROR: {e['file']}: {e.get('error')}")

    print()
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    main()
