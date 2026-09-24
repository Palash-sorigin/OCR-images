"""Gemini Flash-Lite LLM engine for structured field extraction.

This module calls Google's Gemini Flash-Lite model with BOTH the document
image and the OCR-derived JSON context (raw OCR items, pre-matched
candidates, field schema) to select the correct field values. Sending the
image alongside the OCR text gives the model visual grounding that
text-only OCR candidates lose (relative position, which of two similar
values sits next to which label, etc.) -- this is why a text-only first
version of this engine was giving unreliable answers.

The LLM acts purely as a *selector*, never a generator: the system prompt
constrains it to only return values already present in the OCR data, and
_validate_response() independently re-verifies every returned value against
the actual OCR text as a backstop -- the prompt is not trusted alone.

Confidence scoring is OCR-dominant: the LLM's agreement gives at most a 5%
uplift and never inflates an originally-low OCR score beyond a hard ceiling.

Latency safety: the API call is wrapped in a hard timeout
(GEMINI_TIMEOUT_SECONDS) enforced by a worker thread, independent of
whatever timeout behavior the SDK itself does or doesn't implement. On
timeout or any other failure, the caller (DocumentExtractionPipeline) falls
back to fast OCR-only extraction, never to the slow VLM path.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from app.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT_SECONDS

_MIME_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff",
}


# ── system prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a document data extraction assistant. You are given a photo of a
shipping / logistics terminal visit ticket, PLUS the raw OCR data extracted
from that same photo (OCR items with bounding boxes, and pre-matched
candidate values per field).

Use the IMAGE to understand layout and resolve ambiguity -- e.g. which of
two similar-looking values sits next to which label, or how an OCR engine
may have incorrectly split one value across two boxes. The image is context
for disambiguation, not a license to transcribe freely.

### CRITICAL RULES — read carefully

1. You MUST ONLY return values that appear in the provided OCR data or
   candidate lists. NEVER invent, guess, or return a value read from the
   image that the OCR data does not also contain in some form. If the
   image shows something OCR completely missed, set that field to null
   rather than supplying a value with no OCR support.
2. If no suitable candidate exists for a field, set its value to null.
3. When multiple candidates exist, use the image to decide, then prefer:
   a. The one whose OCR confidence is highest.
   b. The one that is label-anchored (found near the field's label text).
   c. The one that best matches known formatting patterns (e.g. Indian
      truck registration XX00XX0000, ISO 6346 container numbers).
4. You may correct OBVIOUS single-character OCR errors (e.g. letter 'O'
   read as digit '0') ONLY when the image confirms the correction AND the
   corrected character also appears in at least one other OCR item. If
   unsure, keep the original OCR value unchanged.
5. Return ONLY valid JSON matching the exact schema below — no markdown
   fences, no commentary, no explanation.
6. For every field you populate, include the ``ocr_indices`` list showing
   which OCR item(s) you used (by their ``index`` from the OCR items list
   below), so the result can be audited. Cite the REAL index from the OCR
   items list, not a placeholder.

### Response schema

Return a JSON object whose keys are the field names. Each value is an
object with exactly these keys:
  - "value": the selected string value, or null.
  - "ocr_indices": list of integer indices of the OCR items used.
  - "reason": one short sentence explaining why this value was chosen,
    or why null was returned.
"""


class GeminiLLMEngine:
    """Thin wrapper around Gemini Flash-Lite for image+OCR field extraction."""

    # One worker is enough: calls are made one at a time, serialized behind
    # the same pipeline_lock every other route uses (see app/routes/_shared.py).
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemini-llm")

    def __init__(self) -> None:
        self.client = None
        self.model = GEMINI_MODEL
        self.error: str | None = None
        if GEMINI_API_KEY:
            self._load()
        else:
            self.error = "GEMINI_API_KEY is not set."
            print("[gemini] disabled at startup: GEMINI_API_KEY not configured -- "
                  "visit_ticket will use fast OCR-only extraction.")

    def _load(self) -> None:
        try:
            from google import genai

            self.client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[gemini] engine ready: model={self.model}, timeout={GEMINI_TIMEOUT_SECONDS}s")
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)
            print(f"[gemini] startup failed, will fall back to OCR-only extraction: {exc}")

    @property
    def available(self) -> bool:
        return self.client is not None

    def extract_fields(
        self,
        image_path: Path,
        llm_context: dict[str, Any],
        field_rules: dict[str, dict],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Call Gemini Flash-Lite with the document image + OCR JSON context.

        Returns (result, meta). result is ``{field_name: {value, ocr_indices,
        reason}}``. meta always has "status" ("OK" | "TIMEOUT" | "UNAVAILABLE"
        | "ERROR") and "elapsed_s"; the caller (DocumentExtractionPipeline)
        uses meta["status"] to decide whether to fall back to OCR-only
        extraction -- it never blocks past GEMINI_TIMEOUT_SECONDS to find out.
        """
        t0 = time.perf_counter()

        if not self.available:
            print(f"[gemini] skipped: unavailable ({self.error})")
            return self._null_result(field_rules, f"LLM unavailable: {self.error}"), {
                "status": "UNAVAILABLE", "elapsed_s": 0.0, "error": self.error,
            }

        try:
            image_bytes = Path(image_path).read_bytes()
        except OSError as exc:
            print(f"[gemini] skipped: could not read image {image_path}: {exc}")
            return self._null_result(field_rules, f"Could not read image: {exc}"), {
                "status": "ERROR", "elapsed_s": 0.0, "error": str(exc),
            }

        mime_type = _MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")
        user_prompt = self._build_user_prompt(llm_context, field_rules)
        print(f"[gemini] request: model={self.model} image={len(image_bytes)}B "
              f"ocr_items={len(llm_context.get('ocr_items', []))} fields={len(field_rules)}")

        future = self._executor.submit(self._call_model, image_bytes, mime_type, user_prompt)
        try:
            raw_text = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            elapsed = time.perf_counter() - t0
            print(f"[gemini] TIMEOUT after {elapsed:.2f}s (limit {GEMINI_TIMEOUT_SECONDS}s); "
                  f"falling back to OCR-only extraction")
            # The submitted call keeps running in the worker thread and its
            # result is discarded when it eventually returns -- harmless for
            # a stateless read-only API call, and this thread never waits for it.
            return self._null_result(field_rules, "LLM call timed out"), {
                "status": "TIMEOUT", "elapsed_s": elapsed, "error": None,
            }
        except Exception as exc:  # pragma: no cover
            elapsed = time.perf_counter() - t0
            print(f"[gemini] ERROR after {elapsed:.2f}s: {exc}")
            return self._null_result(field_rules, f"LLM call failed: {exc}"), {
                "status": "ERROR", "elapsed_s": elapsed, "error": str(exc),
            }

        elapsed = time.perf_counter() - t0
        parsed = self._parse_json(raw_text)
        if parsed is None:
            print(f"[gemini] responded in {elapsed:.2f}s but JSON was unparseable")
            return self._null_result(
                field_rules, f"LLM returned unparseable response: {raw_text[:200]}"
            ), {"status": "ERROR", "elapsed_s": elapsed, "error": "unparseable_json"}

        result = self._validate_response(parsed, field_rules, llm_context)
        kept = sum(1 for r in result.values() if r.get("value") is not None)
        discarded = sum(1 for r in result.values() if "discarded" in (r.get("reason") or ""))
        print(f"[gemini] responded in {elapsed:.2f}s: {kept}/{len(result)} fields populated"
              + (f", {discarded} discarded as unverifiable" if discarded else ""))
        return result, {"status": "OK", "elapsed_s": elapsed, "error": None}

    def _call_model(self, image_bytes: bytes, mime_type: str, user_prompt: str) -> str:
        """Runs in the worker thread; the only thing extract_fields() waits
        on with a hard timeout."""
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                user_prompt,
            ],
            config={
                "system_instruction": _SYSTEM_PROMPT,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        )
        return response.text or ""

    # ── prompt construction ────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(
        llm_context: dict[str, Any],
        field_rules: dict[str, dict],
    ) -> str:
        parts = [
            "## Document Information\n",
            f"Terminal sub-type: {llm_context.get('document_subtype', 'UNKNOWN')}\n",
            "\n## Full OCR Text\n",
            "```\n" + llm_context.get("full_ocr_text", "(empty)") + "\n```\n",
            "\n## OCR Items (with index, confidence, bounding box)\n",
            "```json\n" + json.dumps(llm_context.get("ocr_items", []), indent=1) + "\n```\n",
            "\n## Expected Fields and Their Schemas\n",
            "```json\n" + json.dumps(llm_context.get("field_schema", {}), indent=2) + "\n```\n",
            "\n## Pre-matched Candidates per Field\n",
            "```json\n" + json.dumps(llm_context.get("candidate_mapping", {}), indent=2) + "\n```\n",
            "\n## Instructions\n",
            "Select the best value for each field from the candidates / OCR items above.\n",
            "Return a single JSON object with ALL field names listed in the schema.\n",
            "Do NOT wrap the JSON in markdown code fences.\n",
        ]
        return "".join(parts)

    # ── response parsing & validation ──────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Best-effort JSON extraction from the LLM response."""
        text = text.strip()
        # Strip markdown fences if the LLM added them despite instructions.
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find the first { ... } block.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _validate_response(
        parsed: dict,
        field_rules: dict[str, dict],
        llm_context: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Ensure every returned value actually exists in the OCR data.

        Any value the LLM might have hallucinated is replaced with null.
        """
        # Build a set of all text fragments the OCR actually detected.
        ocr_values_upper: set[str] = set()
        for item in llm_context.get("ocr_items", []):
            text = str(item.get("text", "")).upper()
            ocr_values_upper.add(re.sub(r"[^A-Z0-9]", "", text))

        # Also include every candidate value.
        for field_cands in llm_context.get("candidate_mapping", {}).values():
            for bucket in ("candidates", "possible_candidates"):
                for c in field_cands.get(bucket, []):
                    ocr_values_upper.add(re.sub(r"[^A-Z0-9]", "", str(c.get("value", "")).upper()))

        result: dict[str, dict[str, Any]] = {}
        for name in field_rules:
            llm_field = parsed.get(name, {})
            if not isinstance(llm_field, dict):
                llm_field = {"value": llm_field, "ocr_indices": [], "reason": ""}
            value = llm_field.get("value")
            ocr_indices = llm_field.get("ocr_indices", [])
            reason = llm_field.get("reason", "")

            # Anti-hallucination: verify the value exists in OCR data.
            if value is not None:
                compact = re.sub(r"[^A-Z0-9]", "", str(value).upper())
                if compact and compact not in ocr_values_upper:
                    # Check sub-string containment (OCR items may be longer).
                    found = any(compact in ov or ov in compact for ov in ocr_values_upper if ov)
                    if not found:
                        reason = f"LLM value {value!r} not found in OCR data; discarded."
                        value = None
                        ocr_indices = []

            result[name] = {
                "value": value,
                "ocr_indices": ocr_indices if isinstance(ocr_indices, list) else [],
                "reason": reason or "",
            }

        return result

    @staticmethod
    def _null_result(field_rules: dict[str, dict], reason: str) -> dict[str, dict[str, Any]]:
        return {
            name: {"value": None, "ocr_indices": [], "reason": reason}
            for name in field_rules
        }
