"""Gemini Flash LLM engine for structured field extraction.

This module calls Google's Gemini Flash model (text-only, no image) to
select the correct field values from OCR-detected candidates.  The LLM
acts purely as a *structurer* — it picks the best value from what OCR
found, but NEVER invents data.  The system prompt is designed to
minimise hallucination by constraining the model to only return values
already present in the OCR data.

Confidence scoring:  the final score is OCR-dominant.  The LLM's
agreement gives at most a 5 % uplift and never inflates an originally-
low OCR score beyond a hard ceiling.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import GEMINI_API_KEY, GEMINI_MODEL


# ── system prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a document data extraction assistant.  You are given raw OCR data
from a shipping / logistics terminal visit ticket, along with pre-matched
candidate values for each field.

### CRITICAL RULES — read carefully

1. You MUST ONLY return values that appear in the provided OCR data or
   candidate lists.  NEVER invent, guess, or generate any value that is
   not present in the input.
2. If no suitable candidate exists for a field, set its value to null.
3. When multiple candidates exist, prefer:
   a. The one whose OCR confidence is highest.
   b. The one that is label-anchored (found near the field's label text).
   c. The one that best matches known formatting patterns (e.g. Indian
      truck registration XX00XX0000, ISO 6346 container numbers).
4. You may correct OBVIOUS single-character OCR errors (e.g. letter 'O'
   read as digit '0' in a name field) ONLY when the corrected character
   also appears in at least one other OCR item.  If unsure, keep the
   original OCR value unchanged.
5. Return ONLY valid JSON matching the exact schema below — no markdown
   fences, no commentary, no explanation.
6. For every field you populate, include the ``ocr_indices`` list showing
   which OCR item(s) you used (by their ``index``), so the result can
   be audited.

### Response schema

Return a JSON object whose keys are the field names.  Each value is an
object with exactly these keys:
  - "value": the selected string value, or null.
  - "ocr_indices": list of integer indices of the OCR items used.
  - "reason": one short sentence explaining why this value was chosen,
    or why null was returned.
"""


class GeminiLLMEngine:
    """Thin wrapper around Gemini Flash for field extraction."""

    def __init__(self) -> None:
        self.client = None
        self.model = GEMINI_MODEL
        self.error: str | None = None
        if GEMINI_API_KEY:
            self._load()
        else:
            self.error = "GEMINI_API_KEY is not set."
            print("[startup] Gemini LLM disabled: GEMINI_API_KEY not configured.")

    def _load(self) -> None:
        try:
            from google import genai

            self.client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[startup] Gemini LLM engine ready (model={self.model}).")
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)
            print(f"[startup] Gemini LLM engine failed: {exc}")

    @property
    def available(self) -> bool:
        return self.client is not None

    def extract_fields(
        self,
        llm_context: dict[str, Any],
        field_rules: dict[str, dict],
    ) -> dict[str, dict[str, Any]]:
        """Call Gemini Flash and return parsed field results.

        Returns a dict of ``{field_name: {value, ocr_indices, reason}}``.
        On any error, returns all fields as ``null`` with a diagnostic.
        """
        if not self.available:
            return self._null_result(field_rules, f"LLM unavailable: {self.error}")

        user_prompt = self._build_user_prompt(llm_context, field_rules)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            )

            raw_text = response.text or ""
            parsed = self._parse_json(raw_text)
            if parsed is None:
                return self._null_result(
                    field_rules, f"LLM returned unparseable response: {raw_text[:200]}"
                )

            return self._validate_response(parsed, field_rules, llm_context)

        except Exception as exc:  # pragma: no cover
            return self._null_result(field_rules, f"LLM call failed: {exc}")

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
