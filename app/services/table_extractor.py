from __future__ import annotations

from collections import defaultdict


class TableExtractor:
    """Lightweight row reconstruction from OCR boxes; no hard-coded pixel coordinates."""

    def extract(self, items: list[dict]) -> list[dict]:
        headers = {"PENDENCY", "TOTALALLOCATEDMOVES", "GENERATEDPIN", "BALANCE"}
        if not any(self._compact(i.get("text", "")) in headers for i in items):
            return []

        rows: list[list[dict]] = []
        sorted_items = sorted(items, key=lambda i: self._center(i)[1])
        for item in sorted_items:
            x, y = self._center(item)
            placed = False
            for row in rows:
                _, row_y = self._center(row[0])
                if abs(y - row_y) <= max(12, 0.35 * self._height(item)):
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])

        output = []
        for row in rows:
            row = sorted(row, key=lambda i: self._center(i)[0])
            text = [i.get("text", "") for i in row if i.get("text")]
            if text:
                output.append({"cells": text, "boxes": [i.get("box") for i in row]})
        return output

    @staticmethod
    def _compact(text: str) -> str:
        return "".join(ch for ch in str(text).upper() if ch.isalnum())

    @staticmethod
    def _center(item: dict) -> tuple[float, float]:
        box = item.get("box") or [0, 0, 0, 0]
        try:
            return ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)
        except (TypeError, ValueError, IndexError):
            return 0.0, 0.0

    @staticmethod
    def _height(item: dict) -> float:
        box = item.get("box") or [0, 0, 0, 0]
        try:
            return abs(float(box[3]) - float(box[1]))
        except (TypeError, ValueError, IndexError):
            return 20.0
