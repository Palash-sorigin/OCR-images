import json
import re
from typing import Iterable


ISO_LOOSE_PATTERN = re.compile(
    r"(?<![A-Z0-9])[A-Z]{3}[UJZ][A-Z0-9]{7}(?![A-Z0-9])"
)

# ---------------------------------------------------------
# 10-character prefix pattern.
#
# Matches the owner code + equipment category + 6-digit
# serial WITHOUT the check digit.
#
# Example:   RLTU304903   (check digit will be appended
#            downstream by complete_iso6346_number)
#
# The 11-char pattern often fails when adjacent text is
# glued after normalization (e.g. CSNU572944345G1), but
# matching 10 chars first is more resilient because 6
# consecutive digits form a stronger boundary signal.
# ---------------------------------------------------------

ISO_PREFIX_PATTERN = re.compile(
    r"(?<![A-Z0-9])[A-Z]{3}[UJZ]\d{6}(?!\d)"
)


def normalize_text(text: str) -> str:
    """
    Convert OCR text into a compact uppercase alphanumeric string.
    """

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper(),
    )


def extract_iso_candidates(text: str) -> list[str]:
    """
    Extract possible 10- or 11-character ISO-style container
    number candidates.

    This function intentionally does NOT perform ISO check-digit
    validation. Validation happens separately in iso6346.py.
    """

    candidates = []

    # ---------------------------------------------------------
    # Search the full text as a single blob.
    # ---------------------------------------------------------

    compact = normalize_text(text)

    candidates.extend(
        ISO_LOOSE_PATTERN.findall(compact)
    )

    candidates.extend(
        ISO_PREFIX_PATTERN.findall(compact)
    )

    # ---------------------------------------------------------
    # Search each line individually.
    #
    # VLM block_content often packs multiple values on
    # separate lines (e.g. "CSNU 572944 3\n45G1").
    # Searching line-by-line prevents adjacent lines from
    # breaking regex boundaries.
    # ---------------------------------------------------------

    for line in text.splitlines():

        line_compact = normalize_text(line)

        candidates.extend(
            ISO_LOOSE_PATTERN.findall(
                line_compact
            )
        )

        candidates.extend(
            ISO_PREFIX_PATTERN.findall(
                line_compact
            )
        )

    # ---------------------------------------------------------
    # Remove common OCR separators
    # ---------------------------------------------------------

    normalized = re.sub(
        r"[\s\-_]+",
        "",
        text.upper(),
    )

    normalized = re.sub(
        r"[^A-Z0-9]",
        "",
        normalized,
    )

    candidates.extend(
        ISO_LOOSE_PATTERN.findall(normalized)
    )

    candidates.extend(
        ISO_PREFIX_PATTERN.findall(normalized)
    )

    return list(
        dict.fromkeys(candidates)
    )



def extract_from_json_objects(
    raw_results: Iterable[object],
) -> list[str]:
    """
    Search PaddleOCR / PaddleOCR-VL JSON-like output
    for ISO-style container-number candidates.
    """

    chunks = []

    for result in raw_results:
        if isinstance(result, str):
            chunks.append(result)
        else:
            chunks.append(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )
            )

    return extract_iso_candidates(
        "\n".join(chunks)
    )


def ocr_text_candidates(
    rec_texts: list[str],
) -> list[str]:
    """
    Generate candidates using only OCR text.

    Handles cases such as:

        EXFU 4416716

    or OCR splitting the text into several boxes.
    """

    normalized = []

    for text in rec_texts:
        value = normalize_text(text)

        if value:
            normalized.append(value)

    candidates = []

    # ---------------------------------------------------------
    # Search original OCR strings together
    # ---------------------------------------------------------

    candidates.extend(
        extract_iso_candidates(
            " ".join(rec_texts)
        )
    )

    # ---------------------------------------------------------
    # Search each OCR result individually
    # ---------------------------------------------------------

    for text in normalized:
        candidates.extend(
            extract_iso_candidates(text)
        )

    # ---------------------------------------------------------
    # Search combinations of neighbouring OCR boxes
    #
    # Example:
    #
    # ["EXFU", "4416716"]
    #
    # becomes:
    #
    # EXFU4416716
    # ---------------------------------------------------------

    max_window = min(
        5,
        len(normalized),
    )

    for window_size in range(
        2,
        max_window + 1,
    ):
        for start in range(
            len(normalized) - window_size + 1
        ):
            window = normalized[
                start:start + window_size
            ]

            joined = "".join(window)

            candidates.extend(
                extract_iso_candidates(joined)
            )

    return list(
        dict.fromkeys(candidates)
    )


def spatial_container_candidates(
    rec_texts: list[str],
    rec_scores: list[float],
    rec_boxes: list,
) -> list[str]:
    """
    Generate container-number candidates using OCR text
    together with their bounding boxes.

    This is useful when a difficult photograph causes
    PaddleOCR to detect parts of the container number
    as separate OCR boxes.

    Example:

        CMAU
        980286

    can produce:

        CMAU980286

    which can then be completed using the ISO 6346
    check-digit calculation.
    """

    items = []

    # ---------------------------------------------------------
    # Build structured OCR items
    # ---------------------------------------------------------

    for text, score, box in zip(
        rec_texts,
        rec_scores,
        rec_boxes,
    ):
        normalized = normalize_text(text)

        if not normalized:
            continue

        if not box or len(box) < 4:
            continue

        try:
            x1, y1, x2, y2 = map(
                float,
                box[:4],
            )
        except (TypeError, ValueError):
            continue

        items.append(
            {
                "text": normalized,
                "score": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
            }
        )

    candidates = []

    # ---------------------------------------------------------
    # Find possible 4-character owner/category prefixes.
    #
    # Example:
    #
    # CMAU
    #
    # First 3 = letters
    # Fourth   = U/J/Z
    # ---------------------------------------------------------

    prefixes = []

    for item in items:
        text = item["text"]

        if (
            len(text) == 4
            and text[:3].isalpha()
            and text[3] in "UJZ"
        ):
            prefixes.append(item)

    # ---------------------------------------------------------
    # Search for nearby numeric OCR boxes.
    # ---------------------------------------------------------

    for prefix in prefixes:

        nearby = []

        for item in items:

            if item is prefix:
                continue

            text = item["text"]

            # We need at least one digit in the
            # potential serial-number box.
            if not any(
                char.isdigit()
                for char in text
            ):
                continue

            dx = abs(
                item["cx"] - prefix["cx"]
            )

            dy = abs(
                item["cy"] - prefix["cy"]
            )

            # Don't combine completely unrelated
            # OCR boxes elsewhere in the image.
            if dx > 500 or dy > 500:
                continue

            nearby.append(
                (
                    dx + dy,
                    item,
                )
            )

        nearby.sort(
            key=lambda value: value[0]
        )

        # -----------------------------------------------------
        # Try the closest numeric OCR boxes first.
        # -----------------------------------------------------

        for _, item in nearby:

            numeric = "".join(
                char
                for char in item["text"]
                if char.isdigit()
            )

            # Complete 7-digit serial.
            if len(numeric) == 7:
                candidates.append(
                    prefix["text"] + numeric
                )

            # Six digits + missing ISO check digit.
            #
            # The ISO completion logic in iso6346.py
            # will calculate the final digit.
            elif len(numeric) == 6:
                candidates.append(
                    prefix["text"] + numeric
                )

    return list(
        dict.fromkeys(candidates)
    )