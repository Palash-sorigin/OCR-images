from __future__ import annotations


class ScreenClassifier:
    """Lightweight deterministic classifier based on visible OCR labels."""

    def classify(self, text: str) -> dict:
        upper = text.upper()
        scores = {
            "PIN_GENERATION_PSA": 0,
            "PIN_GENERATION_APM": 0,
            "PIN_GENERATION_NSFT": 0,
            "TRUCK_BOOKING_NSFT": 0,
        }

        if "PIN GENERATION" in upper:
            scores["PIN_GENERATION_PSA"] += 1
            scores["PIN_GENERATION_APM"] += 1
            scores["PIN_GENERATION_NSFT"] += 1
        if "PSA MUMBAI" in upper:
            scores["PIN_GENERATION_PSA"] += 5
        if "APM TERMINALS" in upper:
            scores["PIN_GENERATION_APM"] += 5
        if "NHAVA SHEVA FREEPORT TERMINAL" in upper or "NSFT" in upper:
            scores["PIN_GENERATION_NSFT"] += 4
            scores["TRUCK_BOOKING_NSFT"] += 1
        if "TRUCK BOOKING" in upper:
            scores["TRUCK_BOOKING_NSFT"] += 6

        if max(scores.values(), default=0) == 0:
            return {"screen_type": "UNKNOWN", "confidence": 0.0, "evidence": []}

        screen_type = max(scores, key=scores.get)
        top = scores[screen_type]
        confidence = min(0.99, 0.50 + 0.08 * top)
        evidence = []
        for token in (
            "PIN GENERATION",
            "PSA MUMBAI",
            "APM TERMINALS",
            "NHAVA SHEVA FREEPORT TERMINAL",
            "NSFT",
            "TRUCK BOOKING",
        ):
            if token in upper:
                evidence.append(token)

        return {
            "screen_type": screen_type,
            "confidence": confidence,
            "evidence": evidence,
        }
