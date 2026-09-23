from __future__ import annotations


class DocumentClassifier:
    """Keyword-scored sub-type classifier, generalized from ScreenClassifier
    so every document route can reuse one implementation instead of five
    near-identical copies."""

    def __init__(self, rules: dict[str, list[str]] | None, default_subtype: str) -> None:
        self.rules = rules or {}
        self.default_subtype = default_subtype

    def classify(self, text: str) -> dict:
        if not self.rules:
            return {"subtype": self.default_subtype, "confidence": 1.0, "evidence": []}

        upper = text.upper()
        scores = {subtype: 0 for subtype in self.rules}
        evidence: list[str] = []
        for subtype, keywords in self.rules.items():
            for keyword in keywords:
                if keyword in upper:
                    scores[subtype] += 1
                    evidence.append(keyword)

        if max(scores.values(), default=0) == 0:
            return {"subtype": self.default_subtype, "confidence": 0.0, "evidence": []}

        subtype = max(scores, key=scores.get)
        confidence = min(0.99, 0.5 + 0.15 * scores[subtype])
        return {"subtype": subtype, "confidence": confidence, "evidence": evidence}
