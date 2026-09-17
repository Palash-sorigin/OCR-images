from __future__ import annotations


class CrossFieldValidator:
    def validate(self, fields: dict) -> list[dict]:
        conflicts: list[dict] = []

        def value(name: str):
            item = fields.get(name, {})
            return item.get("value") if isinstance(item, dict) else None

        count = value("container_count")
        container_values = [
            value("export_container_no_1"),
            value("export_container_no_2"),
            value("dpd_container_no_1"),
            value("dpd_container_no_2"),
        ]
        populated = [v for v in container_values if v]
        if count and str(count).isdigit():
            declared = int(count)
            if populated and len(populated) != declared:
                conflicts.append({
                    "type": "CONTAINER_COUNT_MISMATCH",
                    "field": "container_count",
                    "declared": declared,
                    "detected": len(populated),
                    "reason": "Declared container count does not match populated container fields.",
                })

        transaction = (value("transaction_type") or "").replace(" ", "").upper()
        export_values = [value("export_container_no_1"), value("export_container_no_2")]
        if "ONLYIMPORT" in transaction and any(export_values):
            conflicts.append({
                "type": "TRANSACTION_EXPORT_CONFLICT",
                "field": "transaction_type",
                "reason": "Transaction is Only Import but an export container field contains a value.",
            })

        return conflicts
