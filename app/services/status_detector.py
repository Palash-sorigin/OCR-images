from __future__ import annotations


class StatusDetector:
    def detect(self, text: str) -> dict:
        upper = text.upper()
        if "PIN SUCCESSFULLY GENERATED" in upper:
            return {"status": "SUCCESS", "message": "PIN successfully generated."}
        if "TRUCK BOOKING SUCCEEDED" in upper:
            return {"status": "SUCCESS", "message": "Truck booking succeeded."}
        if "FAILED" in upper or "ERROR" in upper:
            return {"status": "FAILURE", "message": None}
        return {"status": "UNKNOWN", "message": None}
