from __future__ import annotations


def kt_cgs_from_row(row: dict) -> float | None:
    s = (row.get("kt_cgs") or row.get("kc_cgs") or "").strip()
    return float(s) if s else None
