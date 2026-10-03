from __future__ import annotations


def kt_cgs_from_row(row: dict) -> float | None:
    s = (row.get("kt_cgs") or row.get("kc_cgs") or "").strip()
    return float(s) if s else None


SIZE_RATIO_COL = "size_ratio"
LEGACY_SIZE_RATIO_COL = "dispersion_ratio"  # pre-2026-10-01 batch CSVs


def size_ratio_col(fieldnames) -> str:
    """Size-ratio column present in ``fieldnames`` (legacy ``dispersion_ratio`` if only that exists)."""
    if SIZE_RATIO_COL not in fieldnames and LEGACY_SIZE_RATIO_COL in fieldnames:
        return LEGACY_SIZE_RATIO_COL
    return SIZE_RATIO_COL


def size_ratio_raw(row: dict) -> str:
    """Raw size-ratio cell; KeyError if neither column exists (same as ``row[col]``)."""
    return row[size_ratio_col(row)]
