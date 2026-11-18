from __future__ import annotations

import csv
import io
from typing import Any

import openpyxl


def sanitize_cell(value: Any) -> Any:
    """Protect against CSV/Excel formula injection.
    Prefix any text cell starting with '=', '+', '-', or '@' with an apostrophe.
    """
    if isinstance(value, str) and value:
        if value[0] in ("=", "+", "-", "@"):
            return "'" + value
    return value


def build_csv(columns: list[str], rows: list[Any]) -> bytes:
    """Build CSV bytes from columns and rows, sanitizing cells against formula injection."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([sanitize_cell(col) for col in columns])
    for row in rows:
        if isinstance(row, dict):
            row_vals = [sanitize_cell(row.get(col, "")) for col in columns]
        else:
            row_vals = [sanitize_cell(val) for val in row]
        writer.writerow(row_vals)
    return output.getvalue().encode("utf-8")


def build_xlsx(columns: list[str], rows: list[Any], sheet_name: str = "Predictions") -> bytes:
    """Build XLSX bytes from columns and rows, sanitizing cells against formula injection."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = (sheet_name or "Sheet1")[:31]
    ws.append([sanitize_cell(col) for col in columns])
    for row in rows:
        if isinstance(row, dict):
            row_vals = [sanitize_cell(row.get(col, "")) for col in columns]
        else:
            row_vals = [sanitize_cell(val) for val in row]
        ws.append(row_vals)
    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()
