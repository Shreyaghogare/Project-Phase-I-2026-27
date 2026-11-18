from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _normalise_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def _detect_positive_label(series: pd.Series) -> str:
    values = [value for value in series.dropna().unique().tolist() if str(value).strip() != ""]
    if not values:
        return "yes"
    normalized = {_normalise_label(value): value for value in values}
    for candidate in ["yes", "y", "true", "1", "churned", "churn", "active"]:
        if candidate in normalized:
            return normalized[candidate]
    for candidate in ["1", "yes", "true", "churn"]:
        if candidate in {str(v).lower() for v in values}:
            return next(v for v in values if str(v).lower() == candidate)
    return str(values[0])


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _is_identifier_column(column: Any) -> bool:
    name = str(column).strip()
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    known_identifiers = {
        "id", "customerid", "userid", "policyid", "accountid",
        "subscriberid", "clientid", "memberid", "recordid",
    }
    return (
        normalized in known_identifiers
        or bool(re.search(r"(?:^|[_\s-])id$", name, re.IGNORECASE))
        or bool(re.search(r"[a-z0-9]ID$", name))
    )


def infer_schema(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    id_columns = [
        column
        for column in df.columns
        if _is_identifier_column(column)
    ]
    feature_columns = [column for column in df.columns if column != target_col and column not in id_columns]

    types: dict[str, str] = {}
    allowed_categories: dict[str, list[str]] = {}
    numeric_ranges: dict[str, dict[str, float]] = {}
    integer_columns: list[str] = []

    for column in feature_columns:
        series = df[column]
        kind = _column_kind(series)
        types[column] = kind
        if kind == "categorical":
            categories = [str(value) for value in pd.Series(series.dropna().unique()).tolist()]
            allowed_categories[column] = sorted(set(categories), key=lambda item: item.lower())
        else:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                low, high = values.min(), values.max()
                numeric_ranges[column] = {
                    "min": float(low),
                    "max": float(high),
                }
                if (values % 1 == 0).all():
                    integer_columns.append(column)

    positive_value = _detect_positive_label(df[target_col].dropna())
    target_values = df[target_col].dropna()
    normalized = target_values.map(lambda value: 1 if _normalise_label(value) == _normalise_label(positive_value) else 0)

    return {
        "columns": list(df.columns),
        "id_columns": id_columns,
        "target_column": target_col,
        "positive_label": str(positive_value),
        "feature_columns": feature_columns,
        "types": types,
        "integer_columns": integer_columns,
        "allowed_categories": allowed_categories,
        "numeric_ranges": numeric_ranges,
        "target_values": normalized.to_dict(),
        "target_values_series": normalized.astype(int),
    }
