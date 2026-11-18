from __future__ import annotations

from typing import Any

import pandas as pd


def build_data_quality_report(frame: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    rows = int(len(frame))
    columns = list(frame.columns)
    missing_pct = {column: round(float(frame[column].isna().mean() * 100), 2) for column in columns}
    duplicate_rows = int(frame.duplicated().sum())

    constant_columns = [
        col for col in columns if frame[col].nunique(dropna=True) <= 1
    ]
    high_cardinality = [
        col for col in columns if frame[col].nunique(dropna=True) > 25 and frame[col].dtype == object
    ]

    warnings: list[str] = []
    if duplicate_rows > 0:
        warnings.append(f"Duplicate rows detected: {duplicate_rows} records." )
    if constant_columns:
        warnings.append(f"Constant columns detected: {', '.join(constant_columns)}.")
    if high_cardinality:
        warnings.append(f"High-cardinality columns detected: {', '.join(high_cardinality)}.")

    class_balance = None
    if target_col and target_col in frame.columns:
        class_counts = frame[target_col].dropna().value_counts().to_dict()
        class_balance = {str(key): int(value) for key, value in class_counts.items()}
        if class_counts:
            positive_total = sum(value for key, value in class_counts.items() if str(key).lower() in {"yes", "y", "true", "1", "churned", "churn"})
            total = sum(class_counts.values())
            if total:
                positive_share = positive_total / total
                if positive_share > 0.85:
                    warnings.append("Churn is highly imbalanced: the positive class is over 85% of targets.")
                elif positive_share < 0.15:
                    warnings.append("Churn is unusually low for the positive class; consider checking the target encoding.")

    outliers: dict[str, float] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty or series.nunique() <= 1:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_fraction = float(((series < lower) | (series > upper)).mean())
        if outlier_fraction > 0.02:
            outliers[column] = round(outlier_fraction * 100, 2)

    return {
        "rows": rows,
        "columns": columns,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "high_cardinality": high_cardinality,
        "class_balance": class_balance,
        "outliers": outliers,
        "warnings": warnings,
    }
