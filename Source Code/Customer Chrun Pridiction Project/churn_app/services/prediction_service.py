from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from churn_app.extensions import db
from churn_app.models import Company, CompanyDataset, TrainedModel
from churn_app.services.schema_service import infer_schema


def get_active_model(company: Company) -> TrainedModel | None:
    return TrainedModel.query.filter_by(company_id=company.id, is_active=True).order_by(TrainedModel.version.desc()).first()


def get_active_dataset(company: Company) -> CompanyDataset | None:
    return CompanyDataset.query.filter_by(company_id=company.id, is_active=True).order_by(CompanyDataset.uploaded_at.desc()).first()


def get_dataset_schema(dataset: CompanyDataset) -> dict[str, Any]:
    if not dataset.data_path:
        raise ValueError(f"The active dataset for {dataset.company.name} has no data file.")
    data_path = Path(dataset.data_path)
    if not data_path.is_file():
        raise ValueError(f"The active dataset file for {dataset.company.name} is unavailable.")

    frame = pd.read_csv(data_path)
    stored_schema = dataset.schema_json or {}
    target_column = stored_schema.get("target_column") or "Churn"
    if target_column not in frame.columns:
        raise ValueError(f"The active dataset for {dataset.company.name} has no configured target column.")

    schema = infer_schema(frame, target_column)
    id_columns = list(dict.fromkeys(schema["id_columns"] + stored_schema.get("id_columns", [])))
    feature_columns = [
        column for column in frame.columns
        if column != target_column and column not in id_columns
    ]
    schema["id_columns"] = id_columns
    schema["feature_columns"] = feature_columns
    schema["types"] = {column: schema["types"][column] for column in feature_columns}
    schema["allowed_categories"] = {
        column: schema["allowed_categories"][column]
        for column in feature_columns if schema["types"][column] == "categorical"
    }
    schema["numeric_ranges"] = {
        column: schema["numeric_ranges"][column]
        for column in feature_columns if column in schema["numeric_ranges"]
    }
    schema["integer_columns"] = [
        column for column in feature_columns if column in schema["integer_columns"]
    ]
    return schema


def validate_input_against_schema(
    company: Company,
    input_dict: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = get_active_dataset(company)
    if dataset is None:
        raise ValueError(f"No dataset has been uploaded for {company.name}.")

    schema = schema or get_dataset_schema(dataset)
    required = schema.get("feature_columns", [])
    types = schema.get("types", schema.get("column_types", {}))
    ranges = schema.get("numeric_ranges", {})
    categories = schema.get("allowed_categories", {})
    integer_columns = set(schema.get("integer_columns", []))
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    unexpected = sorted(set(input_dict) - set(required))
    if unexpected:
        errors.append(f"Unexpected column(s): {', '.join(unexpected)}.")

    for column in required:
        if column not in input_dict:
            errors.append(f"Missing required field: {column}.")
            continue
        value = input_dict[column]
        if value is None or pd.isna(value) or str(value).strip() == "":
            errors.append(f"Field {column} cannot be empty.")
            continue

        if types.get(column) == "categorical":
            normalized = str(value).strip()
            allowed = [str(item) for item in categories.get(column, [])]
            if normalized not in allowed:
                errors.append(f"Invalid value for {column}. Allowed values: {', '.join(allowed)}.")
            else:
                cleaned[column] = normalized
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            errors.append(f"Field {column} must be numeric.")
            continue
        if not math.isfinite(numeric_value):
            errors.append(f"Field {column} must be a finite number.")
            continue
        if column in integer_columns and not numeric_value.is_integer():
            errors.append(f"Field {column} must be a whole number.")
            continue

        limits = ranges.get(column)
        if limits:
            if numeric_value < float(limits["min"]) or numeric_value > float(limits["max"]):
                errors.append(
                    f"Invalid value for {column}. Enter a value from "
                    f"{limits['min']:g} to {limits['max']:g}, based on this company's dataset."
                )
                continue
        cleaned[column] = int(numeric_value) if column in integer_columns else numeric_value

    if errors:
        raise ValueError(" ".join(errors))
    return cleaned


def _risk_from_probability(probability: float, threshold_medium: float, threshold_high: float) -> str:
    if probability >= threshold_high:
        return "High Risk"
    if probability >= threshold_medium:
        return "Medium Risk"
    return "Low Risk"


def _explain_prediction(company: Company, pipeline, input_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_names = []
    try:
        if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
            feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        feature_names = list(input_frame.columns)

    importance = []
    if hasattr(pipeline, "named_steps") and hasattr(pipeline.named_steps["model"], "feature_importances_"):
        if feature_names:
            for name, value in zip(feature_names, pipeline.named_steps["model"].feature_importances_):
                importance.append((name, float(value)))
    if not importance:
        return ([], [])

    ranked = sorted(importance, key=lambda item: abs(item[1]), reverse=True)[:5]
    risk_factors = [f"{name} influences churn risk" for name, _ in ranked]
    suggested_actions = ["Review the customer account proactively", "Offer a retention plan for the highest-risk drivers", "Schedule outreach within 24 hours"]
    return risk_factors, suggested_actions


def predict_one(
    company: Company,
    input_dict: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = get_active_dataset(company)
    if dataset is None:
        raise ValueError(f"No dataset has been uploaded for {company.name}.")
    schema = schema or get_dataset_schema(dataset)
    cleaned = validate_input_against_schema(company, input_dict, schema)
    model_record = get_active_model(company)
    if model_record is None:
        raise ValueError(f"No active model exists for {company.name}. An administrator must train a model first.")
    if model_record.dataset_id != dataset.id:
        raise ValueError(f"The active model for {company.name} was trained on an older dataset. Retrain the model before predicting.")

    pipeline = joblib.load(model_record.model_path)
    frame = pd.DataFrame([cleaned], columns=schema["feature_columns"])
    prediction = pipeline.predict(frame)[0]
    probability = float(pipeline.predict_proba(frame)[0, 1])
    risk_level = _risk_from_probability(probability, model_record.threshold_medium, model_record.threshold_high)
    factors, actions = _explain_prediction(company, pipeline, frame)
    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": probability,
        "risk_level": risk_level,
        "risk_factors": factors[:3] or ["Customer profile is in a normal retention range."],
        "suggested_actions": actions[:3],
    }


def predict_many(company: Company, frame: pd.DataFrame) -> pd.DataFrame:
    model_record = get_active_model(company)
    dataset = get_active_dataset(company)
    if model_record is None:
        raise ValueError(f"No active model exists for {company.name}.")
    if dataset is None or model_record.dataset_id != dataset.id:
        raise ValueError(f"The active model for {company.name} is not trained on its active dataset.")
    pipeline = joblib.load(model_record.model_path)
    probabilities = pipeline.predict_proba(frame)[:, 1]
    predictions = (probabilities >= model_record.threshold_medium).astype(int)
    result = frame.copy()
    result["Prediction"] = ["Yes" if value == 1 else "No" for value in predictions]
    result["Probability"] = probabilities
    result["RiskLevel"] = [
        _risk_from_probability(value, model_record.threshold_medium, model_record.threshold_high)
        for value in probabilities
    ]
    return result
