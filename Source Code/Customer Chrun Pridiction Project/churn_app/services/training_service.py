from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from churn_app.extensions import db
from churn_app.models import Company, CompanyDataset, TrainedModel
from churn_app.services.schema_service import infer_schema

ROOT = Path(__file__).resolve().parents[2]


def _choose_model_candidates():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }


def _build_preprocessor(X: pd.DataFrame, schema: dict) -> ColumnTransformer:
    numeric_columns = [column for column in X.columns if schema["types"].get(column, "numeric") == "numeric"]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns)
        )
    if categorical_columns:
        transformers.append(
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def train_company_model(company_id: int, dataset_id: int) -> TrainedModel:
    company = db.session.get(Company, company_id)
    dataset_record = db.session.get(CompanyDataset, dataset_id)
    if company is None:
        raise ValueError(f"Unknown company id: {company_id}")
    if dataset_record is None:
        raise ValueError(f"Unknown dataset id: {dataset_id}")
    if dataset_record.company_id != company_id:
        raise ValueError("Dataset does not belong to this company.")

    frame = pd.read_csv(dataset_record.data_path)
    schema_target = dataset_record.schema_json.get("target_column") or "Churn"
    schema = infer_schema(frame, schema_target)
    feature_columns = [column for column in frame.columns if column not in {schema_target, *schema["id_columns"]}]
    X = frame[feature_columns].copy()
    y_raw = frame[schema_target]
    y = y_raw.map(lambda value: 1 if str(value).strip().lower() == str(schema["positive_label"]).strip().lower() else 0).astype(int)

    if y.nunique() < 2:
        raise ValueError("The target column has too little variation for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    comparison = []
    best_pipeline = None
    best_score = -1.0
    best_name = ""

    for name, estimator in _choose_model_candidates().items():
        pipeline = Pipeline([
            ("preprocessor", _build_preprocessor(X_train, schema)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        record = {
            "name": name,
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "roc_auc": _safe_roc_auc(y_test.to_numpy(), test_prob),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        }
        comparison.append(record)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        record["cv_mean"] = float(scores.mean())
        record["cv_std"] = float(scores.std())

        ranking_score = (record["roc_auc"], record["f1"])
        if ranking_score > (best_score, -1.0):
            best_score = record["roc_auc"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise ValueError("No model could be trained for this company dataset.")

    latest_version = db.session.query(TrainedModel.version).filter_by(company_id=company_id).order_by(TrainedModel.version.desc()).first()
    next_version = (latest_version[0] if latest_version else 0) + 1
    model_dir = ROOT / "models" / company.slug
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"v{next_version}.joblib"
    joblib.dump(best_pipeline, model_path)

    TrainedModel.query.filter_by(company_id=company_id, is_active=True).update({"is_active": False})
    best_predictions = best_pipeline.predict(X_test)
    best_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    best_model = TrainedModel(
        company_id=company_id,
        dataset_id=dataset_id,
        algorithm=best_name,
        version=next_version,
        metrics_json={
            "accuracy": float(accuracy_score(y_test, best_predictions)),
            "precision": float(precision_score(y_test, best_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, best_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, best_predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, best_probabilities)),
        },
        feature_importance_json={"feature_importance": "pipeline_based"},
        model_path=str(model_path),
        is_active=True,
        threshold_medium=0.3,
        threshold_high=0.6,
    )
    db.session.add(best_model)
    db.session.commit()
    return best_model
