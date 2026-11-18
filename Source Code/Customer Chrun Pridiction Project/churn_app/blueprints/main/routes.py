import csv
import html
import io
import math
import smtplib
from datetime import date, datetime, time, timedelta
from email.message import EmailMessage
from pathlib import Path

import joblib
import pandas as pd
from flask import abort, current_app, flash, make_response, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required
from flask_wtf import FlaskForm
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import or_
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from wtforms import SelectField, StringField
from wtforms.validators import InputRequired

from ...extensions import db
from ...forms import BulkUploadForm, ChangePasswordForm, EmailHistoryForm, PredictionForm, ProfileForm
from ...models import Company, LoginLog, Prediction, User, UserCompanyAccess, utc_now
from ...services.email_service import (
    EmailConfigurationError,
    check_outgoing_email_rate_limit,
    record_outgoing_email_sent,
    send_email,
    send_export_email,
)
from ...services.export_service import build_csv, build_xlsx
from ...services.prediction_service import (
    get_active_dataset,
    get_active_model,
    get_dataset_schema,
    predict_one,
    predict_many,
    validate_input_against_schema,
)
from . import bp

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_DIR / "cleaned_dataset.csv"
MODEL_PATH = PROJECT_DIR / "churn_model.pkl"
ENCODERS_PATH = PROJECT_DIR / "encoders.pkl"
FEATURES = ["Age", "Gender", "Tenure", "MonthlyCharges", "ContractType", "InternetService", "TotalCharges", "TechSupport"]
CATEGORICAL_FEATURES = ["Gender", "ContractType", "InternetService", "TechSupport"]

try:
    dataset = pd.read_csv(DATA_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    model = joblib.load(MODEL_PATH)
    model_error = None
except Exception as error:
    dataset = pd.DataFrame()
    encoders = {}
    model = None
    model_error = str(error)


def encoded_dataset():
    frame = dataset.copy()
    for column in CATEGORICAL_FEATURES + ["Churn"]:
        frame[column] = encoders[column].transform(frame[column])
    return frame


def metric_values(actual, predicted):
    return {
        "accuracy": round(accuracy_score(actual, predicted) * 100, 2),
        "precision": round(precision_score(actual, predicted, zero_division=0) * 100, 2),
        "recall": round(recall_score(actual, predicted, zero_division=0) * 100, 2),
        "f1": round(f1_score(actual, predicted, zero_division=0) * 100, 2),
    }


def evaluation_data():
    frame = encoded_dataset()
    x = frame[FEATURES]
    y = frame["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "Existing Random Forest": model,
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    }
    results = {}
    matrix = [[0, 0], [0, 0]]
    for name, candidate in candidates.items():
        if candidate is None:
            continue
        if name != "Existing Random Forest":
            candidate.fit(x_train, y_train)
        predictions = candidate.predict(x_test)
        results[name] = metric_values(y_test, predictions)
        if name == "Existing Random Forest":
            matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    return results, matrix


def select_options(column):
    encoder = encoders.get(column)
    if encoder is None:
        return []
    classes = getattr(encoder, "classes_", [])
    return [str(value) for value in classes]


def form_values():
    keys = ["age", "gender", "tenure", "monthly", "contract", "internet", "total", "support"]
    return {key: request.form.get(key, "") for key in keys}


def risk_factors(values):
    factors = []
    if values["tenure"] < 12:
        factors.append("Short tenure")
    if values["contract"] == "month-to-month":
        factors.append("Month-to-month contract")
    if values["monthly"] >= dataset["MonthlyCharges"].median():
        factors.append("Higher monthly charges")
    if values["support"] == "no":
        factors.append("No technical support")
    return factors or ["No common risk factor identified from these inputs"]


def retention_actions(risk):
    if risk == "High Risk":
        return ["Contact the customer", "Consider a suitable contract plan", "Offer technical support if applicable", "Consider a personalized retention offer"]
    if risk == "Medium Risk":
        return ["Monitor customer activity", "Provide proactive support", "Encourage a longer-term plan"]
    return ["Continue regular customer engagement", "Keep service support easy to access"]


def prediction_form():
    form = PredictionForm()
    form.gender.choices = [(value, value.title()) for value in select_options("Gender")]
    form.contract.choices = [(value, value.replace("-", " ").title()) for value in select_options("ContractType")]
    form.internet.choices = [(value, value.title()) for value in select_options("InternetService")]
    form.support.choices = [(value, value.title()) for value in select_options("TechSupport")]
    return form


def predict_customer(values):
    numeric_values = {
        "age": int(values["age"]),
        "tenure": int(values["tenure"]),
        "monthly": float(values["monthly"]),
        "total": float(values["total"]),
    }
    if not 18 <= numeric_values["age"] <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if not 0 <= numeric_values["tenure"] <= 120:
        raise ValueError("Tenure must be between 0 and 120 months.")
    if not all(math.isfinite(numeric_values[key]) for key in ("monthly", "total")):
        raise ValueError("Charges must be finite numbers.")
    if numeric_values["monthly"] < 0 or numeric_values["total"] < 0:
        raise ValueError("Charges cannot be negative.")

    categories = {
        "gender": "Gender",
        "contract": "ContractType",
        "internet": "InternetService",
        "support": "TechSupport",
    }
    for form_key, encoder_key in categories.items():
        if values[form_key] not in select_options(encoder_key):
            raise ValueError(f"Invalid {form_key.replace('_', ' ')} value.")

    encoded = [
        numeric_values["age"],
        encoders["Gender"].transform([values["gender"]])[0],
        numeric_values["tenure"],
        numeric_values["monthly"],
        encoders["ContractType"].transform([values["contract"]])[0],
        encoders["InternetService"].transform([values["internet"]])[0],
        numeric_values["total"],
        encoders["TechSupport"].transform([values["support"]])[0],
    ]
    input_data = pd.DataFrame([encoded], columns=FEATURES)
    predicted_class = int(model.predict(input_data)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probability = float(model.predict_proba(input_data)[0][classes.index(1)])
    risk = "High Risk" if probability is not None and probability >= 0.6 else "Medium Risk" if probability is not None and probability >= 0.3 else "Low Risk"
    analysis_values = {
        "tenure": numeric_values["tenure"],
        "monthly": numeric_values["monthly"],
        "contract": values["contract"],
        "support": values["support"],
    }
    return {
        "prediction": "Yes" if predicted_class == 1 else "No",
        "probability": probability,
        "risk": risk,
        "factors": risk_factors(analysis_values),
        "actions": retention_actions(risk),
        "customer": {
            "Age": numeric_values["age"],
            "Gender": values["gender"],
            "Tenure": numeric_values["tenure"],
            "Monthly Charges": numeric_values["monthly"],
            "Contract Type": values["contract"],
            "Internet Service": values["internet"],
            "Total Charges": numeric_values["total"],
            "Tech Support": values["support"],
        },
        "inputs": {
            "age": numeric_values["age"],
            "gender": values["gender"],
            "tenure": numeric_values["tenure"],
            "monthly_charges": numeric_values["monthly"],
            "total_charges": numeric_values["total"],
            "contract_type": values["contract"],
            "internet_service": values["internet"],
            "tech_support": values["support"],
        },
    }


def save_prediction(user_id, result):
    active_company = get_active_company()
    db.session.add(Prediction(
        user_id=user_id,
        company_id=active_company.id if active_company else None,
        **result["inputs"],
        prediction=result["prediction"],
        probability=result["probability"] if result["probability"] is not None else 0.0,
        risk_level=result["risk"],
        risk_factors=result["factors"],
        suggested_actions=result["actions"],
    ))


class CompanyPredictionForm(FlaskForm):
    record_id = StringField("Customer or policy ID", validators=[InputRequired(message="Choose a record from this company's dataset.")])


def build_manual_prediction_form(schema):
    attributes = {}
    fields = []
    for index, column in enumerate(schema["feature_columns"]):
        name = f"feature_{index}"
        validators = [InputRequired(message=f"{column} is required.")]
        kind = schema["types"].get(column, "numeric")
        if kind == "categorical":
            choices = [("", f"Choose {column}")]
            choices.extend((value, value) for value in schema["allowed_categories"].get(column, []))
            attributes[name] = SelectField(column, choices=choices, validators=validators)
        else:
            attributes[name] = StringField(column, validators=validators)
        fields.append({
            "name": name,
            "column": column,
            "kind": kind,
            "limits": schema["numeric_ranges"].get(column),
            "integer": column in schema["integer_columns"],
        })

    form_type = type("ManualCompanyPredictionForm", (FlaskForm,), attributes)
    form = form_type()
    for field in fields:
        field["input"] = form[field["name"]]
    return form, fields


def company_record_data(company, schema):
    dataset_record = get_active_dataset(company)
    if dataset_record is None:
        raise ValueError(f"No active dataset is available for {company.name}.")
    frame = pd.read_csv(dataset_record.data_path)
    feature_columns = schema["feature_columns"]
    complete_records = frame.dropna(subset=feature_columns)
    record_id_column = next(iter(schema["id_columns"]), None)
    if record_id_column:
        complete_records = complete_records.dropna(subset=[record_id_column])
        duplicate_ids = complete_records[record_id_column].astype(str).duplicated(keep=False)
        complete_records = complete_records.loc[~duplicate_ids]
        options = [
            {"value": str(row[record_id_column]), "label": f"{row[record_id_column]} · {row[feature_columns[0]]}"}
            for _, row in complete_records.iterrows()
        ]
        record_label = record_id_column
    else:
        options = [
            {"value": f"row:{index + 2}", "label": f"CSV row {index + 2}"}
            for index in complete_records.index
        ]
        record_label = "Dataset row"
    return frame, complete_records, record_id_column, options, record_label


def render_company_prediction(
    company,
    form=None,
    result=None,
    values=None,
    selected_record=None,
    mode="id",
    manual_fields=None,
    status=200,
):
    schema = None
    schema_error = None
    model_available = False
    record_options = []
    record_label = "Customer or policy ID"
    if company is None:
        schema_error = "Choose a company workspace from the dashboard before predicting."
    else:
        try:
            dataset_record = get_active_dataset(company)
            if dataset_record is None:
                raise ValueError(f"No active dataset is available for {company.name}.")
            schema = get_dataset_schema(dataset_record)
            _frame, _complete_records, _record_id_column, record_options, record_label = company_record_data(company, schema)
            if not record_options:
                raise ValueError(f"The active dataset for {company.name} has no complete records to predict.")
            if form is None:
                if mode == "manual":
                    form, manual_fields = build_manual_prediction_form(schema)
                else:
                    form = CompanyPredictionForm()
            model_record = get_active_model(company)
            model_available = bool(model_record and model_record.dataset_id == dataset_record.id)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            schema_error = str(error)

    return render_template(
        "prediction.html",
        form=form,
        record_options=record_options,
        record_label=record_label,
        record_count=len(record_options),
        mode=mode,
        manual_fields=manual_fields or [],
        schema_mode=True,
        schema_error=schema_error,
        model_available=model_available,
        active_company=company,
        result=result,
        result_values=values or {},
        selected_record=selected_record,
    ), status


def predict_for_active_company(company, mode="id"):
    if company is None:
        flash("Choose a company workspace from the dashboard before predicting.", "error")
        return render_company_prediction(None, status=400)

    try:
        dataset_record = get_active_dataset(company)
        if dataset_record is None:
            raise ValueError(f"No active dataset is available for {company.name}.")
        schema = get_dataset_schema(dataset_record)
        frame, complete_records, record_id_column, record_options, record_label = company_record_data(company, schema)
        if not record_options:
            raise ValueError(f"The active dataset for {company.name} has no complete records to predict.")
        if mode == "manual":
            form, manual_fields = build_manual_prediction_form(schema)
        else:
            form, manual_fields = CompanyPredictionForm(), []
    except (OSError, ValueError, pd.errors.ParserError) as error:
        flash(str(error), "error")
        return render_company_prediction(company, status=400)

    if not form.validate_on_submit():
        for field_name, errors in form.errors.items():
            if field_name == "csrf_token":
                continue
            label = form[field_name].label.text
            for error in errors:
                flash(f"Data is wrong: {label}: {error}", "error")
        return render_company_prediction(company, form, mode=mode, manual_fields=manual_fields, status=400)

    accepted_fields = {"csrf_token", "submit", "mode"}
    accepted_fields.add("record_id" if mode == "id" else None)
    if mode == "manual":
        accepted_fields.update(field["name"] for field in manual_fields)
    accepted_fields.discard(None)
    unexpected_fields = set(request.form) - accepted_fields
    if unexpected_fields:
        flash("Data is wrong: submitted fields do not match this prediction mode.", "error")
        return render_company_prediction(company, form, mode=mode, manual_fields=manual_fields, status=400)

    if mode == "manual":
        submitted_values = {
            field["column"]: form[field["name"]].data
            for field in manual_fields
        }
        submitted_record_id = ""
    else:
        submitted_record_id = form.record_id.data.strip()
        submitted_values = None
        if record_id_column:
            matches = complete_records.loc[
                complete_records[record_id_column].astype(str).str.strip() == submitted_record_id
            ]
        elif submitted_record_id.startswith("row:"):
            try:
                source_index = int(submitted_record_id.removeprefix("row:")) - 2
            except ValueError:
                source_index = -1
            matches = complete_records.loc[complete_records.index == source_index]
        else:
            matches = complete_records.iloc[0:0]
        if len(matches) != 1:
            flash("Data is wrong: choose an existing complete record from the active company's dataset.", "error")
            return render_company_prediction(company, form, mode=mode, manual_fields=manual_fields, status=400)
        source_record = matches.iloc[0]
        submitted_values = source_record[schema["feature_columns"]].to_dict()

    try:
        cleaned_values = validate_input_against_schema(company, submitted_values, schema)
    except ValueError as error:
        flash(f"Data is wrong: {error}", "error")
        return render_company_prediction(company, form, mode=mode, manual_fields=manual_fields, status=400)

    if mode == "manual":
        matches = complete_records
        for column in schema["feature_columns"]:
            if schema["types"].get(column) == "categorical":
                matches = matches.loc[matches[column].astype(str).str.strip() == str(cleaned_values[column])]
            else:
                expected = float(cleaned_values[column])
                numeric = pd.to_numeric(matches[column], errors="coerce")
                tolerance = max(1e-9, abs(expected) * 1e-9)
                matches = matches.loc[(numeric - expected).abs() <= tolerance]
        if matches.empty:
            flash(
                "Data is wrong: manual values must match one complete row in this company's dataset. "
                "Use Select by ID or Bulk CSV for new customers.",
                "error",
            )
            return render_company_prediction(company, form, mode=mode, manual_fields=manual_fields, status=400)
        source_record = matches.iloc[0]
        submitted_record_id = (
            str(source_record[record_id_column])
            if record_id_column
            else f"row:{source_record.name + 2}"
        )

    try:
        model_record = get_active_model(company)
        result = predict_one(company, cleaned_values, schema)
        probability = result.get("probability")
        if probability is None or not math.isfinite(float(probability)):
            raise ValueError("The active model did not return a valid probability. Please retrain the company model.")
        result["probability"] = float(probability)
        normalized_columns = {"".join(character for character in key.lower() if character.isalnum()): value for key, value in cleaned_values.items()}
        mapped_inputs = {}
        for source, destination in {
            "age": "age",
            "gender": "gender",
            "tenure": "tenure",
            "monthlycharges": "monthly_charges",
            "totalcharges": "total_charges",
            "contracttype": "contract_type",
            "internetservice": "internet_service",
            "techsupport": "tech_support",
        }.items():
            if source in normalized_columns:
                mapped_inputs[destination] = normalized_columns[source]
        for integer_field in ("age", "tenure"):
            if integer_field in mapped_inputs:
                mapped_inputs[integer_field] = int(mapped_inputs[integer_field])

        stored_input = dict(cleaned_values)
        if record_id_column:
            stored_input[record_id_column] = submitted_record_id

        db.session.add(Prediction(
            user_id=current_user.id,
            company_id=company.id,
            model_id=model_record.id,
            input_json=stored_input,
            **mapped_inputs,
            prediction=result["prediction"],
            probability=result["probability"],
            risk_level=result["risk_level"],
            risk_factors=result["risk_factors"],
            suggested_actions=result["suggested_actions"],
        ))
        db.session.commit()
    except (OSError, ValueError, KeyError, TypeError) as error:
        db.session.rollback()
        flash(str(error), "error")
        return render_company_prediction(company, form, mode=mode, manual_fields=manual_fields, status=400)

    result_view = {
        "prediction": result["prediction"],
        "probability": result["probability"],
        "risk": result["risk_level"],
        "factors": result["risk_factors"],
        "actions": result["suggested_actions"],
    }
    return render_company_prediction(
        company,
        result=result_view,
        values=cleaned_values,
        selected_record=submitted_record_id,
    )


def dashboard_data(form, result=None):
    query = Prediction.query.filter_by(user_id=current_user.id)
    active_company = get_active_company()
    if active_company:
        query = query.filter_by(company_id=active_company.id)
    total = query.count()
    churned = query.filter_by(prediction="Yes").count()
    non_churned = query.filter_by(prediction="No").count()
    high_risk = query.filter_by(risk_level="High Risk").count()
    risk_counts = [query.filter_by(risk_level=level).count() for level in ("Low Risk", "Medium Risk", "High Risk")]
    recent = query.order_by(Prediction.created_at.desc()).limit(5).all()
    return {
        "form": form,
        "result": result,
        "total": total,
        "churned": churned,
        "non_churned": non_churned,
        "high_risk": high_risk,
        "churn_chart": [churned, non_churned],
        "risk_chart": risk_counts,
        "recent": recent,
    }


def history_query():
    query = Prediction.query.join(Prediction.user)
    active_company = get_active_company()
    if active_company:
        query = query.filter(Prediction.company_id == active_company.id)
    if current_user.role != "admin":
        query = query.filter(Prediction.user_id == current_user.id)
    search = request.args.get("q", "").strip()
    prediction_filter = request.args.get("prediction", "")
    risk_filter = request.args.get("risk", "")
    from_value = request.args.get("from", "")
    to_value = request.args.get("to", "")
    if search:
        pattern = f"%{search}%"
        query = query.filter(or_(
            Prediction.gender.ilike(pattern),
            Prediction.contract_type.ilike(pattern),
            Prediction.internet_service.ilike(pattern),
            Prediction.tech_support.ilike(pattern),
            User.name.ilike(pattern),
            User.email.ilike(pattern),
        ))
    if prediction_filter in ("Yes", "No"):
        query = query.filter(Prediction.prediction == prediction_filter)
    if risk_filter in ("Low Risk", "Medium Risk", "High Risk"):
        query = query.filter(Prediction.risk_level == risk_filter)
    try:
        if from_value:
            query = query.filter(Prediction.created_at >= datetime.combine(date.fromisoformat(from_value), time.min))
        if to_value:
            query = query.filter(Prediction.created_at < datetime.combine(date.fromisoformat(to_value) + timedelta(days=1), time.min))
    except ValueError:
        flash("Date filters must use valid calendar dates.", "warning")
    sort = request.args.get("sort", "newest")
    sort_columns = {
        "newest": Prediction.created_at.desc(),
        "oldest": Prediction.created_at.asc(),
        "probability": Prediction.probability.desc(),
    }
    query = query.order_by(sort_columns.get(sort, sort_columns["newest"]))
    filters = {"q": search, "prediction": prediction_filter, "risk": risk_filter, "from": from_value, "to": to_value, "sort": sort}
    return query, filters


def get_accessible_companies(user=None):
    user = user or current_user
    if not user or not user.is_authenticated:
        return []
    if user.role == "admin":
        return Company.query.filter_by(is_active=True).order_by(Company.name.asc()).all()
    return (
        Company.query.join(UserCompanyAccess, UserCompanyAccess.company_id == Company.id)
        .filter(UserCompanyAccess.user_id == user.id, Company.is_active.is_(True))
        .order_by(Company.name.asc())
        .all()
    )


def get_active_company(user=None):
    user = user or current_user
    if not user or not user.is_authenticated:
        return None
    companies = get_accessible_companies(user)
    if not companies:
        session.pop("active_company_id", None)
        return None

    active_company_id = session.get("active_company_id")
    if active_company_id:
        try:
            company = db.session.get(Company, int(active_company_id))
        except (TypeError, ValueError):
            company = None
        if company and company in companies:
            return company

    first_company = companies[0]
    session["active_company_id"] = str(first_company.id)
    return first_company


def prediction_for_current_user(prediction_id):
    prediction = db.get_or_404(Prediction, prediction_id)
    if current_user.role != "admin" and prediction.user_id != current_user.id:
        abort(404)
    return prediction


@bp.route("/")
@login_required
def home():
    return redirect(url_for("main.dashboard"))


@bp.route("/predict", methods=["POST"])
@login_required
def predict():
    active_company = get_active_company()
    mode = request.args.get("mode") or request.form.get("mode", "id")
    if mode not in {"id", "manual"}:
        mode = "id"
    return predict_for_active_company(active_company, mode)


@bp.route("/company/select", methods=["POST"])
@login_required
def select_company():
    company_id = request.form.get("company_id")
    company = None
    if company_id not in (None, ""):
        try:
            company = db.session.get(Company, int(company_id))
        except (TypeError, ValueError):
            company = None

    if company is None:
        flash("Please choose a valid company.", "error")
        return redirect(request.referrer or url_for("main.dashboard"))

    if company not in get_accessible_companies():
        abort(403)

    session["active_company_id"] = str(company.id)
    flash(f"Active company set to {company.name}.", "success")
    if request.form.get("destination") == "prediction":
        return redirect(url_for("main.prediction"))
    return redirect(request.referrer or url_for("main.dashboard"))


@bp.route("/dashboard")
@login_required
def dashboard():
    active_company = get_active_company()
    company_cards = []
    for company in get_accessible_companies():
        active_datasets = [item for item in company.datasets if item.is_active]
        active_models = [item for item in company.trained_models if item.is_active]
        prediction_query = Prediction.query.filter_by(company_id=company.id)
        if current_user.role != "admin":
            prediction_query = prediction_query.filter_by(user_id=current_user.id)
        company_cards.append({
            "company": company,
            "dataset": max(active_datasets, key=lambda item: item.uploaded_at, default=None),
            "model": max(active_models, key=lambda item: item.version, default=None),
            "prediction_count": prediction_query.count(),
        })
    return render_template(
        "dashboard.html",
        **dashboard_data(prediction_form()),
        active_company=active_company,
        company_cards=company_cards,
    )


@bp.route("/prediction")
@login_required
def prediction():
    mode = request.args.get("mode", "id")
    if mode not in {"id", "manual"}:
        mode = "id"
    return render_company_prediction(get_active_company(), mode=mode)


@bp.route("/analytics")
@login_required
def analytics():
    missing = dataset.isna().sum().to_dict() if not dataset.empty else {}
    counts = dataset["Churn"].value_counts() if not dataset.empty else pd.Series(dtype=int)
    return render_template("analytics.html", records=len(dataset), features=len(dataset.columns) - 1 if not dataset.empty else 0, feature_names=FEATURES, missing=missing, churned=int(counts.get("yes", 0)), retained=int(counts.get("no", 0)))


@bp.route("/performance")
@login_required
def performance():
    if model_error or dataset.empty:
        flash(f"Model performance is unavailable: {model_error or 'dataset is empty'}", "error")
        return render_template("performance.html", results={}, matrix=[[0, 0], [0, 0]])
    results, matrix = evaluation_data()
    return render_template("performance.html", results=results, matrix=matrix)


@bp.route("/about")
@login_required
def about():
    return render_template("about.html")


@bp.route("/history")
@login_required
def history():
    query, filters = history_query()
    pagination = query.paginate(page=request.args.get("page", 1, type=int), per_page=10, error_out=False)
    return render_template(
        "history.html",
        pagination=pagination,
        predictions=pagination.items,
        filters=filters,
        is_admin=current_user.role == "admin",
        email_form=EmailHistoryForm(),
    )


def get_history_export_data(query, is_admin=False):
    columns = [
        "Created At", "Age", "Gender", "Tenure", "Monthly Charges", "Total Charges",
        "Contract Type", "Internet Service", "Tech Support", "Prediction",
        "Probability", "Risk Level"
    ]
    if is_admin:
        columns.append("Predicted By")
    rows = []
    for prediction in query.all():
        row = {
            "Created At": prediction.created_at.isoformat() if prediction.created_at else "",
            "Age": prediction.age if prediction.age is not None else "",
            "Gender": prediction.gender or "",
            "Tenure": prediction.tenure if prediction.tenure is not None else "",
            "Monthly Charges": prediction.monthly_charges if prediction.monthly_charges is not None else "",
            "Total Charges": prediction.total_charges if prediction.total_charges is not None else "",
            "Contract Type": prediction.contract_type or "",
            "Internet Service": prediction.internet_service or "",
            "Tech Support": prediction.tech_support or "",
            "Prediction": prediction.prediction or "",
            "Probability": round(prediction.probability * 100, 2) if prediction.probability is not None else "",
            "Risk Level": prediction.risk_level or "",
        }
        if is_admin:
            row["Predicted By"] = prediction.user.name if prediction.user else ""
        rows.append(row)
    return columns, rows


def history_csv(query):
    columns, rows = get_history_export_data(query, is_admin=current_user.role == "admin")
    return build_csv(columns, rows).decode("utf-8")


@bp.route("/history/export.csv")
@login_required
def export_history():
    query, _filters = history_query()
    columns, rows = get_history_export_data(query, is_admin=current_user.role == "admin")
    csv_content = build_csv(columns, rows)
    response = make_response(csv_content)
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=churn-predictions.csv"
    return response


@bp.route("/history/email.csv", methods=["POST"])
@login_required
def email_history():
    form = EmailHistoryForm()
    if not form.validate_on_submit():
        for errors in form.errors.values():
            for error in errors:
                flash(error, "error")
        return redirect(url_for("main.history", **history_query()[1]))

    query, filters = history_query()
    if not query.first():
        flash("There are no predictions matching these filters to email.", "warning")
        return redirect(url_for("main.history", **filters))

    if not check_outgoing_email_rate_limit(current_user.id):
        flash("You have reached the limit of 10 outgoing emails per hour. Please try again later.", "warning")
        return redirect(url_for("main.history", **filters))

    file_format = form.file_format.data or "csv"
    columns, rows = get_history_export_data(query, is_admin=current_user.role == "admin")
    csv_bytes = build_csv(columns, rows) if file_format in ("csv", "both") else None
    xlsx_bytes = build_xlsx(columns, rows, sheet_name="Prediction History") if file_format in ("xlsx", "both") else None

    try:
        send_export_email(
            recipient_email=current_user.email,
            subject="Your churn prediction history",
            message_text=form.message.data,
            csv_bytes=csv_bytes,
            csv_filename="churn-predictions.csv",
            xlsx_bytes=xlsx_bytes,
            xlsx_filename="churn-predictions.xlsx",
        )
        record_outgoing_email_sent(current_user.id)
    except EmailConfigurationError as error:
        current_app.logger.error("Could not email prediction history to account %s: %s", current_user.id, error)
        flash(f"Email delivery is not configured: {error}", "error")
        return redirect(url_for("main.history", **filters))
    except (ValueError, OSError, smtplib.SMTPException) as error:
        current_app.logger.error("Could not email prediction history to account %s: %s", current_user.id, error)
        flash(f"The email could not be sent: {error}", "error")
        return redirect(url_for("main.history", **filters))

    format_desc = "Excel export" if file_format == "xlsx" else "CSV and Excel export" if file_format == "both" else "CSV"
    flash(f"Prediction {format_desc} emailed to your account ({current_user.email}).", "success")
    return redirect(url_for("main.history", **filters))


@bp.route("/history/<int:prediction_id>/delete", methods=["POST"])
@login_required
def delete_prediction(prediction_id):
    prediction = prediction_for_current_user(prediction_id)
    db.session.delete(prediction)
    db.session.commit()
    flash("Prediction deleted.", "success")
    return redirect(url_for("main.history"))


@bp.route("/prediction/<int:prediction_id>/report.pdf")
@login_required
def prediction_report(prediction_id):
    prediction = prediction_for_current_user(prediction_id)
    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Customer Churn Prediction Report", styles["Title"]), Spacer(1, 16)]
    details = [
        ("Prediction date", prediction.created_at.strftime("%Y-%m-%d %H:%M UTC")),
        ("Prediction", prediction.prediction),
        ("Churn probability", f"{prediction.probability * 100:.2f}%"),
        ("Risk level", prediction.risk_level),
    ]
    stored_inputs = prediction.input_json or {}
    if stored_inputs:
        details.extend((str(label), str(value) if value is not None else "—") for label, value in stored_inputs.items())
    else:
        details.extend([
            ("Age", str(prediction.age) if prediction.age is not None else "—"),
            ("Gender", prediction.gender or "—"),
            ("Tenure", f"{prediction.tenure} months" if prediction.tenure is not None else "—"),
            ("Monthly charges", f"${prediction.monthly_charges:.2f}" if prediction.monthly_charges is not None else "—"),
            ("Total charges", f"${prediction.total_charges:.2f}" if prediction.total_charges is not None else "—"),
            ("Contract type", prediction.contract_type or "—"),
            ("Internet service", prediction.internet_service or "—"),
            ("Tech support", prediction.tech_support or "—"),
        ])
    details.extend([
        ("Risk factors", "; ".join(prediction.risk_factors) if prediction.risk_factors else "—"),
        ("Suggested actions", "; ".join(prediction.suggested_actions) if prediction.suggested_actions else "—"),
    ])
    table = Table([[Paragraph(html.escape(str(label)), styles["BodyText"]), Paragraph(html.escape(str(value)), styles["BodyText"])] for label, value in details], colWidths=[1.8 * inch, 5.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f1ed")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4e1db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    document.build(story)
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=prediction-{prediction.id}.pdf"
    return response


@bp.route("/bulk", methods=["GET", "POST"])
@login_required
def bulk_predict():
    form = BulkUploadForm()
    company = get_active_company()
    schema = None
    schema_error = None
    model_available = False
    if company:
        try:
            dataset_record = get_active_dataset(company)
            if dataset_record is None:
                raise ValueError(f"No active dataset is available for {company.name}.")
            schema = get_dataset_schema(dataset_record)
            model_record = get_active_model(company)
            model_available = bool(model_record and model_record.dataset_id == dataset_record.id)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            schema_error = str(error)
    else:
        schema_error = "Choose a company workspace from the dashboard before uploading a CSV."

    required = schema["feature_columns"] if schema else []
    column_details = [
        {
            "name": column,
            "kind": schema["types"].get(column, "numeric"),
            "categories": schema["allowed_categories"].get(column, []),
            "limits": schema["numeric_ranges"].get(column),
        }
        for column in required
    ] if schema else []
    if form.validate_on_submit():
        upload = form.file.data
        try:
            if schema_error:
                raise ValueError(schema_error)
            if not upload.filename.lower().endswith(".csv"):
                raise ValueError("Upload a .csv file.")
            frame = pd.read_csv(upload, nrows=501)
            missing = [column for column in required if column not in frame.columns]
            unexpected = [column for column in frame.columns if column not in required]
            if missing or unexpected:
                details = []
                if missing:
                    details.append("missing columns: " + ", ".join(missing))
                if unexpected:
                    details.append("unexpected columns: " + ", ".join(unexpected))
                raise ValueError("CSV columns do not match this company's dataset (" + "; ".join(details) + ").")
            if frame.empty or len(frame) > 500:
                raise ValueError("The CSV must contain between 1 and 500 customer rows.")

            valid_rows = []
            row_errors = []
            for row_number, row in frame.iterrows():
                row_index = row_number + 2
                try:
                    valid_rows.append(validate_input_against_schema(company, row[required].to_dict(), schema))
                except ValueError as error:
                    row_errors.append(f"Row {row_index}: data is wrong: {error}")

            if row_errors:
                for message in row_errors[:10]:
                    flash(message, "error")
                if len(row_errors) > 10:
                    flash(f"{len(row_errors) - 10} additional rows contain invalid data.", "error")
                return render_template(
                    "bulk.html", form=form, active_company=company,
                    required_columns=required, schema_error=schema_error,
                    column_details=column_details, model_available=model_available,
                ), 400

            if not model_available:
                raise ValueError(f"No active model is trained on the current dataset for {company.name}.")

            scored = predict_many(company, pd.DataFrame(valid_rows, columns=required))
            model_record = get_active_model(company)
            db.session.commit()

            export_columns = required + ["Prediction", "Probability", "Risk Level"]
            export_rows = []
            for values, (_, row) in zip(valid_rows, scored.iterrows()):
                probability = float(row["Probability"])
                normalized_columns = {"".join(character for character in key.lower() if character.isalnum()): value for key, value in values.items()}
                mapped_inputs = {}
                for source, destination in {
                    "age": "age", "gender": "gender", "tenure": "tenure",
                    "monthlycharges": "monthly_charges", "totalcharges": "total_charges",
                    "contracttype": "contract_type", "internetservice": "internet_service",
                    "techsupport": "tech_support",
                }.items():
                    if source in normalized_columns:
                        mapped_inputs[destination] = normalized_columns[source]
                for integer_field in ("age", "tenure"):
                    if integer_field in mapped_inputs:
                        mapped_inputs[integer_field] = int(mapped_inputs[integer_field])
                db.session.add(Prediction(
                    user_id=current_user.id,
                    company_id=company.id,
                    model_id=model_record.id,
                    input_json=values,
                    **mapped_inputs,
                    prediction=row["Prediction"],
                    probability=probability,
                    risk_level=row["RiskLevel"],
                    risk_factors=[],
                    suggested_actions=[],
                ))
                export_rows.append([values[column] for column in required] + [
                    row["Prediction"], round(probability * 100, 2), row["RiskLevel"],
                ])
            db.session.commit()

            if form.email_results.data:
                if not check_outgoing_email_rate_limit(current_user.id):
                    flash("You have reached the limit of 10 outgoing emails per hour. Predictions were saved, but email could not be sent.", "warning")
                    return redirect(url_for("main.bulk_predict"))

                file_format = form.file_format.data or "csv"
                csv_bytes = build_csv(export_columns, export_rows) if file_format in ("csv", "both") else None
                xlsx_bytes = build_xlsx(export_columns, export_rows, sheet_name="Bulk Predictions") if file_format in ("xlsx", "both") else None

                try:
                    send_export_email(
                        recipient_email=current_user.email,
                        subject="Your bulk churn predictions",
                        message_text=form.message.data,
                        csv_bytes=csv_bytes,
                        csv_filename="bulk-churn-predictions.csv",
                        xlsx_bytes=xlsx_bytes,
                        xlsx_filename="bulk-churn-predictions.xlsx",
                    )
                    record_outgoing_email_sent(current_user.id)
                    flash(f"Bulk predictions saved and emailed to your account ({current_user.email}).", "success")
                except EmailConfigurationError as error:
                    current_app.logger.error("Could not email bulk predictions to account %s: %s", current_user.id, error)
                    flash(f"Predictions saved, but email delivery is not configured: {error}", "error")
                except (ValueError, OSError, smtplib.SMTPException) as error:
                    current_app.logger.error("Could not email bulk predictions to account %s: %s", current_user.id, error)
                    flash(f"Predictions saved, but the email could not be sent: {error}", "error")

                return redirect(url_for("main.bulk_predict"))

            csv_content = build_csv(export_columns, export_rows)
            response = make_response(csv_content)
            response.headers["Content-Type"] = "text/csv; charset=utf-8"
            response.headers["Content-Disposition"] = "attachment; filename=bulk-churn-predictions.csv"
            return response
        except (OSError, ValueError, KeyError, TypeError) as error:
            db.session.rollback()
            flash(f"Data is wrong: {error}", "error")
    elif request.method == "POST":
        for errors in form.errors.values():
            for error in errors:
                flash(error, "error")
    return render_template(
        "bulk.html", form=form, active_company=company,
        required_columns=required, schema_error=schema_error,
        column_details=column_details, model_available=model_available,
    )


@bp.route("/bulk/template.csv")
@login_required
def bulk_template():
    company = get_active_company()
    if company is None:
        abort(400, description="Choose a company workspace before downloading a CSV template.")
    try:
        dataset_record = get_active_dataset(company)
        if dataset_record is None:
            raise ValueError(f"No active dataset is available for {company.name}.")
        required = get_dataset_schema(dataset_record)["feature_columns"]
        sample = pd.read_csv(dataset_record.data_path, usecols=required).dropna(subset=required).head(500)
        if sample.empty:
            abort(422, description=f"The active dataset for {company.name} has no complete feature rows to download.")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        abort(400, description=str(error))
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(required)
    writer.writerows(sample.itertuples(index=False, name=None))
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f"attachment; filename={company.slug}-prediction-template.csv"
    return response


@bp.route("/profile")
@login_required
def profile():
    profile_form = ProfileForm(obj=current_user)
    password_form = ChangePasswordForm()
    login_activities = LoginLog.query.filter_by(user_id=current_user.id).order_by(LoginLog.timestamp.desc()).limit(5).all()
    prediction_count = Prediction.query.filter_by(user_id=current_user.id).count()
    return render_template("profile.html", profile_form=profile_form, password_form=password_form, login_activities=login_activities, prediction_count=prediction_count)


@bp.route("/profile/edit", methods=["POST"])
@login_required
def edit_profile():
    form = ProfileForm()
    if form.validate_on_submit():
        current_user.name = form.name.data
        db.session.commit()
        flash("Profile updated.", "success")
    else:
        for errors in form.errors.values():
            for error in errors:
                flash(error, "error")
    return redirect(url_for("main.profile"))


@bp.route("/profile/password", methods=["POST"])
@login_required
def change_password():
    from werkzeug.security import check_password_hash, generate_password_hash

    form = ChangePasswordForm()
    if form.validate_on_submit():
        if not check_password_hash(current_user.password_hash, form.current_password.data):
            flash("Current password is incorrect.", "error")
        else:
            current_user.password_hash = generate_password_hash(form.password.data)
            db.session.commit()
            flash("Password updated.", "success")
    else:
        for errors in form.errors.values():
            for error in errors:
                flash(error, "error")
    return redirect(url_for("main.profile"))
