import io
from datetime import timedelta

import pytest
from werkzeug.security import generate_password_hash

from churn_app.blueprints.main import routes as main_routes
from churn_app.extensions import db
from churn_app.models import (
    AdminActivityLog,
    Company,
    CompanyDataset,
    EmailOTP,
    LoginLog,
    Prediction,
    TrainedModel,
    User,
    UserCompanyAccess,
    utc_now,
)
from churn_app.services import build_data_quality_report, infer_schema, train_company_model


def add_user(app, *, name="Test User", email="user@example.com", role="user", email_verified=True):
    with app.app_context():
        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash("StrongPass1!"),
            role=role,
            email_verified=email_verified,
        )
        db.session.add(user)
        db.session.commit()
        return user.id


def authenticate(client, user_id):
    with client.session_transaction() as session:
        session["_user_id"] = str(user_id)
        session["_fresh"] = True


def test_register_uses_admin_code_and_assigns_users_by_default(app, client):
    weak = client.post("/register", data={
        "name": "First Admin",
        "email": "FIRST@example.com",
        "password": "weakpass",
        "confirm_password": "weakpass",
    })
    assert weak.status_code == 200
    assert b"uppercase" in weak.data

    response = client.post("/register", data={
        "name": "First Admin",
        "email": "FIRST@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
        "admin_code": "wrong-code",
    })
    assert response.status_code == 200
    assert b"admin registration code is invalid" in response.data

    response = client.post("/register", data={
        "name": "First Admin",
        "email": "FIRST@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
        "admin_code": "test-admin-code",
    })
    assert response.status_code == 302
    with app.app_context():
        first = User.query.one()
        assert first.email == "first@example.com"
        assert first.role == "admin"

    client.post("/register", data={
        "name": "Second User",
        "email": "second@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
    })
    with app.app_context():
        assert User.query.filter_by(email="second@example.com").one().role == "user"


def test_registration_and_email_verification_flow(app, client, monkeypatch):
    import re

    sent_messages = []
    monkeypatch.setattr("churn_app.services.email_service.send_email", lambda msg: sent_messages.append(msg))

    # 1. Register new admin account
    registration = client.post("/register", data={
        "name": "First Admin",
        "email": "admin@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
        "admin_code": "test-admin-code",
    })

    assert registration.status_code == 302
    assert "/verify-email" in registration.headers["Location"]
    assert len(sent_messages) == 1
    assert sent_messages[0]["To"] == "admin@example.com"
    email_content = sent_messages[0].get_body(preferencelist=("plain",)).get_content()
    match = re.search(r"\b(\d{6})\b", email_content)
    assert match is not None
    code = match.group(1)

    # User in DB is not verified yet
    with app.app_context():
        user = User.query.filter_by(email="admin@example.com").one()
        assert user.email_verified is False
        assert user.role == "admin"

    # 2. Login before verification must be blocked and redirect to /verify-email
    blocked_login = client.post("/login", data={
        "email": "admin@example.com",
        "password": "StrongPass1!",
    }, follow_redirects=False)
    assert blocked_login.status_code == 302
    assert "/verify-email" in blocked_login.headers["Location"]

    # 3. Verify with code
    verify_response = client.post(
        "/verify-email?email=admin@example.com",
        data={"code": code},
        follow_redirects=True,
    )
    assert verify_response.status_code == 200
    assert b"Your email has been verified" in verify_response.data

    with app.app_context():
        user = User.query.filter_by(email="admin@example.com").one()
        assert user.email_verified is True
        assert user.email_verified_at is not None

    # 4. Login after verification now works
    login_response = client.post("/login", data={
        "email": "admin@example.com",
        "password": "StrongPass1!",
    })
    assert login_response.status_code == 302
    assert login_response.headers["Location"].endswith("/admin/dashboard")


def test_verify_email_wrong_code_expired_code_and_lockout(app, client, monkeypatch):
    sent_messages = []
    monkeypatch.setattr("churn_app.services.email_service.send_email", lambda msg: sent_messages.append(msg))

    client.post("/register", data={
        "name": "Lock User",
        "email": "lock@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
    })

    # Wrong code 4 times
    for attempt in range(1, 5):
        resp = client.post("/verify-email?email=lock@example.com", data={"code": "000000"})
        assert resp.status_code == 200
        assert f"{5 - attempt} attempt".encode() in resp.data

    # 5th wrong code locks / invalidates the code
    resp5 = client.post("/verify-email?email=lock@example.com", data={"code": "000000"})
    assert resp5.status_code == 200
    assert b"invalidated" in resp5.data

    # 6th attempt shows locked/invalidated message
    resp6 = client.post("/verify-email?email=lock@example.com", data={"code": "000000"})
    assert resp6.status_code == 200
    assert b"locked" in resp6.data or b"invalidated" in resp6.data

    # Expired code test
    with app.app_context():
        from datetime import timedelta
        user = User.query.filter_by(email="lock@example.com").one()
        otp = EmailOTP.query.filter_by(user_id=user.id).order_by(EmailOTP.created_at.desc()).first()
        otp.expires_at = utc_now() - timedelta(minutes=1)
        otp.attempts = 0
        db.session.commit()

    expired_resp = client.post("/verify-email?email=lock@example.com", data={"code": "123456"})
    assert expired_resp.status_code == 200
    assert b"expired" in expired_resp.data


def test_resend_code_cooldown_and_rate_limit(app, client, monkeypatch):
    sent_messages = []
    monkeypatch.setattr("churn_app.services.email_service.send_email", lambda msg: sent_messages.append(msg))

    client.post("/register", data={
        "name": "Resend User",
        "email": "resend@example.com",
        "password": "StrongPass1!",
        "confirm_password": "StrongPass1!",
    })
    assert len(sent_messages) == 1

    # Immediate resend triggers 60s cooldown
    cooldown_resp = client.post("/resend-code", data={"email": "resend@example.com"}, follow_redirects=True)
    assert b"Please wait" in cooldown_resp.data
    assert len(sent_messages) == 1

    # Fast forward OTP created_at in database
    with app.app_context():
        from datetime import timedelta
        user = User.query.filter_by(email="resend@example.com").one()
        otp = EmailOTP.query.filter_by(user_id=user.id).order_by(EmailOTP.created_at.desc()).first()
        otp.created_at = utc_now() - timedelta(seconds=65)
        db.session.commit()

    resend_ok = client.post("/resend-code", data={"email": "resend@example.com"}, follow_redirects=True)
    assert b"new verification code has been sent" in resend_ok.data
    assert len(sent_messages) == 2

    # Simulate reaching 5 sends in last hour
    with app.app_context():
        from datetime import timedelta
        user = User.query.filter_by(email="resend@example.com").one()
        for i in range(4):
            db.session.add(EmailOTP(
                user_id=user.id,
                code_hash="fake",
                purpose="verify_email",
                expires_at=utc_now() + timedelta(minutes=10),
                created_at=utc_now() - timedelta(minutes=i + 5),
            ))
        db.session.commit()

    limit_resp = client.post("/resend-code", data={"email": "resend@example.com"}, follow_redirects=True)
    assert b"maximum of 5 verification requests per hour" in limit_resp.data


def test_login_generic_failure_lockout_and_audit(app, client):
    user_id = add_user(app)
    for _attempt in range(5):
        response = client.post("/login", data={"email": "user@example.com", "password": "wrong"})
        assert response.status_code == 200
        assert b"Invalid email or password" in response.data
    with app.app_context():
        user = db.session.get(User, user_id)
        assert user.failed_attempts == 5
        assert user.locked_until is not None
        assert LoginLog.query.filter_by(success=False).count() == 5

    response = client.post("/login", data={"email": "missing@example.com", "password": "wrong"})
    assert response.status_code == 200
    assert b"Invalid email or password" in response.data
    with app.app_context():
        assert LoginLog.query.count() == 6
        assert LoginLog.query.filter_by(email_entered="missing@example.com", user_id=None).count() == 1


def test_existing_account_can_sign_in_without_reverification(app, client):
    add_user(app)

    response = client.post("/login", data={
        "email": "user@example.com",
        "password": "StrongPass1!",
    })

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_admin_guard_and_self_protection(app, client):
    admin_id = add_user(app, name="Admin User", email="admin@example.com", role="admin")
    user_id = add_user(app, name="Regular User")
    authenticate(client, user_id)
    assert client.get("/admin/dashboard").status_code == 403

    authenticate(client, admin_id)
    response = client.post(f"/admin/users/{admin_id}/role", data={"role": "user"})
    assert response.status_code == 302
    response = client.post(f"/admin/users/{admin_id}/delete")
    assert response.status_code == 302
    with app.app_context():
        assert db.session.get(User, admin_id).role == "admin"
        assert db.session.get(User, admin_id) is not None
        assert AdminActivityLog.query.filter(AdminActivityLog.action.like("blocked self-%")).count() == 2


@pytest.mark.parametrize(
    ("probability", "risk"),
    [(0.2999, "Low Risk"), (0.3, "Medium Risk"), (0.5999, "Medium Risk"), (0.6, "High Risk")],
)
def test_prediction_validation_persistence_and_risk_thresholds(app, client, monkeypatch, tmp_path, probability, risk):
    import pandas as pd

    admin_id = add_user(app, name="Prediction Admin", email="prediction-admin@example.com", role="admin")
    user_id = add_user(app)
    data_path = tmp_path / "company.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "Age": [20, 35, 50],
        "Gender": ["Female", "Male", "Female"],
        "Churn": ["No", "Yes", "No"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="SchemaCo", slug="schemaco", industry="Telecom")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="company.csv",
            row_count=3,
            schema_json={
                "columns": ["CustomerID", "Age", "Gender", "Churn"],
                "target_column": "Churn",
                "positive_label": "Yes",
                "id_columns": ["CustomerID"],
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        model_record = TrainedModel(
            company_id=company.id,
            dataset_id=dataset.id,
            algorithm="StubModel",
            model_path="stub.joblib",
            is_active=True,
        )
        db.session.add_all([
            model_record,
        ])
        db.session.commit()
        company_id = company.id
        model_id = model_record.id

    authenticate(client, admin_id)
    access_response = client.post(
        f"/admin/users/{user_id}/access",
        data={"company_ids": str(company_id)},
    )
    assert access_response.status_code == 302

    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)
    monkeypatch.setattr(main_routes, "predict_one", lambda _company, _values, _schema: {
        "prediction": "Yes",
        "probability": probability,
        "risk_level": risk,
        "risk_factors": ["Test risk factor"],
        "suggested_actions": ["Test retention action"],
    })

    prediction_page = client.get("/prediction")
    assert prediction_page.status_code == 200
    assert b'name="record_id"' in prediction_page.data
    assert b'name="feature_0"' not in prediction_page.data
    assert b"C001" in prediction_page.data

    invalid = client.post("/predict", data={"record_id": "C999"})
    assert invalid.status_code == 400
    assert b"Data is wrong" in invalid.data

    arbitrary_values = client.post("/predict", data={"record_id": "C001", "feature_0": "999"})
    assert arbitrary_values.status_code == 400
    assert b"submitted fields do not match this prediction mode" in arbitrary_values.data

    response = client.post("/predict", data={"record_id": "C002"})
    assert response.status_code == 200
    assert risk.encode() in response.data
    with app.app_context():
        saved = Prediction.query.one()
        assert saved.user_id == user_id
        assert saved.company_id == company_id
        assert saved.model_id == model_id
        assert saved.input_json == {"Age": 35, "Gender": "Male", "CustomerID": "C002"}
        assert saved.age == 35
        assert saved.probability == pytest.approx(probability)
        assert saved.risk_level == risk

    manual_page = client.get("/prediction?mode=manual")
    assert manual_page.status_code == 200
    assert b'name="feature_0"' in manual_page.data
    assert b'name="feature_1"' in manual_page.data

    unseen_combination = client.post("/predict", data={
        "mode": "manual", "feature_0": "35", "feature_1": "Female",
    })
    assert unseen_combination.status_code == 400
    assert b"must match one complete row" in unseen_combination.data

    manual_response = client.post("/predict", data={
        "mode": "manual", "feature_0": "35", "feature_1": "Male",
    })
    assert manual_response.status_code == 200
    with app.app_context():
        saved = Prediction.query.order_by(Prediction.id.desc()).first()
        assert saved.input_json == {"Age": 35, "Gender": "Male", "CustomerID": "C002"}


def test_csrf_rejects_unprotected_form_post():
    from churn_app import create_app

    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "pytest-secret-key",
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "WTF_CSRF_ENABLED": True,
    })
    response = app.test_client().post("/register", data={"name": "No Token"})
    assert response.status_code == 400


def test_dashboard_links_to_separate_feature_pages(app, client, tmp_path):
    import pandas as pd

    user_id = add_user(app)
    data_path = tmp_path / "dashboard-company.csv"
    pd.DataFrame({"Age": [20, 40], "Plan": ["Basic", "Plus"], "Churn": ["No", "Yes"]}).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="DashboardCo", slug="dashboardco", industry="SaaS")
        db.session.add(company)
        db.session.flush()
        db.session.add_all([
            CompanyDataset(
                company_id=company.id,
                filename="dashboard-company.csv",
                row_count=2,
                schema_json={"columns": ["Age", "Plan", "Churn"], "target_column": "Churn"},
                data_path=str(data_path),
            ),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
    authenticate(client, user_id)

    dashboard = client.get("/dashboard")
    prediction_page = client.get("/prediction")

    assert dashboard.status_code == 200
    assert b"Prediction history" in dashboard.data
    assert b"Bulk CSV" in dashboard.data
    assert b"action=\"/predict\"" not in dashboard.data
    assert prediction_page.status_code == 200
    assert b"action=\"/predict?mode=id\"" in prediction_page.data


def test_prediction_requires_a_company_workspace(app, client):
    user_id = add_user(app)
    authenticate(client, user_id)

    response = client.post("/predict", data={})

    assert response.status_code == 400
    assert b"Choose a company workspace" in response.data


def test_history_renders_company_prediction_without_legacy_telco_fields(app, client):
    user_id = add_user(app)
    with app.app_context():
        company = Company(name="HistoryBank", slug="historybank", industry="Banking")
        db.session.add(company)
        db.session.flush()
        prediction = Prediction(
            user_id=user_id,
            company_id=company.id,
            prediction="No",
            probability=0.12,
            risk_level="Low Risk",
            risk_factors=[],
            suggested_actions=[],
            input_json={"CustomerID": "BK000001", "CreditScore": 650, "Geography": "France"},
        )
        db.session.add_all([prediction, UserCompanyAccess(user_id=user_id, company_id=company.id)])
        db.session.commit()
        company_id = company.id
    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)

    response = client.get("/history")

    assert response.status_code == 200
    assert b"BK000001" in response.data
    assert b"CreditScore" in response.data
    with app.app_context():
        prediction_id = Prediction.query.one().id
    report = client.get(f"/prediction/{prediction_id}/report.pdf")
    assert report.status_code == 200
    assert report.mimetype == "application/pdf"


def test_email_history_sends_filtered_csv_to_signed_in_account(app, client, monkeypatch):
    user_id = add_user(app)
    with app.app_context():
        db.session.add(Prediction(
            user_id=user_id,
            prediction="Yes",
            probability=0.82,
            risk_level="High Risk",
            risk_factors=[],
            suggested_actions=[],
        ))
        db.session.commit()
    authenticate(client, user_id)
    app.config.update({
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": 587,
        "SMTP_USERNAME": "mailer@example.com",
        "SMTP_PASSWORD": "smtp-password",
        "SMTP_SENDER": "mailer@example.com",
        "SMTP_USE_TLS": True,
    })
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent["connection"] = (host, port, timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def starttls(self):
            sent["tls"] = True

        def login(self, username, password):
            sent["login"] = (username, password)

        def send_message(self, message):
            sent["message"] = message

    monkeypatch.setattr(main_routes.smtplib, "SMTP", FakeSMTP)

    response = client.post(
        "/history/email.csv?prediction=Yes",
        data={"message": "Please review this high-risk customer."},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Prediction CSV emailed to your account" in response.data
    assert sent["connection"] == ("smtp.example.com", 587, 15)
    assert sent["tls"] is True
    assert sent["login"] == ("mailer@example.com", "smtp-password")
    message = sent["message"]
    assert message["To"] == "user@example.com"
    assert message.get_body(preferencelist=("plain",)).get_content().strip() == "Please review this high-risk customer."
    assert "Please review this high-risk customer." in message.get_body(preferencelist=("html",)).get_content()
    attachments = list(message.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "churn-predictions.csv"
    assert b"Yes" in attachments[0].get_payload(decode=True)


def test_email_history_formats_csv_xlsx_and_both(app, client, monkeypatch):
    import io
    import openpyxl
    from churn_app.services.email_service import reset_outgoing_email_rate_limits

    reset_outgoing_email_rate_limits()
    user_id = add_user(app, email="export-test@example.com")
    with app.app_context():
        db.session.add(Prediction(
            user_id=user_id,
            prediction="Yes",
            probability=0.82,
            risk_level="High Risk",
            gender="Female",
            age=45,
            tenure=24,
            monthly_charges=75.5,
            total_charges=1800.0,
            contract_type="Month-to-month",
            internet_service="Fiber optic",
            tech_support="No",
            risk_factors=[],
            suggested_actions=[],
        ))
        db.session.commit()

    authenticate(client, user_id)
    app.config.update({
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": 587,
        "SMTP_USERNAME": "mailer@example.com",
        "SMTP_PASSWORD": "smtp-password",
        "SMTP_SENDER": "mailer@example.com",
        "SMTP_USE_TLS": True,
    })

    sent_messages = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def starttls(self):
            pass

        def login(self, username, password):
            pass

        def send_message(self, message):
            sent_messages.append(message)

    monkeypatch.setattr(main_routes.smtplib, "SMTP", FakeSMTP)

    # 1. Test CSV format
    resp_csv = client.post("/history/email.csv", data={"file_format": "csv", "message": "Here is CSV."}, follow_redirects=True)
    assert resp_csv.status_code == 200
    assert b"Prediction CSV emailed to your account" in resp_csv.data
    msg = sent_messages[-1]
    assert msg["To"] == "export-test@example.com"
    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "churn-predictions.csv"
    assert b"Yes" in attachments[0].get_payload(decode=True)

    # 2. Test XLSX format
    resp_xlsx = client.post("/history/email.csv", data={"file_format": "xlsx", "message": "Here is Excel."}, follow_redirects=True)
    assert resp_xlsx.status_code == 200
    assert b"Prediction Excel export emailed to your account" in resp_xlsx.data
    msg = sent_messages[-1]
    assert msg["To"] == "export-test@example.com"
    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "churn-predictions.xlsx"
    assert attachments[0].get_content_type() == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    wb = openpyxl.load_workbook(io.BytesIO(attachments[0].get_payload(decode=True)))
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    assert "Prediction" in headers
    assert "Probability" in headers
    assert "Risk Level" in headers

    # 3. Test Both format
    resp_both = client.post("/history/email.csv", data={"file_format": "both", "message": "Here is both."}, follow_redirects=True)
    assert resp_both.status_code == 200
    assert b"Prediction CSV and Excel export emailed to your account" in resp_both.data
    msg = sent_messages[-1]
    assert msg["To"] == "export-test@example.com"
    attachments = list(msg.iter_attachments())
    assert len(attachments) == 2
    filenames = {att.get_filename() for att in attachments}
    assert filenames == {"churn-predictions.csv", "churn-predictions.xlsx"}


def test_formula_injection_cells_escaped():
    import io
    import openpyxl
    from churn_app.services.export_service import build_csv, build_xlsx, sanitize_cell

    assert sanitize_cell("=1+1") == "'=1+1"
    assert sanitize_cell("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"
    assert sanitize_cell("-2+3") == "'-2+3"
    assert sanitize_cell("@SUM(1,2)") == "'@SUM(1,2)"
    assert sanitize_cell("Normal Text") == "Normal Text"
    assert sanitize_cell(42) == 42
    assert sanitize_cell(-3.14) == -3.14

    columns = ["Name", "Payload"]
    rows = [{"Name": "Test", "Payload": "=cmd|' /C calc'!A0"}]

    csv_data = build_csv(columns, rows)
    assert b"'=cmd|' /C calc'!A0" in csv_data

    xlsx_data = build_xlsx(columns, rows)
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_data))
    assert wb.active["B2"].value == "'=cmd|' /C calc'!A0"


def test_outgoing_email_rate_limit(app, client, monkeypatch):
    from churn_app.services.email_service import reset_outgoing_email_rate_limits

    reset_outgoing_email_rate_limits()
    user_id = add_user(app, email="ratelimit@example.com")
    with app.app_context():
        db.session.add(Prediction(
            user_id=user_id,
            prediction="No",
            probability=0.2,
            risk_level="Low Risk",
            risk_factors=[],
            suggested_actions=[],
        ))
        db.session.commit()
    authenticate(client, user_id)
    app.config.update({
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": 587,
        "SMTP_USERNAME": "mailer@example.com",
        "SMTP_PASSWORD": "smtp-password",
        "SMTP_SENDER": "mailer@example.com",
        "SMTP_USE_TLS": True,
    })

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            pass

        def login(self, *args):
            pass

        def send_message(self, *args):
            pass

    monkeypatch.setattr(main_routes.smtplib, "SMTP", FakeSMTP)

    # Send 10 emails (limit is 10/hour)
    for _ in range(10):
        res = client.post("/history/email.csv", data={"file_format": "csv"}, follow_redirects=True)
        assert b"Prediction CSV emailed to your account" in res.data

    # 11th email must be rate limited
    blocked = client.post("/history/email.csv", data={"file_format": "csv"}, follow_redirects=True)
    assert b"limit of 10 outgoing emails per hour" in blocked.data


def test_email_history_requires_login_and_reports_missing_smtp_config(app, client):
    assert client.post("/history/email.csv").status_code == 302

    user_id = add_user(app)
    with app.app_context():
        db.session.add(Prediction(
            user_id=user_id,
            prediction="No",
            probability=0.1,
            risk_level="Low Risk",
            risk_factors=[],
            suggested_actions=[],
        ))
        db.session.commit()
    authenticate(client, user_id)
    response = client.post("/history/email.csv", data={"message": "Please review."}, follow_redirects=True)

    assert response.status_code == 200
    assert b"Email delivery is not configured" in response.data


def test_prediction_handles_missing_model_probability(app, client, monkeypatch, tmp_path):
    import pandas as pd

    user_id = add_user(app)
    data_path = tmp_path / "probability-company.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002"],
        "Age": [30, 40],
        "Gender": ["Female", "Male"],
        "Churn": ["No", "Yes"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="ProbabilityCo", slug="probabilityco", industry="Retail")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="probability-company.csv",
            row_count=2,
            schema_json={"columns": ["CustomerID", "Age", "Gender", "Churn"], "target_column": "Churn"},
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        db.session.add_all([
            TrainedModel(company_id=company.id, dataset_id=dataset.id, model_path="unused.joblib", is_active=True),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
        company_id = company.id
    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)
    monkeypatch.setattr(main_routes, "predict_one", lambda *_args: {
        "prediction": "No",
        "probability": None,
        "risk_level": "Low Risk",
        "risk_factors": [],
        "suggested_actions": [],
    })

    response = client.post("/predict", data={"record_id": "C001"})

    assert response.status_code == 400
    assert b"did not return a valid probability" in response.data
    with app.app_context():
        assert Prediction.query.count() == 0


def test_admin_dashboard_has_shortcuts_to_every_admin_module(app, client):
    admin_id = add_user(app, role="admin")
    authenticate(client, admin_id)

    response = client.get("/dashboard")

    assert response.status_code == 200
    for path in (
        "/admin/dashboard",
        "/admin/companies",
        "/admin/users",
        "/admin/predictions",
        "/admin/login-logs",
        "/admin/activity-logs",
        "/admin/model-info",
        "/admin/company-comparison",
    ):
        assert path.encode() in response.data


def test_company_comparison_is_admin_only(app, client):
    user_id = add_user(app)
    authenticate(client, user_id)
    assert client.get("/admin/company-comparison").status_code == 403

    admin_id = add_user(app, name="Workspace Admin", email="admin@example.com", role="admin")
    authenticate(client, admin_id)
    response = client.get("/admin/company-comparison")
    assert response.status_code == 200
    assert b"Company comparison" in response.data


def test_admin_can_grant_and_revoke_company_access_for_users(app, client):
    admin_id = add_user(app, name="Access Admin", email="access-admin@example.com", role="admin")
    user_id = add_user(app, name="Assigned User")
    with app.app_context():
        company = Company(name="AssignedCo", slug="assignedco", industry="Retail")
        inactive_company = Company(
            name="InactiveCo", slug="inactiveco", industry="Retail", is_active=False
        )
        db.session.add_all([company, inactive_company])
        db.session.commit()
        company_id = company.id
        inactive_company_id = inactive_company.id

    authenticate(client, admin_id)
    page = client.get(f"/admin/users/{user_id}/access")
    assert page.status_code == 200
    assert b"AssignedCo" in page.data
    assert b"InactiveCo" not in page.data

    invalid = client.post(
        f"/admin/users/{user_id}/access",
        data={"company_ids": str(inactive_company_id)},
    )
    assert invalid.status_code == 200
    assert b"Invalid company selection" in invalid.data
    with app.app_context():
        assert UserCompanyAccess.query.filter_by(user_id=user_id).count() == 0

    response = client.post(
        f"/admin/users/{user_id}/access",
        data={"company_ids": [str(company_id)]},
    )
    assert response.status_code == 302
    with app.app_context():
        assert UserCompanyAccess.query.filter_by(
            user_id=user_id, company_id=company_id
        ).count() == 1
        assert AdminActivityLog.query.filter_by(action="granted company access").count() == 1

    authenticate(client, user_id)
    assert client.post(
        "/company/select",
        data={"company_id": str(company_id)},
    ).status_code == 302
    assert client.post(
        "/company/select",
        data={"company_id": str(inactive_company_id)},
    ).status_code == 403

    authenticate(client, admin_id)
    client.post(f"/admin/users/{user_id}/access", data={})
    with app.app_context():
        assert UserCompanyAccess.query.filter_by(user_id=user_id).count() == 0
        assert AdminActivityLog.query.filter_by(action="revoked company access").count() == 1

    authenticate(client, user_id)
    assert client.post(
        "/company/select",
        data={"company_id": str(company_id)},
    ).status_code == 403


def test_users_cannot_manage_company_access(app, client):
    user_id = add_user(app)
    target_id = add_user(app, name="Another User", email="another@example.com")
    authenticate(client, user_id)

    assert client.get(f"/admin/users/{target_id}/access").status_code == 403


def test_company_dashboard_card_opens_selected_workspace(app, client):
    user_id = add_user(app)
    with app.app_context():
        company = Company(name="Northstar", slug="northstar", industry="Telecom")
        db.session.add(company)
        db.session.flush()
        db.session.add(UserCompanyAccess(user_id=user_id, company_id=company.id))
        db.session.commit()
        company_id = company.id
    authenticate(client, user_id)

    dashboard = client.get("/dashboard")
    response = client.post("/company/select", data={
        "company_id": str(company_id),
        "destination": "prediction",
    })

    assert b"Northstar" in dashboard.data
    assert b"Open workspace" in dashboard.data
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/prediction")


def test_company_models_can_store_dataset_and_model_metadata(app):
    with app.app_context():
        admin = User(
            name="Admin",
            email="admin@example.com",
            password_hash="hash",
            role="admin",
        )
        db.session.add(admin)
        db.session.commit()

        company = Company(
            name="TeleConnect",
            slug="teleconnect",
            industry="Telecom",
            description="Customer churn platform",
            logo_color="#123456",
            created_by=admin.id,
            is_active=True,
        )
        db.session.add(company)
        db.session.commit()

        dataset = CompanyDataset(
            company_id=company.id,
            filename="teleconnect.csv",
            row_count=250,
            uploaded_by=admin.id,
            is_active=True,
            schema_json={"columns": ["Age", "Gender"], "target_column": "Churn", "positive_label": "Yes"},
            data_path="data/companies/teleconnect.csv",
        )
        db.session.add(dataset)
        db.session.commit()

        model_record = TrainedModel(
            company_id=company.id,
            dataset_id=dataset.id,
            algorithm="RandomForest",
            version=1,
            metrics_json={"accuracy": 0.82, "f1": 0.77, "roc_auc": 0.89},
            feature_importance_json={"Age": 0.42},
            model_path="models/teleconnect/v1.joblib",
            is_active=True,
            threshold_medium=0.35,
            threshold_high=0.65,
        )
        db.session.add(model_record)
        db.session.commit()

        access = UserCompanyAccess(user_id=admin.id, company_id=company.id)
        db.session.add(access)
        db.session.commit()

        prediction = Prediction(
            user_id=admin.id,
            company_id=company.id,
            model_id=model_record.id,
            age=35,
            gender="Male",
            tenure=12,
            monthly_charges=55.5,
            total_charges=660,
            contract_type="Month-to-Month",
            internet_service="Fiber",
            tech_support="No",
            prediction="Yes",
            probability=0.74,
            risk_level="High Risk",
            risk_factors=["Short tenure"],
            suggested_actions=["Call customer"],
            input_json={"Age": 35, "Gender": "Male"},
        )
        db.session.add(prediction)
        db.session.commit()

        assert Company.query.count() == 1
        assert CompanyDataset.query.filter_by(company_id=company.id).count() == 1
        assert TrainedModel.query.filter_by(company_id=company.id).count() == 1
        assert UserCompanyAccess.query.filter_by(user_id=admin.id, company_id=company.id).count() == 1
        assert Prediction.query.filter_by(company_id=company.id, model_id=model_record.id).count() == 1


def test_schema_and_training_services_work_on_synthetic_data(app, tmp_path):
    import pandas as pd

    with app.app_context():
        admin = User(
            name="Schema Admin",
            email="schema@example.com",
            password_hash="hash",
            role="admin",
        )
        db.session.add(admin)
        db.session.commit()

        company = Company(
            name="TestCompany",
            slug="testcompany",
            industry="Telecom",
            description="Synthetic test",
            logo_color="#111111",
            created_by=admin.id,
            is_active=True,
        )
        db.session.add(company)
        db.session.commit()

        data_path = tmp_path / "synthetic_company.csv"
        frame = pd.DataFrame({
            "CustomerID": [f"C{i:03d}" for i in range(1, 121)],
            "Age": [25 + (i % 40) for i in range(120)],
            "Gender": ["Male" if i % 2 == 0 else "Female" for i in range(120)],
            "Tenure": [6 + (i % 30) for i in range(120)],
            "MonthlyCharges": [50 + (i % 40) for i in range(120)],
            "ContractType": ["Month-to-Month" if i % 2 else "One Year" for i in range(120)],
            "TechSupport": ["No" if i % 3 else "Yes" for i in range(120)],
            "Churn": ["Yes" if (i % 5 == 0 or i % 7 == 0) else "No" for i in range(120)],
        })
        frame.to_csv(data_path, index=False)

        dataset = CompanyDataset(
            company_id=company.id,
            filename="synthetic_company.csv",
            row_count=len(frame),
            uploaded_by=admin.id,
            is_active=True,
            schema_json={
                "columns": list(frame.columns),
                "target_column": "Churn",
                "positive_label": "Yes",
                "id_columns": ["CustomerID"],
                "column_types": {
                    "Age": "numeric",
                    "Gender": "categorical",
                    "Tenure": "numeric",
                    "MonthlyCharges": "numeric",
                    "ContractType": "categorical",
                    "TechSupport": "categorical",
                },
                "allowed_categories": {
                    "Gender": ["Female", "Male"],
                    "ContractType": ["Month-to-Month", "One Year"],
                    "TechSupport": ["No", "Yes"],
                },
                "numeric_ranges": {
                    "Age": {"min": 18, "max": 80},
                    "Tenure": {"min": 0, "max": 72},
                    "MonthlyCharges": {"min": 0, "max": 200},
                },
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.commit()

        schema = infer_schema(frame, "Churn")
        quality = build_data_quality_report(frame, "Churn")
        assert schema["target_column"] == "Churn"
        assert quality["rows"] == 120
        assert "warnings" in quality

        model_record = train_company_model(company.id, dataset.id)
        assert model_record.company_id == company.id
        assert model_record.is_active is True
        assert model_record.model_path.endswith(".joblib")


def test_admin_company_routes_and_session_selection(app, client):
    with app.app_context():
        admin = User(name="Admin", email="company-admin@example.com", password_hash="hash", role="admin")
        db.session.add(admin)
        db.session.commit()
        admin_id = admin.id
        company = Company(name="TeleConnect", slug="teleconnect", industry="Telecom", description="Demo", logo_color="#123456", created_by=admin_id, is_active=True)
        db.session.add(company)
        db.session.commit()
        company_id = company.id
        access = UserCompanyAccess(user_id=admin_id, company_id=company_id)
        db.session.add(access)
        db.session.commit()

    authenticate(client, admin_id)
    response = client.get("/admin/companies")
    assert response.status_code == 200
    assert b"TeleConnect" in response.data

    switch = client.post("/company/select", data={"company_id": str(company_id)})
    assert switch.status_code == 302
    with client.session_transaction() as session:
        assert session.get("active_company_id") == str(company_id)


def test_bulk_upload_rejects_file_with_missing_value_in_row(app, client, tmp_path):
    import pandas as pd

    user_id = add_user(app)
    data_path = tmp_path / "bulk-company.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "Age": [20, 35, 50],
        "Gender": ["Female", "Male", "Female"],
        "Churn": ["No", "Yes", "No"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="BulkCo", slug="bulkco", industry="Retail")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="bulk-company.csv",
            row_count=3,
            schema_json={
                "columns": ["CustomerID", "Age", "Gender", "Churn"],
                "target_column": "Churn",
                "positive_label": "Yes",
                "id_columns": ["CustomerID"],
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        db.session.add_all([
            TrainedModel(
                company_id=company.id,
                dataset_id=dataset.id,
                algorithm="StubModel",
                model_path="stub.joblib",
                is_active=True,
            ),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
        company_id = company.id
    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)

    csv_data = (
        "Age,Gender\n"
        "30,Male\n"
        "25,\n"
    )

    response = client.post(
        "/bulk",
        data={"file": (io.BytesIO(csv_data.encode()), "customers.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"Row 3" in response.data
    assert b"data is wrong" in response.data.lower()
    assert b"Gender" in response.data


def test_company_csv_download_contains_uploadable_dataset_rows(app, client, monkeypatch, tmp_path):
    import pandas as pd

    user_id = add_user(app)
    data_path = tmp_path / "template-company.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "Age": [20, 35, 50],
        "Gender": ["Female", "Male", None],
        "Churn": ["No", "Yes", "No"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="TemplateCo", slug="templateco", industry="Retail")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="template-company.csv",
            row_count=3,
            schema_json={
                "columns": ["CustomerID", "Age", "Gender", "Churn"],
                "target_column": "Churn",
                "id_columns": ["CustomerID"],
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        db.session.add_all([
            TrainedModel(
                company_id=company.id,
                dataset_id=dataset.id,
                algorithm="StubModel",
                model_path="stub.joblib",
                is_active=True,
            ),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
        company_id = company.id
    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)

    template = client.get("/bulk/template.csv")
    assert template.status_code == 200
    frame = pd.read_csv(io.BytesIO(template.data))
    assert list(frame.columns) == ["Age", "Gender"]
    assert len(frame) == 2
    assert frame.iloc[0].to_dict() == {"Age": 20, "Gender": "Female"}

    monkeypatch.setattr(main_routes, "predict_many", lambda _company, data: data.assign(
        Prediction="No", Probability=0.1, RiskLevel="Low Risk"
    ))
    scored = client.post(
        "/bulk",
        data={"file": (io.BytesIO(template.data), "templateco-prediction-template.csv")},
        content_type="multipart/form-data",
    )
    assert scored.status_code == 200
    assert b"Prediction,Probability,Risk Level" in scored.data
    with app.app_context():
        assert Prediction.query.filter_by(company_id=company_id).count() == 2


def test_bulk_upload_emails_results_when_checked(app, client, monkeypatch, tmp_path):
    import io
    import pandas as pd
    from churn_app.services.email_service import reset_outgoing_email_rate_limits

    reset_outgoing_email_rate_limits()
    user_id = add_user(app, email="bulk-email@example.com")
    data_path = tmp_path / "template-company2.csv"
    pd.DataFrame({
        "CustomerID": ["C001", "C002"],
        "Age": [25, 40],
        "Gender": ["Female", "Male"],
        "Churn": ["No", "Yes"],
    }).to_csv(data_path, index=False)
    with app.app_context():
        company = Company(name="BulkEmailCo", slug="bulkemailco", industry="Retail")
        db.session.add(company)
        db.session.flush()
        dataset = CompanyDataset(
            company_id=company.id,
            filename="template-company2.csv",
            row_count=2,
            schema_json={
                "columns": ["CustomerID", "Age", "Gender", "Churn"],
                "target_column": "Churn",
                "id_columns": ["CustomerID"],
            },
            data_path=str(data_path),
        )
        db.session.add(dataset)
        db.session.flush()
        db.session.add_all([
            TrainedModel(
                company_id=company.id,
                dataset_id=dataset.id,
                algorithm="StubModel",
                model_path="stub.joblib",
                is_active=True,
            ),
            UserCompanyAccess(user_id=user_id, company_id=company.id),
        ])
        db.session.commit()
        company_id = company.id

    with client.session_transaction() as session:
        session["active_company_id"] = str(company_id)
    authenticate(client, user_id)

    app.config.update({
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": 587,
        "SMTP_USERNAME": "mailer@example.com",
        "SMTP_PASSWORD": "smtp-password",
        "SMTP_SENDER": "mailer@example.com",
        "SMTP_USE_TLS": True,
    })

    sent_messages = []

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            pass

        def login(self, *args):
            pass

        def send_message(self, message):
            sent_messages.append(message)

    monkeypatch.setattr("churn_app.services.email_service.smtplib.SMTP", FakeSMTP)
    monkeypatch.setattr(main_routes, "predict_many", lambda _company, data: data.assign(
        Prediction="No", Probability=0.15, RiskLevel="Low Risk"
    ))

    csv_upload = "Age,Gender\n25,Female\n40,Male\n"
    response = client.post(
        "/bulk",
        data={
            "file": (io.BytesIO(csv_upload.encode()), "customers.csv"),
            "email_results": "y",
            "file_format": "xlsx",
            "message": "Bulk scored results note.",
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Bulk predictions saved and emailed to your account" in response.data
    assert len(sent_messages) == 1
    assert sent_messages[0]["To"] == "bulk-email@example.com"
    attachments = list(sent_messages[0].iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "bulk-churn-predictions.xlsx"

