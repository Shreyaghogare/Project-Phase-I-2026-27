# Customer Churn Prediction

A Flask application for individual and batch churn predictions using the project's saved Random Forest model. It includes account roles, per-user prediction history, administration tools, login auditing, CSV export, and PDF reports.

## Requirements

- Use the project’s Python 3.14 virtual environment from `.venv` for all local runs. The app and saved model artifacts are regenerated under the current interpreter to avoid scikit-learn pickle compatibility warnings.
- `cleaned_dataset.csv`, `churn_model.pkl`, and `encoders.pkl` must remain in the project root. The optional `label_encoder.pkl` is not used by the application.
- A legacy validation inconsistency remains in the old prediction flow: `predict_customer()` rejects `Tenure > 120`, even though some real rows in `cleaned_dataset.csv` contain `Tenure = 122`. This will be replaced by schema-driven validation in the multi-company upgrade.

## Setup and Run (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$secret = py -c "import secrets; print(secrets.token_hex(32))"
$adminCode = py -c "import secrets; print(secrets.token_urlsafe(24))"
@("SECRET_KEY=$secret", "ADMIN_REGISTRATION_CODE=$adminCode") | Set-Content .env
flask --app app run --debug
```

Open `http://127.0.0.1:5000`. The VS Code launch profiles read the same `.env` file. Use the value of `ADMIN_REGISTRATION_CODE` on the registration form only when creating an administrator; leave it blank for a regular user. Both roles sign in at `/login` with the same email/password form and are sent to their role-specific dashboard. Keep `.env` private. To run with `python app.py` instead of the Flask CLI, set both values in the current PowerShell session first:

```powershell
$env:SECRET_KEY = (py -c "import secrets; print(secrets.token_hex(32))")
$env:ADMIN_REGISTRATION_CODE = (py -c "import secrets; print(secrets.token_urlsafe(24))")
python app.py
```

The SQLite database is created automatically at `instance/churn.sqlite`. New accounts are regular Users unless a valid admin registration code is provided. Existing accounts keep their current roles; administrators can also manage roles from **Admin → Users**. For existing databases created prior to email verification, run the database migration script:
```powershell
python scripts/migrate_v3.py
```

## Usage

- **Register and Email Verification:** Registration validates names, email addresses, and password strength. Upon registration, a 6-digit one-time verification code (OTP) is sent to the registered email address. The code expires in 10 minutes and allows up to 5 attempts before being locked. Users must verify their email at `/verify-email` before signing in. A 60-second cooldown and a limit of 5 requests per hour protect code resending. In local development (`FLASK_DEBUG=1` or testing), generated verification codes are also written to the application logger.
- **Sign In:** Both Admins and Users sign in at `/login`. Accounts with unverified email addresses are prompted to complete email verification before accessing the workspace.
- **Workspace Predictions:** Select a company workspace from the dashboard. Predictions use only that company's active dataset and trained model; numeric values must stay within observed dataset ranges and categories must match dataset values.
- **Train Models:** Before predicting for the first time, train active company datasets with `python scripts/train_company_models.py` from the activated project environment.
- **Bulk Prediction & Email Delivery:** Upload a CSV using the company-specific template on the Bulk Prediction page (up to 500 rows). You can check "Email results to my registered account" with an optional note and choose CSV, Excel (.xlsx), or both. When unchecked, results are directly downloaded as CSV.
- **Prediction History & Export:** Browse, filter, and export prediction history. Individual prediction PDFs can be generated directly. Users can email filtered records to their registered account as CSV, Excel (.xlsx), or both with an optional note. Outgoing exports are protected against formula injection, restricted to 5 MB per email, and rate-limited to 10 emails per hour per user.
- **SMTP Setup:** To enable verification emails and data export delivery, configure SMTP settings in `.env` (such as Gmail with App Passwords or Outlook). Ensure `SMTP_HOST`, `SMTP_PORT` (default 587), `SMTP_USE_TLS=true`, `SMTP_USERNAME`, `SMTP_PASSWORD`, and `SMTP_SENDER` are set.
- **Admin Management:** Admins can manage accounts and roles, grant or revoke user access to active company workspaces from **Users → Manage access**, review workspace-wide predictions and audit logs, and inspect the model card. Users can select and predict only for companies granted to their account; administrators automatically see all active companies.
- **Risk Thresholds:** Prediction-risk bands are Low below 30%, Medium from 30% to below 60%, and High at 60% or higher.

## Tests

```powershell
pytest
```

The tests cover registration, OTP email verification flow (including expiration, attempt lockout, and resend cooldown), existing-account sign-in, login lockout/auditing, export generation (CSV and Excel .xlsx with openpyxl verification), formula injection escaping, outgoing email delivery, admin access controls, and prediction persistence.

## Project Structure

```text
ABCD/
├── app.py                         # Flask entry point
├── config.py                      # Environment and session configuration
├── churn_app/
│   ├── __init__.py                 # Application factory
│   ├── extensions.py               # SQLAlchemy, Flask-Login, CSRF
│   ├── forms.py                    # WTForms validation
│   ├── models.py                   # Current multi-company data model
│   ├── security.py                 # Admin authorization
│   ├── services/                  # Schema, data-quality, training, prediction logic
│   └── blueprints/
│       ├── auth/                   # Registration, login, logout
│       ├── main/                   # Dashboard, prediction, history, profile
│       └── admin/                  # Admin dashboard and company management
├── scripts/                       # Dataset generation and migration helpers
├── data/                          # Synthetic company datasets and metadata
├── templates/                     # Shared, user, admin, and error pages
├── static/                        # Styles and bulk CSV template
├── tests/                         # Pytest suite
├── legacy/                        # Archived single-company scripts kept for reference
├── cleaned_dataset.csv            # Legacy dataset retained for compatibility
├── churn_model.pkl                # Legacy saved model artifact
├── encoders.pkl                  # Legacy encoder artifact
├── README.md
├── requirements.txt
├── run_demo.ps1
└── .env.example
```

## Screenshots

Capture desktop and mobile screenshots into `docs/screenshots/` when preparing a report or demonstration:

- `register.png` — registration and password-strength meter
- `user-dashboard.png` — personal KPIs, charts, and prediction result
- `history.png` — filters, details dialog, and prediction table
- `admin-dashboard.png` — system KPIs and activity charts
- `admin-users.png` — user and role management