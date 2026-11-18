from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from churn_app import create_app
from churn_app.extensions import db
from churn_app.models import Company, CompanyDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "companies"

COMPANIES = [
    {"name": "TeleConnect", "slug": "teleconnect", "industry": "Telecom", "description": "Telecom retention platform", "logo_color": "#2563eb", "file": "teleconnect_telecom.csv"},
    {"name": "SecureBank", "slug": "securebank", "industry": "Banking", "description": "Banking customer risk monitoring", "logo_color": "#059669", "file": "securebank_banking.csv"},
    {"name": "ShopEase", "slug": "shopease", "industry": "E-commerce", "description": "Retail churn prevention", "logo_color": "#f59e0b", "file": "shopease_ecommerce.csv"},
    {"name": "StreamHub", "slug": "streamhub", "industry": "SaaS", "description": "Subscription churn monitor", "logo_color": "#8b5cf6", "file": "streamhub_saas.csv"},
    {"name": "SafeGuard", "slug": "safeguard", "industry": "Insurance", "description": "Policyholder retention analytics", "logo_color": "#ef4444", "file": "safeguard_insurance.csv"},
]


def main() -> None:
    app = create_app({
        "TESTING": False,
        "SECRET_KEY": "seed-demo-key",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{ROOT / 'instance' / 'churn.sqlite'}",
    })

    with app.app_context():
        for config in COMPANIES:
            company = Company.query.filter_by(slug=config["slug"]).first()
            if company is None:
                company = Company(
                    name=config["name"],
                    slug=config["slug"],
                    industry=config["industry"],
                    description=config["description"],
                    logo_color=config["logo_color"],
                    is_active=True,
                )
                db.session.add(company)
                db.session.flush()

            csv_path = DATA_DIR / config["file"]
            df = pd.read_csv(csv_path)
            dataset = CompanyDataset.query.filter_by(company_id=company.id, filename=config["file"]).first()
            if dataset is None:
                dataset = CompanyDataset(
                    company_id=company.id,
                    filename=config["file"],
                    row_count=int(len(df)),
                    uploaded_by=None,
                    is_active=True,
                    schema_json={
                        "columns": list(df.columns),
                        "target_column": "Churn",
                        "positive_label": "Yes",
                        "id_columns": [col for col in df.columns if "id" in col.lower()],
                    },
                    data_path=str(csv_path),
                )
                db.session.add(dataset)

        db.session.commit()
        print(f"Seeded {Company.query.count()} companies and {CompanyDataset.query.count()} datasets.")


if __name__ == "__main__":
    main()
