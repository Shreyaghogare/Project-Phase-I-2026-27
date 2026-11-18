from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from churn_app import create_app
from churn_app.models import Company, CompanyDataset
from churn_app.services import train_company_model


def main() -> None:
    app = create_app({
        "SECRET_KEY": os.environ.get("SECRET_KEY", "local-company-model-training"),
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{ROOT / 'instance' / 'churn.sqlite'}",
    })
    failures = []
    with app.app_context():
        companies = Company.query.filter_by(is_active=True).order_by(Company.name.asc()).all()
        for company in companies:
            dataset = CompanyDataset.query.filter_by(
                company_id=company.id,
                is_active=True,
            ).order_by(CompanyDataset.uploaded_at.desc()).first()
            if dataset is None:
                print(f"Skipped {company.name}: no active dataset.")
                continue
            try:
                model = train_company_model(company.id, dataset.id)
                accuracy = model.metrics_json.get("accuracy")
                score = f"{accuracy:.1%}" if accuracy is not None else "unavailable"
                print(f"Trained {company.name} with {model.algorithm} (accuracy {score}).")
            except (OSError, ValueError, KeyError) as error:
                failures.append((company.name, str(error)))
                print(f"Failed {company.name}: {error}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()