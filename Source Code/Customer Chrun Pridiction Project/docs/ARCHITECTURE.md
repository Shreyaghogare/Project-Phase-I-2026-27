# Architecture overview

```mermaid
flowchart TD
    User[User / Admin] --> App[Flask App Factory]
    App --> Auth[Auth Blueprint]
    App --> Main[Main Blueprint]
    App --> Admin[Admin Blueprint]
    App --> Services[Services Layer]
    Services --> Schema[Schema Service]
    Services --> Quality[Data Quality Service]
    Services --> Train[Training Service]
    Services --> Predict[Prediction Service]
    Train --> Model[Stored joblib models]
    Main --> DB[(SQLite / SQLAlchemy)]
    Admin --> DB
    Predict --> Model
    CompanyData[CSV datasets] --> Train
    CompanyData --> DB
```

## Design summary

The project keeps the original Flask app structure but adds a multi-company model layer. Each company has its own dataset, schema, trained model, thresholds, and access rules. This allows the application to be adapted to different industries without hard-coded rules in the route layer.

## Why this is scalable

- Each company has a dedicated dataset file and model snapshot.
- The schema is stored as JSON to allow different columns per company.
- The training layer builds a generic pipeline using scikit-learn.
- Prediction and risk logic are separated from the route code to improve testing and maintainability.
