# Viva Q&A

## 1. Why not use one single model for all companies?
Because each company has different columns, customer behaviour, and retention drivers. A single hard-coded model would be brittle and inaccurate.

## 2. Why use pipelines?
Pipelines ensure preprocessing and model fitting are kept together, which simplifies retraining and prevents data leakage.

## 3. Why is class imbalance important?
Churn is often a minority class. Accuracy alone can be misleading, so precision, recall, F1, and ROC-AUC should be monitored.

## 4. What changed in the upgrade?
The project now supports multiple companies, schema-driven data handling, and versioned models per company.

## 5. Why are datasets stored per company?
It isolates the data lifecycle, reduces contamination between datasets, and enables per-company model management.

## 6. What is the role of schema inference?
It identifies numeric vs categorical columns, target labels, and allowed categories so the application can validate new data without hard-coded rules.

## 7. Why are there multiple algorithms?
Model comparison reduces overreliance on one algorithm and helps select the best model by ROC-AUC and F1.

## 8. Why are models versioned?
Version control allows rollback and comparison between model generations.

## 9. Why are company access rules important?
They enforce authorization boundaries and ensure users only see the companies they are allowed to work with.

## 10. Why is data quality reporting included?
Poor-quality input can distort churn predictions; early warnings help the user clean the data before training.
