from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "companies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def add_missing(frame: pd.DataFrame, columns: list[str], rate: float = 0.02) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        mask = rng.random(len(result)) < rate
        result.loc[mask, column] = np.nan
    return result


def write_company(rows: int, filename: str, columns: list[str], generator) -> None:
    frame = generator(rows)
    frame = frame[columns]
    frame.to_csv(OUT_DIR / filename, index=False)


def teleconnect(rows: int) -> pd.DataFrame:
    age = rng.integers(18, 72, size=rows)
    gender = rng.choice(["Male", "Female"], size=rows)
    tenure = rng.integers(1, 72, size=rows)
    monthly = np.clip(rng.normal(68, 24, size=rows), 20, 180)
    total = np.maximum(0, monthly * (tenure + rng.integers(0, 18, size=rows)))
    contract = rng.choice(["Month-to-Month", "One Year", "Two Year"], size=rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber Optic", "No Internet"], size=rows, p=[0.35, 0.45, 0.20])
    support = rng.choice(["No", "Yes"], size=rows, p=[0.60, 0.40])
    payment = rng.choice(["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"], size=rows)
    support_calls = rng.integers(0, 12, size=rows)
    score = (
        -1.5 * (tenure / 12)
        + 0.65 * (monthly / 50)
        + 1.3 * (contract == "Month-to-Month")
        + 1.1 * (internet == "Fiber Optic")
        + 1.8 * (support == "No")
        + 0.4 * (support_calls > 4)
        + rng.normal(0, 0.6, size=rows)
    )
    churn = (rng.random(rows) < logistic(score - 1.7)).astype(int)
    frame = pd.DataFrame({
        "CustomerID": [f"TC{idx:05d}" for idx in range(1, rows + 1)],
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "ContractType": contract,
        "InternetService": internet,
        "TechSupport": support,
        "PaymentMethod": payment,
        "SupportCalls": support_calls,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return add_missing(frame, ["Age", "MonthlyCharges", "SupportCalls", "PaymentMethod"]) 


def banking(rows: int) -> pd.DataFrame:
    geography = rng.choice(["France", "Germany", "Spain"], size=rows)
    age = rng.integers(22, 70, size=rows)
    gender = rng.choice(["Male", "Female"], size=rows)
    credit_score = rng.integers(300, 850, size=rows)
    balance = np.clip(rng.normal(100000, 45000, size=rows), 0, 400000)
    products = rng.integers(1, 4, size=rows)
    has_card = rng.choice([0, 1], size=rows, p=[0.35, 0.65])
    active = rng.choice([0, 1], size=rows, p=[0.45, 0.55])
    salary = np.clip(rng.normal(110000, 45000, size=rows), 20000, 300000)
    tenure = rng.integers(1, 15, size=rows)
    score = (
        0.015 * (credit_score - 600)
        + 0.00001 * balance
        + 0.8 * (products == 1)
        + 1.3 * (active == 0)
        + 0.7 * (tenure < 3)
        + rng.normal(0, 1.2, size=rows)
    )
    churn = (rng.random(rows) < logistic(score - 1.8)).astype(int)
    frame = pd.DataFrame({
        "CustomerID": [f"BK{idx:06d}" for idx in range(1, rows + 1)],
        "Age": age,
        "Gender": gender,
        "Geography": geography,
        "CreditScore": credit_score,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCreditCard": has_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
        "TenureYears": tenure,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return add_missing(frame, ["Balance", "CreditScore", "EstimatedSalary"]) 


def ecommerce(rows: int) -> pd.DataFrame:
    age = rng.integers(18, 75, size=rows)
    gender = rng.choice(["Male", "Female", "Other"], size=rows)
    tenure = rng.integers(1, 48, size=rows)
    orders = rng.integers(0, 25, size=rows)
    avg_order = np.clip(rng.normal(52, 30, size=rows), 5, 300)
    days_since = rng.integers(1, 180, size=rows)
    return_rate = np.clip(rng.normal(0.18, 0.10, size=rows), 0.01, 0.80)
    category = rng.choice(["Electronics", "Fashion", "Home", "Beauty", "Groceries"], size=rows)
    app = rng.choice(["Yes", "No"], size=rows, p=[0.68, 0.32])
    complaints = rng.integers(0, 6, size=rows)
    score = (
        0.02 * (days_since - 30)
        + 0.13 * (complaints > 1)
        + 0.8 * (return_rate > 0.35)
        + 1.2 * (app == "No")
        + 1.0 * (tenure < 6)
        + rng.normal(0, 1.0, size=rows)
    )
    churn = (rng.random(rows) < logistic(score - 1.25)).astype(int)
    frame = pd.DataFrame({
        "CustomerID": [f"SE{idx:06d}" for idx in range(1, rows + 1)],
        "Age": age,
        "Gender": gender,
        "TenureMonths": tenure,
        "OrdersLast90Days": orders,
        "AvgOrderValue": avg_order,
        "DaysSinceLastOrder": days_since,
        "ReturnRate": return_rate,
        "PreferredCategory": category,
        "UsesApp": app,
        "ComplaintsFiled": complaints,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return add_missing(frame, ["OrdersLast90Days", "AvgOrderValue", "DaysSinceLastOrder"]) 


def saas(rows: int) -> pd.DataFrame:
    plan = rng.choice(["Basic", "Standard", "Premium", "Enterprise"], size=rows)
    months = rng.integers(1, 48, size=rows)
    fee = np.clip(rng.normal(28, 14, size=rows), 8, 120)
    watch_hours = np.clip(rng.normal(6, 4, size=rows), 0, 30)
    devices = rng.integers(1, 6, size=rows)
    tickets = rng.integers(0, 12, size=rows)
    last_login = rng.integers(1, 90, size=rows)
    payment_failures = rng.integers(0, 6, size=rows)
    auto_renew = rng.choice(["Yes", "No"], size=rows, p=[0.70, 0.30])
    score = (
        0.8 * (plan == "Basic")
        + 0.5 * (months < 6)
        + 0.7 * (watch_hours < 2)
        + 0.75 * (devices > 4)
        + 0.6 * (tickets > 3)
        + 0.7 * (payment_failures > 2)
        + 1.0 * (auto_renew == "No")
        + rng.normal(0, 0.9, size=rows)
    )
    churn = (rng.random(rows) < logistic(score - 2.2)).astype(int)
    frame = pd.DataFrame({
        "UserID": [f"SH{idx:07d}" for idx in range(1, rows + 1)],
        "PlanType": plan,
        "MonthsSubscribed": months,
        "MonthlyFee": fee,
        "AvgWatchHoursPerWeek": watch_hours,
        "DevicesUsed": devices,
        "SupportTickets": tickets,
        "LastLoginDaysAgo": last_login,
        "PaymentFailures": payment_failures,
        "AutoRenew": auto_renew,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return add_missing(frame, ["MonthlyFee", "LastLoginDaysAgo", "SupportTickets"]) 


def insurance(rows: int) -> pd.DataFrame:
    age = rng.integers(21, 75, size=rows)
    gender = rng.choice(["Male", "Female"], size=rows)
    policy_type = rng.choice(["Auto", "Home", "Life", "Health"], size=rows)
    tenure = np.clip(rng.normal(7, 4, size=rows), 0.5, 30)
    premium = np.clip(rng.normal(1800, 700, size=rows), 300, 8000)
    claims = rng.integers(0, 6, size=rows)
    missed = rng.integers(0, 4, size=rows)
    contacts = rng.integers(0, 10, size=rows)
    discount = rng.choice(["Yes", "No"], size=rows, p=[0.45, 0.55])
    score = (
        0.04 * (age - 40)
        + 0.6 * (claims > 2)
        + 1.2 * (missed > 0)
        + 0.5 * (tenure < 2)
        + 1.0 * (discount == "No")
        + 0.4 * (contacts > 5)
        + rng.normal(0, 0.9, size=rows)
    )
    churn = (rng.random(rows) < logistic(score - 1.9)).astype(int)
    frame = pd.DataFrame({
        "PolicyID": [f"IN{idx:07d}" for idx in range(1, rows + 1)],
        "Age": age,
        "Gender": gender,
        "PolicyType": policy_type,
        "PolicyTenureYears": tenure,
        "AnnualPremium": premium,
        "ClaimsLast3Years": claims,
        "MissedPayments": missed,
        "AgentContactsPerYear": contacts,
        "DiscountApplied": discount,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return add_missing(frame, ["AnnualPremium", "PolicyTenureYears", "AgentContactsPerYear"]) 


COMPANIES = {
    "teleconnect_telecom.csv": (3000, [
        "CustomerID", "Age", "Gender", "Tenure", "MonthlyCharges", "TotalCharges",
        "ContractType", "InternetService", "TechSupport", "PaymentMethod",
        "SupportCalls", "Churn"
    ], teleconnect),
    "securebank_banking.csv": (3200, [
        "CustomerID", "Age", "Gender", "Geography", "CreditScore", "Balance",
        "NumOfProducts", "HasCreditCard", "IsActiveMember", "EstimatedSalary",
        "TenureYears", "Churn"
    ], banking),
    "shopease_ecommerce.csv": (2800, [
        "CustomerID", "Age", "Gender", "TenureMonths", "OrdersLast90Days",
        "AvgOrderValue", "DaysSinceLastOrder", "ReturnRate", "PreferredCategory",
        "UsesApp", "ComplaintsFiled", "Churn"
    ], ecommerce),
    "streamhub_saas.csv": (2700, [
        "UserID", "PlanType", "MonthsSubscribed", "MonthlyFee",
        "AvgWatchHoursPerWeek", "DevicesUsed", "SupportTickets", "LastLoginDaysAgo",
        "PaymentFailures", "AutoRenew", "Churn"
    ], saas),
    "safeguard_insurance.csv": (2600, [
        "PolicyID", "Age", "Gender", "PolicyType", "PolicyTenureYears",
        "AnnualPremium", "ClaimsLast3Years", "MissedPayments", "AgentContactsPerYear",
        "DiscountApplied", "Churn"
    ], insurance),
}


if __name__ == "__main__":
    for filename, (rows, columns, generator) in COMPANIES.items():
        write_company(rows, filename, columns, generator)
        print(f"Created {OUT_DIR / filename}")
