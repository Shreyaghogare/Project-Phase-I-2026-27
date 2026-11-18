# Multi-company churn dataset library

This folder contains five realistic company datasets for the upgraded churn platform. Each dataset is synthetic but designed to resemble a real churn problem in a different industry.

## Included datasets

### 1. TeleConnect (Telecom)
- Key features: `Age`, `Tenure`, `MonthlyCharges`, `ContractType`, `InternetService`, `SupportCalls`
- Typical churn drivers: short tenure, month-to-month contracts, weak support coverage, higher monthly charges

### 2. SecureBank (Banking)
- Key features: `CreditScore`, `Balance`, `NumOfProducts`, `IsActiveMember`, `TenureYears`
- Typical churn drivers: low product count, inactive accounts, low tenure, high balances with low engagement

### 3. ShopEase (E-commerce)
- Key features: `OrdersLast90Days`, `AvgOrderValue`, `DaysSinceLastOrder`, `ReturnRate`, `ComplaintsFiled`
- Typical churn drivers: infrequent purchases, high return rates, poor app usage, customer dissatisfaction

### 4. StreamHub (SaaS)
- Key features: `PlanType`, `MonthsSubscribed`, `MonthlyFee`, `SupportTickets`, `PaymentFailures`, `AutoRenew`
- Typical churn drivers: low subscription length, payment failures, unpaid renewals, limited engagement

### 5. SafeGuard (Insurance)
- Key features: `PolicyTenureYears`, `AnnualPremium`, `ClaimsLast3Years`, `MissedPayments`, `AgentContactsPerYear`
- Typical churn drivers: missed payments, frequent claims, low tenure, weak policy retention interactions

## Notes

These datasets can be replaced or supplemented by real customer churn files in the same upload format. The system expects a target column named `Churn`, with `Yes` as the positive churn label in the generated examples.

The files are safe to use for experimentation, model training, and demo deployment. They are intentionally synthetic and should not be treated as production customer data.
