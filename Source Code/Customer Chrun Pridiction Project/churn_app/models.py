from datetime import datetime, timezone

from flask_login import UserMixin

from .extensions import db


def utc_now():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), nullable=False, default="user")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    failed_attempts = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    last_login = db.Column(db.DateTime(timezone=True))
    email_verified = db.Column(db.Boolean, nullable=False, default=False)
    email_verified_at = db.Column(db.DateTime(timezone=True))
    predictions = db.relationship("Prediction", backref="user", cascade="all, delete-orphan")
    login_logs = db.relationship("LoginLog", backref="user", passive_deletes=True)
    created_companies = db.relationship("Company", foreign_keys="Company.created_by", backref="creator")
    company_accesses = db.relationship("UserCompanyAccess", backref="user", cascade="all, delete-orphan")
    email_otps = db.relationship("EmailOTP", backref="user", cascade="all, delete-orphan", passive_deletes=True)


class EmailOTP(db.Model):
    __tablename__ = "email_otp"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    code_hash = db.Column(db.String(255), nullable=False)
    purpose = db.Column(db.String(50), nullable=False, default="verify_email")
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    used_at = db.Column(db.DateTime(timezone=True))


class Company(db.Model):
    __tablename__ = "company"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    slug = db.Column(db.String(120), unique=True, nullable=False, index=True)
    industry = db.Column(db.String(80), nullable=False)
    description = db.Column(db.Text, default="")
    logo_color = db.Column(db.String(20), default="#4f46e5")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    datasets = db.relationship("CompanyDataset", backref="company", cascade="all, delete-orphan")
    trained_models = db.relationship("TrainedModel", backref="company", cascade="all, delete-orphan")
    predictions = db.relationship("Prediction", backref="company_ref", cascade="all, delete-orphan")
    access_rows = db.relationship("UserCompanyAccess", backref="company", cascade="all, delete-orphan")


class CompanyDataset(db.Model):
    __tablename__ = "company_dataset"

    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    row_count = db.Column(db.Integer, nullable=False, default=0)
    uploaded_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    uploaded_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    schema_json = db.Column(db.JSON, nullable=False, default=dict)
    data_path = db.Column(db.String(500), nullable=True)
    trained_models = db.relationship("TrainedModel", backref="dataset", cascade="all, delete-orphan")


class TrainedModel(db.Model):
    __tablename__ = "trained_model"

    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("company_dataset.id"), nullable=False, index=True)
    algorithm = db.Column(db.String(80), nullable=False, default="RandomForest")
    version = db.Column(db.Integer, nullable=False, default=1)
    trained_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    metrics_json = db.Column(db.JSON, nullable=False, default=dict)
    feature_importance_json = db.Column(db.JSON, nullable=False, default=dict)
    model_path = db.Column(db.String(500), nullable=False)
    is_active = db.Column(db.Boolean, nullable=False, default=False)
    threshold_medium = db.Column(db.Float, nullable=False, default=0.3)
    threshold_high = db.Column(db.Float, nullable=False, default=0.6)
    predictions = db.relationship("Prediction", backref="model_ref", cascade="all, delete-orphan")


class UserCompanyAccess(db.Model):
    __tablename__ = "user_company_access"

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), primary_key=True, index=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"), primary_key=True, index=True)
    granted_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)


class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"), nullable=True, index=True)
    model_id = db.Column(db.Integer, db.ForeignKey("trained_model.id"), nullable=True, index=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(30), nullable=True)
    tenure = db.Column(db.Integer, nullable=True)
    monthly_charges = db.Column(db.Float, nullable=True)
    total_charges = db.Column(db.Float, nullable=True)
    contract_type = db.Column(db.String(50), nullable=True)
    internet_service = db.Column(db.String(50), nullable=True)
    tech_support = db.Column(db.String(30), nullable=True)
    prediction = db.Column(db.String(3), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    risk_factors = db.Column(db.JSON, nullable=False, default=list)
    suggested_actions = db.Column(db.JSON, nullable=False, default=list)
    input_json = db.Column(db.JSON, nullable=True, default=dict)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class LoginLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"), index=True)
    email_entered = db.Column(db.String(254), nullable=False)
    success = db.Column(db.Boolean, nullable=False)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class AdminActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"))
    action = db.Column(db.String(100), nullable=False)
    target = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    admin = db.relationship("User", foreign_keys=[admin_id])
