import re

from flask_wtf import FlaskForm
from wtforms import BooleanField, FileField, FloatField, IntegerField, PasswordField, SelectField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange, Optional, Regexp, ValidationError

from .models import User


def validate_password(_form, field):
    password = field.data or ""
    if not re.fullmatch(r"(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}", password):
        raise ValidationError("Use 8+ characters with uppercase, lowercase, a number, and a special character.")


def validate_name(_form, field):
    name = " ".join((field.data or "").split())
    if not 2 <= len(name) <= 50 or not all(character.isalpha() or character.isspace() for character in name):
        raise ValidationError("Use 2-50 letters and spaces only.")
    field.data = name


class RegisterForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(min=2, max=50), validate_name])
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=254)])
    password = PasswordField("Password", validators=[DataRequired(), validate_password])
    confirm_password = PasswordField("Confirm password", validators=[DataRequired(), EqualTo("password", message="Passwords must match.")])
    admin_code = PasswordField("Admin registration code", validators=[Optional(), Length(max=256)])
    submit = SubmitField("Create account")

    def validate_email(self, field):
        email = (field.data or "").strip().lower()
        if User.query.filter_by(email=email).first():
            raise ValidationError("An account with this email already exists.")
        field.data = email


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=254)])
    password = PasswordField("Password", validators=[DataRequired()])
    remember_me = BooleanField("Remember me")
    submit = SubmitField("Sign in")


class ProfileForm(FlaskForm):
    name = StringField("Full name", validators=[DataRequired(), Length(min=2, max=50), validate_name])
    submit = SubmitField("Save profile")


class ChangePasswordForm(FlaskForm):
    current_password = PasswordField("Current password", validators=[DataRequired()])
    password = PasswordField("New password", validators=[DataRequired(), validate_password])
    confirm_password = PasswordField("Confirm new password", validators=[DataRequired(), EqualTo("password", message="Passwords must match.")])
    submit = SubmitField("Update password")


class PredictionForm(FlaskForm):
    age = IntegerField("Age", validators=[DataRequired(), NumberRange(min=18, max=100, message="Age must be between 18 and 100.")])
    gender = SelectField("Gender", validators=[DataRequired()])
    tenure = IntegerField("Tenure (months)", validators=[DataRequired(), NumberRange(min=0, max=120, message="Tenure must be between 0 and 120 months.")])
    monthly = FloatField("Monthly charges", validators=[DataRequired(), NumberRange(min=0)])
    contract = SelectField("Contract type", validators=[DataRequired()])
    internet = SelectField("Internet service", validators=[DataRequired()])
    total = FloatField("Total charges", validators=[DataRequired(), NumberRange(min=0)])
    support = SelectField("Tech support", validators=[DataRequired()])
    submit = SubmitField("Predict churn")


class VerifyEmailForm(FlaskForm):
    code = StringField(
        "Verification code",
        validators=[
            DataRequired(message="Enter the 6-digit verification code."),
            Length(min=6, max=6, message="The verification code must be 6 digits."),
            Regexp(r"^\d{6}$", message="The verification code must be 6 digits."),
        ],
    )
    submit = SubmitField("Verify email")


class BulkUploadForm(FlaskForm):
    file = FileField("Customer CSV", validators=[DataRequired()])
    email_results = BooleanField("Email results to my account")
    file_format = SelectField(
        "File format",
        choices=[("csv", "CSV"), ("xlsx", "Excel (.xlsx)"), ("both", "Both (CSV and Excel)")],
        default="csv",
        validate_choice=False,
    )
    message = TextAreaField("Message (optional)", validators=[Optional(), Length(max=1000)])
    submit = SubmitField("Run bulk prediction")


class EmailHistoryForm(FlaskForm):
    file_format = SelectField(
        "File format",
        choices=[("csv", "CSV"), ("xlsx", "Excel (.xlsx)"), ("both", "Both (CSV and Excel)")],
        default="csv",
        validate_choice=False,
    )
    message = TextAreaField("Message (optional)", validators=[Optional(), Length(max=1000)])
    submit = SubmitField("Email to my account")
