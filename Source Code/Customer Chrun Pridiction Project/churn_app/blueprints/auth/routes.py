import hmac
import smtplib
from datetime import timedelta, timezone

from flask import current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from ...extensions import db
from ...forms import LoginForm, RegisterForm, VerifyEmailForm
from ...models import EmailOTP, LoginLog, User, utc_now
from ...services.email_service import EmailConfigurationError, send_verification_otp
from . import bp

LOCKOUT_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
RESEND_COOLDOWN_SECONDS = 60
MAX_RESENDS_PER_HOUR = 5
MAX_OTP_ATTEMPTS = 5


def _locked(user, now):
    if not user.locked_until:
        return False
    locked_until = user.locked_until
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until > now


def _log_attempt(user, email, success):
    db.session.add(LoginLog(
        user_id=user.id if user else None,
        email_entered=(email or "")[:254],
        success=success,
        ip_address=request.remote_addr,
        user_agent=(request.user_agent.string or "")[:500],
    ))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard" if current_user.role == "admin" else "main.dashboard"))
    form = RegisterForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        admin_code = (form.admin_code.data or "").strip()
        configured_admin_code = current_app.config.get("ADMIN_REGISTRATION_CODE") or ""
        if admin_code and (
            not configured_admin_code
            or not hmac.compare_digest(admin_code, configured_admin_code)
        ):
            form.admin_code.errors.append("The admin registration code is invalid.")
            return render_template("register.html", form=form)

        is_first_user = db.session.query(User.id).first() is None
        role = "admin" if (is_first_user or admin_code) else "user"
        user = User(
            name=form.name.data,
            email=email,
            password_hash=generate_password_hash(form.password.data),
            role=role,
            email_verified=False,
        )
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            form.email.errors.append("An account with this email already exists.")
            return render_template("register.html", form=form)

        session["verify_email"] = user.email

        try:
            send_verification_otp(user)
            flash("A 6-digit verification code has been sent to your email address.", "info")
        except EmailConfigurationError as error:
            current_app.logger.error("Could not send verification email: %s", error)
            flash(f"Email delivery is not configured: {error}", "error")
        except (OSError, smtplib.SMTPException) as error:
            current_app.logger.error("Could not send verification email: %s", error)
            flash("Verification email could not be sent. Check SMTP settings and try resending.", "error")

        return redirect(url_for("auth.verify_email", email=user.email))

    return render_template("register.html", form=form)


@bp.route("/verify-email", methods=["GET", "POST"])
def verify_email():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard" if current_user.role == "admin" else "main.dashboard"))

    email = request.args.get("email") or session.get("verify_email")
    if not email:
        flash("Please sign in or register to verify your account.", "warning")
        return redirect(url_for("auth.login"))

    user = User.query.filter_by(email=email.strip().lower()).first()
    if not user:
        flash("Account not found. Please register.", "error")
        return redirect(url_for("auth.register"))

    if user.email_verified:
        flash("Your email address is already verified. You can sign in.", "info")
        return redirect(url_for("auth.login"))

    form = VerifyEmailForm()
    if form.validate_on_submit():
        submitted_code = form.code.data.strip()
        now = utc_now()
        otp = (
            EmailOTP.query.filter_by(user_id=user.id, purpose="verify_email", used_at=None)
            .order_by(EmailOTP.created_at.desc())
            .first()
        )

        if not otp:
            flash("No verification code found. Please request a new code.", "error")
            return render_template("verify_email.html", form=form, email=user.email)

        if otp.attempts >= MAX_OTP_ATTEMPTS:
            flash("This verification code has been locked due to too many failed attempts. Please request a new code.", "error")
            return render_template("verify_email.html", form=form, email=user.email)

        expires_at = otp.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if now > expires_at:
            flash("Your verification code has expired. Please request a new code.", "error")
            return render_template("verify_email.html", form=form, email=user.email)

        if check_password_hash(otp.code_hash, submitted_code):
            otp.used_at = now
            user.email_verified = True
            user.email_verified_at = now
            db.session.commit()
            session.pop("verify_email", None)
            flash("Your email has been verified! You can now sign in.", "success")
            return redirect(url_for("auth.login"))
        else:
            otp.attempts += 1
            if otp.attempts >= MAX_OTP_ATTEMPTS:
                otp.expires_at = now
                db.session.commit()
                flash("Too many failed attempts. This verification code has been invalidated. Please request a new code.", "error")
            else:
                db.session.commit()
                remaining = MAX_OTP_ATTEMPTS - otp.attempts
                flash(f"Invalid verification code. You have {remaining} attempt{'s' if remaining != 1 else ''} remaining.", "error")
            return render_template("verify_email.html", form=form, email=user.email)

    return render_template("verify_email.html", form=form, email=user.email)


@bp.route("/resend-code", methods=["POST"])
def resend_code():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard" if current_user.role == "admin" else "main.dashboard"))

    email = request.form.get("email") or session.get("verify_email")
    if not email:
        flash("Please provide an email address to resend the code.", "warning")
        return redirect(url_for("auth.login"))

    user = User.query.filter_by(email=email.strip().lower()).first()
    if not user:
        flash("If an account exists with this email, a code will be sent.", "info")
        return redirect(url_for("auth.login"))

    if user.email_verified:
        flash("Your email address is already verified. You can sign in.", "info")
        return redirect(url_for("auth.login"))

    now = utc_now()
    one_hour_ago = now - timedelta(hours=1)
    sends_last_hour = EmailOTP.query.filter(
        EmailOTP.user_id == user.id,
        EmailOTP.purpose == "verify_email",
        EmailOTP.created_at >= one_hour_ago,
    ).count()
    if sends_last_hour >= MAX_RESENDS_PER_HOUR:
        flash(f"You have reached the maximum of {MAX_RESENDS_PER_HOUR} verification requests per hour. Please try again later.", "error")
        return redirect(url_for("auth.verify_email", email=user.email))

    last_otp = (
        EmailOTP.query.filter_by(user_id=user.id, purpose="verify_email")
        .order_by(EmailOTP.created_at.desc())
        .first()
    )
    if last_otp:
        created_at = last_otp.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        elapsed = (now - created_at).total_seconds()
        if elapsed < RESEND_COOLDOWN_SECONDS:
            wait_seconds = int(RESEND_COOLDOWN_SECONDS - elapsed)
            flash(f"Please wait {wait_seconds} second{'s' if wait_seconds != 1 else ''} before requesting another code.", "warning")
            return redirect(url_for("auth.verify_email", email=user.email))

    try:
        send_verification_otp(user)
        flash("A new verification code has been sent to your email address.", "success")
    except EmailConfigurationError as error:
        current_app.logger.error("Could not send verification email: %s", error)
        flash(f"Email delivery is not configured: {error}", "error")
    except (OSError, smtplib.SMTPException) as error:
        current_app.logger.error("Could not send verification email: %s", error)
        flash("Verification email could not be sent. Check SMTP settings and try again.", "error")

    return redirect(url_for("auth.verify_email", email=user.email))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard" if current_user.role == "admin" else "main.dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = User.query.filter_by(email=email).first()
        now = utc_now()
        valid = bool(
            user
            and user.is_active
            and not _locked(user, now)
            and check_password_hash(user.password_hash, form.password.data)
        )
        if valid:
            if not user.email_verified:
                _log_attempt(user, email, False)
                db.session.commit()
                session["verify_email"] = user.email
                flash("Please verify your email address before signing in. You can enter your verification code below or request a new one.", "warning")
                return redirect(url_for("auth.verify_email", email=user.email))

            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = now
            _log_attempt(user, email, True)
            db.session.commit()
            login_user(user, remember=form.remember_me.data)
            return redirect(url_for("admin.dashboard" if user.role == "admin" else "main.dashboard"))

        if user and user.is_active and not _locked(user, now):
            user.failed_attempts += 1
            if user.failed_attempts >= LOCKOUT_ATTEMPTS:
                user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
        _log_attempt(user, email, False)
        db.session.commit()
        form.password.errors.append("Invalid email or password.")
    elif request.method == "POST":
        email = request.form.get("email", "")[:254]
        user = User.query.filter_by(email=email.strip().lower()).first() if email else None
        _log_attempt(user, email, False)
        db.session.commit()
    return render_template("login.html", form=form)


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for("auth.login"))
