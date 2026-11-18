from __future__ import annotations

import collections
import html
import secrets
import smtplib
import time
from datetime import timedelta
from email.message import EmailMessage

from flask import current_app
from werkzeug.security import generate_password_hash

from ..extensions import db
from ..models import EmailOTP, User, utc_now

MAX_ATTACHMENT_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
_outgoing_email_timestamps: dict[int, list[float]] = collections.defaultdict(list)


class EmailConfigurationError(RuntimeError):
    pass


def check_outgoing_email_rate_limit(user_id: int, max_per_hour: int = 10) -> bool:
    """Check if the user has reached the outgoing email rate limit within the last hour."""
    now = time.time()
    cutoff = now - 3600
    timestamps = [t for t in _outgoing_email_timestamps[user_id] if t > cutoff]
    _outgoing_email_timestamps[user_id] = timestamps
    return len(timestamps) < max_per_hour


def record_outgoing_email_sent(user_id: int) -> None:
    """Record an outgoing email timestamp for rate limiting."""
    now = time.time()
    cutoff = now - 3600
    timestamps = [t for t in _outgoing_email_timestamps[user_id] if t > cutoff]
    timestamps.append(now)
    _outgoing_email_timestamps[user_id] = timestamps


def reset_outgoing_email_rate_limits() -> None:
    """Reset all in-memory outgoing email rate limits (used in tests)."""
    _outgoing_email_timestamps.clear()


def send_email(message: EmailMessage):
    host = current_app.config.get("SMTP_HOST")
    sender = current_app.config.get("SMTP_SENDER")
    username = current_app.config.get("SMTP_USERNAME")
    password = current_app.config.get("SMTP_PASSWORD")
    if not host or not sender:
        raise EmailConfigurationError("Set SMTP_HOST and SMTP_SENDER to enable email delivery.")
    if bool(username) != bool(password):
        raise EmailConfigurationError("Set both SMTP_USERNAME and SMTP_PASSWORD, or neither.")

    # Strip CR/LF from sender just in case
    clean_sender = sender.replace("\r", "").replace("\n", "").strip()
    message["From"] = clean_sender
    with smtplib.SMTP(host, current_app.config.get("SMTP_PORT", 587), timeout=15) as smtp:
        if current_app.config.get("SMTP_USE_TLS", True):
            smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(message)


def send_verification_otp(user: User) -> str:
    """Generate a random 6-digit verification code, save its hash to EmailOTP expiring in 10 minutes,
    and email it to the user.
    """
    code = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = utc_now() + timedelta(minutes=10)
    otp = EmailOTP(
        user_id=user.id,
        code_hash=generate_password_hash(code),
        purpose="verify_email",
        expires_at=expires_at,
        attempts=0,
    )
    db.session.add(otp)
    db.session.commit()

    if current_app.debug or current_app.testing:
        current_app.logger.info("Verification code for %s: %s", user.email, code)

    clean_email = (user.email or "").replace("\r", "").replace("\n", "").strip()
    clean_name = (user.name or "").replace("\r", "").replace("\n", "").strip()

    message = EmailMessage()
    message["Subject"] = "Verify your email address - Churn Compass"
    message["To"] = clean_email

    plain_content = (
        f"Hello {clean_name},\n\n"
        f"Your verification code is: {code}\n\n"
        "This code will expire in 10 minutes.\n"
        "If you did not request this, please ignore this email."
    )
    html_content = f"""\
<!DOCTYPE html>
<html>
  <body>
    <p>Hello {html.escape(clean_name)},</p>
    <p>Your verification code is: <strong style="font-size: 1.25em; letter-spacing: 2px;">{code}</strong></p>
    <p>This code will expire in 10 minutes.</p>
    <p>If you did not request this, please ignore this email.</p>
  </body>
</html>
"""
    message.set_content(plain_content)
    message.add_alternative(html_content, subtype="html")

    send_email(message)
    return code


def send_export_email(
    recipient_email: str,
    subject: str,
    message_text: str | None = None,
    csv_bytes: bytes | None = None,
    csv_filename: str = "churn-predictions.csv",
    xlsx_bytes: bytes | None = None,
    xlsx_filename: str = "churn-predictions.xlsx",
) -> None:
    """Send an export email with CSV and/or XLSX attachment(s) to recipient_email.
    Protects headers against CRLF injection and HTML body against XSS.
    Validates total attachment size against MAX_ATTACHMENT_SIZE_BYTES.
    """
    clean_recipient = (recipient_email or "").replace("\r", "").replace("\n", "").strip()
    clean_subject = (subject or "").replace("\r", "").replace("\n", "").strip()

    total_size = (len(csv_bytes) if csv_bytes else 0) + (len(xlsx_bytes) if xlsx_bytes else 0)
    if total_size > MAX_ATTACHMENT_SIZE_BYTES:
        raise ValueError("Attachment size exceeds the 5 MB limit. Please narrow your query to export fewer records.")

    message = EmailMessage()
    message["Subject"] = clean_subject
    message["To"] = clean_recipient

    user_note = (message_text or "").strip()
    plain_body = user_note if user_note else f"{clean_subject}."
    html_body = f"<p>{html.escape(user_note)}</p>" if user_note else f"<p>{html.escape(clean_subject)}.</p>"
    message.set_content(plain_body)
    message.add_alternative(html_body, subtype="html")

    if csv_bytes:
        message.add_attachment(
            csv_bytes,
            maintype="text",
            subtype="csv",
            filename=csv_filename,
        )
    if xlsx_bytes:
        message.add_attachment(
            xlsx_bytes,
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=xlsx_filename,
        )

    send_email(message)
