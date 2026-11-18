import pytest

from churn_app import create_app
from churn_app.extensions import db
from churn_app.services.email_service import reset_outgoing_email_rate_limits


@pytest.fixture
def app():
    reset_outgoing_email_rate_limits()
    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "pytest-secret-key",
        "ADMIN_REGISTRATION_CODE": "test-admin-code",
        "SQLALCHEMY_DATABASE_URI": "sqlite://",
        "WTF_CSRF_ENABLED": False,
    })
    yield app
    reset_outgoing_email_rate_limits()
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()