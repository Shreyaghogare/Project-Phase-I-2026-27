from sqlalchemy import event
from sqlalchemy.engine import Engine

from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()


@event.listens_for(Engine, "connect")
def enable_sqlite_foreign_keys(connection, _record):
    if connection.__class__.__module__ == "sqlite3":
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@login_manager.user_loader
def load_user(user_id):
    from .models import User

    try:
        user = db.session.get(User, int(user_id))
        return user if user and user.is_active else None
    except (TypeError, ValueError):
        return None
