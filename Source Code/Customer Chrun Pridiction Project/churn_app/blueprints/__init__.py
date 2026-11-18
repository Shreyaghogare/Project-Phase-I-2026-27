from .admin import bp as admin_bp
from .auth import bp as auth_bp
from .main import bp as main_bp


def register_blueprints(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)
