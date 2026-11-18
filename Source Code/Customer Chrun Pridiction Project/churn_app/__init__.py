from pathlib import Path

from flask import Flask, session
from flask_login import current_user

from config import Config
from .extensions import csrf, db, login_manager


def create_app(config_class=Config):
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder="../templates",
        static_folder="../static",
    )
    if isinstance(config_class, dict):
        app.config.update(config_class)
    else:
        app.config.from_object(config_class)
    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("Set the SECRET_KEY environment variable before starting the application.")
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "warning"

    @app.context_processor
    def inject_company_context():
        from .models import Company, UserCompanyAccess

        companies = []
        active_company = None
        if current_user.is_authenticated:
            if current_user.role == "admin":
                companies = Company.query.filter_by(is_active=True).order_by(Company.name.asc()).all()
            else:
                companies = (
                    Company.query.join(UserCompanyAccess, UserCompanyAccess.company_id == Company.id)
                    .filter(UserCompanyAccess.user_id == current_user.id, Company.is_active.is_(True))
                    .order_by(Company.name.asc())
                    .all()
                )
            if not companies:
                session.pop("active_company_id", None)
            else:
                try:
                    active_company_id = session.get("active_company_id")
                    active_company = db.session.get(Company, int(active_company_id)) if active_company_id else None
                except (TypeError, ValueError):
                    active_company = None
                if active_company is None or active_company not in companies:
                    active_company = companies[0]
                    session["active_company_id"] = str(active_company.id)
        return {
            "accessible_companies": companies,
            "active_company": active_company,
        }

    from .blueprints import register_blueprints

    register_blueprints(app)

    with app.app_context():
        from . import models

        db.create_all()

    @app.errorhandler(400)
    def bad_request(error):
        from flask import render_template

        description = getattr(error, "description", None)
        return render_template("errors/400.html", description=description), 400

    @app.errorhandler(403)
    def forbidden(_error):
        from flask import render_template

        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found(_error):
        from flask import render_template

        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_error(_error):
        from flask import render_template

        db.session.rollback()
        return render_template("errors/500.html"), 500

    return app
