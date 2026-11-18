from calendar import monthrange
from datetime import datetime, time, timedelta

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import case, func, or_
from sqlalchemy.orm import joinedload

from ...extensions import db
from ...models import AdminActivityLog, Company, LoginLog, Prediction, User, UserCompanyAccess, utc_now
from ...security import admin_required
from ..main.routes import evaluation_data, model
from . import bp


def log_activity(action, target):
    db.session.add(AdminActivityLog(
        admin_id=current_user.id,
        action=action,
        target=str(target)[:255],
        timestamp=utc_now(),
    ))


def _month_keys(end, count):
    keys = []
    year, month = end.year, end.month
    for _ in range(count):
        keys.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(keys))


@bp.route("/dashboard")
@admin_required
def dashboard():
    now = utc_now().replace(tzinfo=None)
    today_start = datetime.combine(now.date(), time.min)
    tomorrow = today_start + timedelta(days=1)
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_predictions = Prediction.query.count()
    churned = Prediction.query.filter_by(prediction="Yes").count()
    non_churned = Prediction.query.filter_by(prediction="No").count()
    high_risk = Prediction.query.filter_by(risk_level="High Risk").count()
    failed_today = LoginLog.query.filter(
        LoginLog.success.is_(False), LoginLog.timestamp >= today_start, LoginLog.timestamp < tomorrow
    ).count()

    day_keys = [(now.date() - timedelta(days=offset)).isoformat() for offset in range(13, -1, -1)]
    daily_rows = db.session.query(func.date(Prediction.created_at), func.count(Prediction.id)).filter(
        Prediction.created_at >= now - timedelta(days=13)
    ).group_by(func.date(Prediction.created_at)).all()
    daily_counts = {str(day): count for day, count in daily_rows}

    month_keys = _month_keys(now, 12)
    monthly_rows = db.session.query(func.strftime("%Y-%m", User.created_at), func.count(User.id)).group_by(
        func.strftime("%Y-%m", User.created_at)
    ).all()
    monthly_counts = {str(month): count for month, count in monthly_rows}

    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        active_users=active_users,
        total_predictions=total_predictions,
        churned=churned,
        non_churned=non_churned,
        high_risk=high_risk,
        failed_today=failed_today,
        churn_chart=[churned, non_churned],
        risk_chart=[
            Prediction.query.filter_by(risk_level=level).count()
            for level in ("Low Risk", "Medium Risk", "High Risk")
        ],
        day_labels=day_keys,
        day_values=[daily_counts.get(day, 0) for day in day_keys],
        month_labels=month_keys,
        month_values=[monthly_counts.get(month, 0) for month in month_keys],
        login_chart=[
            LoginLog.query.filter_by(success=True).count(),
            LoginLog.query.filter_by(success=False).count(),
        ],
    )


@bp.route("/companies")
@admin_required
def companies():
    companies_list = Company.query.order_by(Company.name.asc()).all()
    company_rows = []
    for company in companies_list:
        company_rows.append({
            "company": company,
            "dataset_count": len(company.datasets),
            "model_count": len(company.trained_models),
            "access_count": len(company.access_rows),
        })
    return render_template("admin/companies.html", companies=company_rows)


@bp.route("/company-comparison")
@admin_required
def company_comparison():
    companies_list = Company.query.order_by(Company.name.asc()).all()
    prediction_rows = db.session.query(
        Prediction.company_id,
        func.count(Prediction.id),
        func.sum(case((Prediction.prediction == "Yes", 1), else_=0)),
        func.avg(Prediction.probability),
    ).filter(Prediction.company_id.isnot(None)).group_by(Prediction.company_id).all()
    prediction_stats = {
        company_id: {"total": total, "churned": churned or 0, "average_probability": average_probability}
        for company_id, total, churned, average_probability in prediction_rows
    }

    comparison_rows = []
    for company in companies_list:
        datasets = [item for item in company.datasets if item.is_active]
        models = [item for item in company.trained_models if item.is_active]
        comparison_rows.append({
            "company": company,
            "dataset": max(datasets, key=lambda item: item.uploaded_at, default=None),
            "model": max(models, key=lambda item: item.version, default=None),
            "predictions": prediction_stats.get(company.id, {"total": 0, "churned": 0, "average_probability": None}),
        })
    return render_template("admin/company_comparison.html", companies=comparison_rows)


@bp.route("/users")
@admin_required
def users():
    query = User.query
    search = request.args.get("q", "").strip()
    role = request.args.get("role", "")
    status = request.args.get("status", "")
    if search:
        pattern = f"%{search}%"
        query = query.filter(or_(User.name.ilike(pattern), User.email.ilike(pattern)))
    if role in ("admin", "user"):
        query = query.filter(User.role == role)
    if status == "active":
        query = query.filter(User.is_active.is_(True))
    elif status == "inactive":
        query = query.filter(User.is_active.is_(False))
    pagination = query.order_by(User.created_at.desc()).paginate(
        page=request.args.get("page", 1, type=int), per_page=20, error_out=False
    )
    user_ids = [user.id for user in pagination.items]
    counts = dict(db.session.query(Prediction.user_id, func.count(Prediction.id)).filter(
        Prediction.user_id.in_(user_ids)
    ).group_by(Prediction.user_id).all()) if user_ids else {}
    access_counts = dict(db.session.query(
        UserCompanyAccess.user_id, func.count(UserCompanyAccess.company_id)
    ).join(Company, Company.id == UserCompanyAccess.company_id).filter(
        UserCompanyAccess.user_id.in_(user_ids), Company.is_active.is_(True)
    ).group_by(UserCompanyAccess.user_id).all()) if user_ids else {}
    return render_template(
        "admin/users.html",
        pagination=pagination,
        users=pagination.items,
        prediction_counts=counts,
        access_counts=access_counts,
        filters={"q": search, "role": role, "status": status},
    )


@bp.route("/users/<int:user_id>/access", methods=["GET", "POST"])
@admin_required
def update_company_access(user_id):
    user = db.get_or_404(User, user_id)
    companies = Company.query.filter_by(is_active=True).order_by(Company.name.asc()).all()
    company_ids = {company.id for company in companies}

    if user.role == "admin":
        flash("Administrators automatically have access to every active company.", "warning")
        return redirect(url_for("admin.users"))

    selected_ids = {
        access.company_id
        for access in UserCompanyAccess.query.filter_by(user_id=user.id).all()
        if access.company_id in company_ids
    }
    if request.method == "POST":
        try:
            requested_ids = {int(value) for value in request.form.getlist("company_ids")}
        except ValueError:
            flash("Invalid company selection. Choose from the active companies listed.", "error")
            return render_template(
                "admin/user_access.html",
                user=user,
                companies=companies,
                selected_ids=selected_ids,
            )
        else:
            if not requested_ids.issubset(company_ids):
                flash("Invalid company selection. Choose from the active companies listed.", "error")
                return render_template(
                    "admin/user_access.html",
                    user=user,
                    companies=companies,
                    selected_ids=selected_ids,
                )

            existing = UserCompanyAccess.query.filter_by(user_id=user.id).all()
            existing_by_company = {access.company_id: access for access in existing}
            changed = False
            for company_id, access in existing_by_company.items():
                if company_id not in requested_ids:
                    db.session.delete(access)
                    log_activity("revoked company access", f"{user.email} -> {access.company.name}")
                    changed = True
            for company in companies:
                if company.id in requested_ids and company.id not in existing_by_company:
                    db.session.add(UserCompanyAccess(user_id=user.id, company_id=company.id))
                    log_activity("granted company access", f"{user.email} -> {company.name}")
                    changed = True
            if changed:
                db.session.commit()
            flash(f"Company access updated for {user.name}.", "success")
            return redirect(url_for("admin.users"))

    return render_template(
        "admin/user_access.html",
        user=user,
        companies=companies,
        selected_ids=selected_ids,
    )


@bp.route("/users/<int:user_id>/role", methods=["POST"])
@admin_required
def update_role(user_id):
    user = db.get_or_404(User, user_id)
    role = request.form.get("role")
    if role not in ("admin", "user"):
        flash("Invalid role.", "error")
    elif user.id == current_user.id and role != "admin":
        log_activity("blocked self-demotion", user.email)
        db.session.commit()
        flash("You cannot demote your own account.", "error")
    elif user.role != role:
        old_role = user.role
        user.role = role
        log_activity("promoted user" if role == "admin" else "demoted user", user.email)
        db.session.commit()
        flash(f"{user.name} changed from {old_role} to {role}.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/users/<int:user_id>/status", methods=["POST"])
@admin_required
def update_status(user_id):
    user = db.get_or_404(User, user_id)
    active = request.form.get("active") == "true"
    if user.id == current_user.id and not active:
        log_activity("blocked self-deactivation", user.email)
        db.session.commit()
        flash("You cannot deactivate your own account.", "error")
    elif user.is_active != active:
        user.is_active = active
        log_activity("activated user" if active else "deactivated user", user.email)
        db.session.commit()
        flash(f"{user.name} is now {'active' if active else 'deactivated'}.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/users/<int:user_id>/reset-lock", methods=["POST"])
@admin_required
def reset_lock(user_id):
    user = db.get_or_404(User, user_id)
    user.failed_attempts = 0
    user.locked_until = None
    log_activity("reset login lock", user.email)
    db.session.commit()
    flash(f"Login lock reset for {user.email}.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete_user(user_id):
    user = db.get_or_404(User, user_id)
    if user.id == current_user.id:
        log_activity("blocked self-deletion", user.email)
        db.session.commit()
        flash("You cannot delete your own account.", "error")
    else:
        target = user.email
        db.session.delete(user)
        log_activity("deleted user", target)
        db.session.commit()
        flash("User deleted.", "success")
    return redirect(url_for("admin.users"))


@bp.route("/predictions")
@admin_required
def predictions():
    query = Prediction.query.join(Prediction.user)
    search = request.args.get("q", "").strip()
    outcome = request.args.get("prediction", "")
    risk = request.args.get("risk", "")
    if search:
        pattern = f"%{search}%"
        query = query.filter(or_(
            Prediction.gender.ilike(pattern),
            Prediction.contract_type.ilike(pattern),
            User.name.ilike(pattern),
            User.email.ilike(pattern),
        ))
    if outcome in ("Yes", "No"):
        query = query.filter(Prediction.prediction == outcome)
    if risk in ("Low Risk", "Medium Risk", "High Risk"):
        query = query.filter(Prediction.risk_level == risk)
    pagination = query.order_by(Prediction.created_at.desc()).paginate(
        page=request.args.get("page", 1, type=int), per_page=10, error_out=False
    )
    return render_template(
        "admin/predictions.html",
        predictions=pagination.items,
        pagination=pagination,
        filters={"q": search, "prediction": outcome, "risk": risk},
        is_admin=True,
    )


@bp.route("/login-logs")
@admin_required
def login_logs():
    query = LoginLog.query
    search = request.args.get("q", "").strip()
    result = request.args.get("result", "")
    if search:
        query = query.filter(LoginLog.email_entered.ilike(f"%{search}%"))
    if result == "success":
        query = query.filter(LoginLog.success.is_(True))
    elif result == "failure":
        query = query.filter(LoginLog.success.is_(False))
    pagination = query.order_by(LoginLog.timestamp.desc()).paginate(
        page=request.args.get("page", 1, type=int), per_page=25, error_out=False
    )
    return render_template("admin/login_logs.html", logs=pagination.items, pagination=pagination, filters={"q": search, "result": result})


@bp.route("/activity-logs")
@admin_required
def activity_logs():
    query = AdminActivityLog.query.options(joinedload(AdminActivityLog.admin))
    search = request.args.get("q", "").strip()
    if search:
        pattern = f"%{search}%"
        query = query.filter(or_(AdminActivityLog.action.ilike(pattern), AdminActivityLog.target.ilike(pattern)))
    pagination = query.order_by(AdminActivityLog.timestamp.desc()).paginate(
        page=request.args.get("page", 1, type=int), per_page=25, error_out=False
    )
    return render_template("admin/activity_logs.html", logs=pagination.items, pagination=pagination, search=search)


@bp.route("/model-info")
@admin_required
def model_info():
    metrics = {}
    matrix = [[0, 0], [0, 0]]
    if model is not None:
        all_results, matrix = evaluation_data()
        metrics = all_results.get("Existing Random Forest", {})
    importances = getattr(model, "feature_importances_", []) if model is not None else []
    feature_names = list(getattr(model, "feature_names_in_", [])) if model is not None else []
    importance_rows = sorted(zip(feature_names, [float(value) for value in importances]), key=lambda item: item[1], reverse=True)
    return render_template(
        "admin/model_info.html",
        metrics=metrics,
        matrix=matrix,
        feature_names=[name for name, _value in importance_rows],
        feature_values=[round(value * 100, 2) for _name, value in importance_rows],
        model=model,
    )