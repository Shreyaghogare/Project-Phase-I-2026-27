from __future__ import annotations


def get_retention_actions(industry: str, risk_level: str) -> list[str]:
    actions = {
        "telecom": {
            "Low Risk": ["Maintain proactive account outreach", "Keep digital self-service simple"],
            "Medium Risk": ["Offer plan review", "Schedule proactive support call"],
            "High Risk": ["Deliver retention offer", "Escalate to account manager", "Offer contract upgrade plan"],
        },
        "banking": {
            "Low Risk": ["Continue loyalty communications", "Review product usage"],
            "Medium Risk": ["Offer better financial guidance", "Encourage product diversification"],
            "High Risk": ["Offer branch follow-up", "Launch personalised product save campaign"],
        },
        "e-commerce": {
            "Low Risk": ["Continue personalised offers", "Reward repeat purchases"],
            "Medium Risk": ["Trigger recovery email", "Offer discount on next order"],
            "High Risk": ["Send win-back campaign", "Offer loyalty incentive"],
        },
        "saas": {
            "Low Risk": ["Maintain onboarding experience", "Encourage feature adoption"],
            "Medium Risk": ["Offer account health review", "Send feature tutorial"],
            "High Risk": ["Offer renewal discount", "Move to success team outreach"],
        },
        "insurance": {
            "Low Risk": ["Maintain renewal reminders", "Review customer satisfaction"],
            "Medium Risk": ["Follow up on claims experience", "Offer coverage review"],
            "High Risk": ["Offer retention consultation", "Coordinate with policy advisor"],
        },
    }

    key = industry.lower().replace(" ", "")
    industry_actions = actions.get(key, actions.get("telecom", {}))
    return industry_actions.get(risk_level, ["Review account health", "Offer a customer retention touchpoint"])
