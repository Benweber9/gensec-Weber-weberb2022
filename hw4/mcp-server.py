from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("cybersecurity-server")


@mcp.tool()
def password_strength_check(password: str) -> dict:
    """Evaluate a password's strength based on length and complexity.

    The password is analyzed only in memory for the current request. It is never
    stored, logged, or persisted to disk.
    """
    if not password:
        return {
            "valid": False,
            "score": 0,
            "strength": "empty",
            "length": 0,
            "checks": {
                "length": False,
                "uppercase": False,
                "lowercase": False,
                "number": False,
                "symbol": False,
            },
            "issues": ["Provide a password to evaluate."],
            "note": "This check only inspects the password in memory and does not store it.",
        }

    length = len(password)
    has_upper = any(char.isupper() for char in password)
    has_lower = any(char.islower() for char in password)
    has_number = any(char.isdigit() for char in password)
    has_symbol = any(not char.isalnum() for char in password)

    score = 0
    issues = []

    if length >= 12:
        score += 2
    elif length >= 8:
        score += 1
    else:
        issues.append("Use at least 12 characters when possible.")

    if has_upper:
        score += 1
    else:
        issues.append("Add at least one uppercase letter.")

    if has_lower:
        score += 1
    else:
        issues.append("Add at least one lowercase letter.")

    if has_number:
        score += 1
    else:
        issues.append("Add at least one number.")

    if has_symbol:
        score += 1
    else:
        issues.append("Add at least one special character.")

    if score >= 6:
        strength = "strong"
    elif score >= 4:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "valid": True,
        "score": score,
        "strength": strength,
        "length": length,
        "checks": {
            "length": length >= 8,
            "uppercase": has_upper,
            "lowercase": has_lower,
            "number": has_number,
            "symbol": has_symbol,
        },
        "issues": issues,
        "note": "This check only inspects the password in memory and does not store it.",
    }


@mcp.tool()
def phishing_checklist() -> list[str]:
    """Return a short list of tips to help identify phishing emails."""
    return [
        "Check the sender address carefully for misspellings or unusual domains.",
        "Look for urgent language such as 'act now,' 'immediate action required,' or threats.",
        "Do not click links in unexpected emails; instead, open the official website directly.",
        "Inspect attachments before opening them, especially from unknown senders.",
        "Verify requests by contacting the company or person through a trusted channel.",
        "Be cautious of generic greetings, poor grammar, and mismatched branding.",
    ]


@mcp.tool()
def security_checklist(topic: str) -> dict:
    """Provide a customized cybersecurity checklist for a given topic.

    The topic is used only to tailor recommendations for the current request.
    """
    topic_name = (topic or "general").strip().lower()

    topic_map = {
        "passwords": [
            "Use a unique password for each account.",
            "Create passwords with at least 12 characters and a mix of letters, numbers, and symbols.",
            "Use a password manager to store and generate strong credentials securely.",
            "Enable multifactor authentication whenever possible.",
        ],
        "phishing": [
            "Pause before clicking links in unexpected emails or texts.",
            "Verify the sender using a trusted communication channel.",
            "Report suspicious messages to your security team or IT support.",
            "Train yourself and your team to spot urgency, impersonation, and odd domains.",
        ],
        "devices": [
            "Keep your operating system and apps updated.",
            "Use a screen lock and device encryption when available.",
            "Only install software from trusted sources.",
            "Back up important files regularly.",
        ],
        "remote-work": [
            "Use a VPN when working on public or untrusted networks.",
            "Avoid sharing login credentials over chat or email.",
            "Use company-approved collaboration tools and access controls.",
            "Lock your workstation whenever you step away.",
        ],
    }

    recommendations = topic_map.get(
        topic_name,
        [
            "Keep software and devices updated.",
            "Use strong, unique passwords and enable multifactor authentication.",
            "Be careful with email links, attachments, and unexpected requests.",
            "Back up important information and report suspicious activities promptly.",
        ],
    )

    return {
        "topic": topic_name if topic_name else "general",
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
