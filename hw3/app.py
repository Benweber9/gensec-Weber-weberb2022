"""Run a command-line Security Assistant with LangChain and Python calculations.

Install dependencies with:
    uv add langchain langchain-openai langchain-experimental

Set OPENAI_API_KEY and OPENAI_MODEL in the environment, then run uv run python app.py.
OPENAI_BASE_URL may optionally point to an OpenAI-compatible endpoint. Choose
a model that supports tool calling. Enter 'exit' or 'quit' to end the session.

The PythonREPL tool executes Python locally with the process's permissions;
use this assistant only with trusted inputs.
"""

import os
import sys

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI


@tool
def security_checklist(topic: str) -> str:
    """Return a short defensive security checklist for the requested topic.

    Args:
        topic: Security topic: passwords, phishing, network, or web application.

    Returns:
        An educational checklist, or supported topics if the topic is unknown.
    """
    checklists = {
        "passwords": [
            "Use a long, unique password for each account.",
            "Store passwords in a password manager.",
            "Enable multifactor authentication where available.",
            "Change compromised passwords and review account activity.",
        ],
        "phishing": [
            "Verify unexpected requests through a separate, trusted channel.",
            "Check sender addresses and link destinations carefully.",
            "Avoid opening unexpected attachments or sharing login codes.",
            "Report suspicious messages to your security team or provider.",
        ],
        "network": [
            "Keep routers, firewalls, and connected devices updated.",
            "Replace default administrator credentials.",
            "Use WPA2 or WPA3 for Wi-Fi and restrict unnecessary access.",
            "Separate guest devices and review logs for unusual activity.",
        ],
        "web application": [
            "Validate input and use parameterized database queries.",
            "Enforce authentication and authorization on the server.",
            "Use HTTPS and secure session cookies.",
            "Update dependencies and keep secrets out of source code.",
        ],
    }
    normalized_topic = " ".join(topic.lower().split())
    checklist = checklists.get(normalized_topic)
    if checklist is None:
        return "Supported topics: " + ", ".join(checklists) + "."
    return f"Defensive checklist for {normalized_topic}:\n" + "\n".join(
        f"- {item}" for item in checklist
    )


def build_agent():
    """Create the security agent using environment settings and security tools.

    Raise ValueError if the API key or model name is missing.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_name = os.environ.get("OPENAI_MODEL", "").strip()
    if not api_key or not model_name:
        raise ValueError("Set OPENAI_API_KEY and OPENAI_MODEL before running.")

    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    # Add custom tools to this list in later homework tasks.
    tools = [PythonREPLTool(), security_checklist]
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a Security Assistant. Explain security concepts clearly "
            "and help users reason about defensive security tasks. "
            "Use security_checklist for short defensive topic checklists. "
            "Use the Python_REPL tool for Python calculations when useful. "
            "Print calculation results so you can read the tool output. "
            "Explain your results and acknowledge uncertainty."
        ),
    )


def main() -> int:
    """Run an interactive conversation and return a process exit status."""
    try:
        agent = build_agent()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    messages = []
    print("Security Assistant — enter 'exit' or 'quit' to leave.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            result = agent.invoke(
                {"messages": messages + [{"role": "user", "content": question}]}
            )
        except KeyboardInterrupt:
            print()
            break
        except Exception as error:
            print(f"Agent error: {error}", file=sys.stderr)
            continue
        messages = result["messages"]
        print(f"Assistant: {messages[-1].content}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
