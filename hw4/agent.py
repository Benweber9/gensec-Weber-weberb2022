from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport, StdioTransport


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "sample_reports"
CUSTOM_SERVER = BASE_DIR / "mcp-server.py"
REPORT_PATH = REPORTS_DIR / "cyber_incident_report.txt"


def load_environment() -> None:
    """Load environment variables from a local .env file if present."""
    load_dotenv(BASE_DIR / ".env")


def determine_topic(report_text: str) -> str:
    """Infer a relevant cybersecurity topic from the report content."""
    lowered = report_text.lower()
    if "phishing" in lowered or "email" in lowered:
        return "phishing"
    if "password" in lowered:
        return "passwords"
    if "device" in lowered or "laptop" in lowered or "mobile" in lowered:
        return "devices"
    if "remote" in lowered or "vpn" in lowered:
        return "remote-work"
    return "general"


async def main() -> None:
    """Read a fictional report, generate tailored security recommendations, and display the combined result."""
    load_environment()

    if not REPORTS_DIR.exists():
        raise FileNotFoundError(f"Sample reports directory not found: {REPORTS_DIR}")

    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Sample report not found: {REPORT_PATH}")

    # Keep any credentials in environment variables instead of hardcoding them.
    api_key = os.getenv("API_KEY")
    if api_key:
        print(f"API_KEY loaded from environment: {api_key[:4]}********")
    else:
        print("No API_KEY configured; this demo uses no external API calls.")

    custom_transport = PythonStdioTransport(str(CUSTOM_SERVER), cwd=str(BASE_DIR))
    filesystem_transport = StdioTransport(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(REPORTS_DIR)],
        env=os.environ.copy(),
        cwd=str(BASE_DIR),
    )

    async with Client(custom_transport) as custom_client, Client(filesystem_transport) as filesystem_client:
        report_result = await filesystem_client.call_tool(
            "read_text_file",
            {"path": str(REPORT_PATH)},
        )
        report_text = report_result.structured_content["content"]
        topic = determine_topic(report_text)

        recommendations_result = await custom_client.call_tool(
            "security_checklist",
            {"topic": topic},
        )

        combined = {
            "source_file": str(REPORT_PATH),
            "topic": topic,
            "report_summary": report_text,
            "recommendations": recommendations_result.structured_content["recommendations"],
        }

        print("Combined MCP analysis result:")
        print(combined)


if __name__ == "__main__":
    asyncio.run(main())
