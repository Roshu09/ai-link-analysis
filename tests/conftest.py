import os

# Must be set before the app is imported, so settings pick them up.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"

import pytest

from app.utils import validators


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """No real DNS in tests. Every hostname is 'public' unless we say otherwise."""
    private_hosts = {"intranet.example.com": ["10.0.0.5"], "sneaky.example.com": ["127.0.0.1"]}

    async def resolve(host: str, port: int) -> list[str]:
        return private_hosts.get(host, ["93.184.216.34"])

    monkeypatch.setattr(validators, "_resolve", resolve)
