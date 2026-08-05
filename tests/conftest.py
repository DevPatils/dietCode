"""Suite-wide guards.

The loop tests drive a scripted fake client, so nothing here should ever reach
a provider. That was true by convention until a test called `main()`, which
loads .env, found the developer's real key, and spent tokens on a live request.
Convention is not a guard, so this makes it fail loudly instead.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any real HTTP request fails the test that made it.

    Only httpx is blocked -- that is what the OpenAI SDK uses. The Docker tests
    talk to the daemon over requests and a local socket, and must still work.
    """
    import httpx

    def blocked(self, request, *args, **kwargs):
        raise RuntimeError(
            f"test tried to reach {request.url} -- use the fake client, and check "
            "that .env is not being loaded"
        )

    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
