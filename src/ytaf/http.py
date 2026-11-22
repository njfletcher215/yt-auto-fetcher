"""The single HTTP seam.

RSS feeds, the Jellyfin API and ntfy all go through here, so the test suite
has exactly one thing to fake and the pipeline carries no `requests`
dependency onto the Pi.
"""

from __future__ import annotations

import json as _json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .errors import YtafError

DEFAULT_TIMEOUT = 30.0
USER_AGENT = "yt-auto-fetcher/0.1"


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    url: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return _json.loads(self.text)


def _unverified_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    verify_tls: bool = True,
) -> Response:
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    context = None if verify_tls else _unverified_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
            return Response(
                status=resp.status,
                body=resp.read(),
                headers=dict(resp.headers),
                url=resp.url,
            )
    except urllib.error.HTTPError as exc:
        return Response(
            status=exc.code,
            body=exc.read() or b"",
            headers=dict(exc.headers or {}),
            url=url,
        )
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise YtafError(f"request to {url} failed: {exc}") from exc


def get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    verify_tls: bool = True,
) -> Response:
    """GET `url`. Raises YtafError on transport failure, never on 4xx/5xx."""
    return _request("GET", url, headers=headers, timeout=timeout, verify_tls=verify_tls)


def post(
    url: str,
    *,
    body: bytes | str | None = None,
    json_body: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    verify_tls: bool = True,
) -> Response:
    """POST to `url`. `json_body` sets the content type and encodes for you."""
    req_headers = dict(headers or {})
    data: bytes
    if json_body is not None:
        data = _json.dumps(json_body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    elif isinstance(body, str):
        data = body.encode("utf-8")
    elif body is None:
        data = b""
    else:
        data = body
    return _request("POST", url, headers=req_headers, body=data, timeout=timeout, verify_tls=verify_tls)


def raise_for_status(response: Response) -> Response:
    """Return the response, or raise YtafError describing the status."""
    if not response.ok:
        raise YtafError(f"HTTP {response.status} from {response.url}")
    return response
