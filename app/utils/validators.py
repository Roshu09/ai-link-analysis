"""URL validation and SSRF protection.

SSRF = Server-Side Request Forgery: tricking our server into fetching something
it shouldn't, like http://localhost, a cloud metadata endpoint, or an office
intranet page. We defend in layers:

1. `normalize_url` cleans the input and rejects obviously bad / internal URLs.
2. `assert_public_url` resolves the hostname and rejects it if ANY resolved IP
   is private, loopback, link-local, reserved, etc.
3. The website extractor calls `assert_public_url` again on every redirect hop,
   so a public site can't bounce us to an internal address.

Known limit: DNS can change between our check and the actual connection
("DNS rebinding"). Fully closing that needs IP pinning at the socket level;
see the README's limitations section.
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit

from app.exceptions import BlockedURLError, InvalidURLError, WebsiteUnreachableError

MAX_URL_LENGTH = 2048
ALLOWED_PORTS = {None, 80, 443, 8080, 8443}
BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",
}
BLOCKED_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet")


def is_public_ip(value: str) -> bool:
    """True only for globally routable addresses."""
    ip = ipaddress.ip_address(value.split("%")[0])  # drop IPv6 zone ids like fe80::1%eth0
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 must be judged as 127.0.0.1
    return ip.is_global and not ip.is_multicast


def _parse_ip(host: str) -> str | None:
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        return None


def normalize_url(raw: str) -> str:
    """Turn user input into a clean absolute http(s) URL, or raise a friendly error."""
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidURLError("Please provide a website URL, e.g. https://example.com")

    value = raw.strip()
    if len(value) > MAX_URL_LENGTH:
        raise InvalidURLError("That URL is too long.")
    if any(ch.isspace() for ch in value):
        raise InvalidURLError("A URL can't contain spaces.")
    if "://" not in value:
        value = "https://" + value  # be forgiving: "example.com" -> "https://example.com"

    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise InvalidURLError() from None

    if parts.scheme.lower() not in ("http", "https"):
        raise InvalidURLError("Only http:// and https:// URLs are supported.")
    if parts.username or parts.password:
        raise InvalidURLError("URLs with embedded usernames or passwords are not allowed.")
    if not parts.hostname:
        raise InvalidURLError()

    host = parts.hostname.rstrip(".").lower()
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES):
        raise BlockedURLError()

    ip_literal = _parse_ip(host)
    if ip_literal is not None:
        if not is_public_ip(ip_literal):
            raise BlockedURLError()
    else:
        try:
            host = host.encode("idna").decode("ascii")  # support unicode domain names
        except UnicodeError:
            raise InvalidURLError() from None
        if "." not in host:
            raise InvalidURLError("That doesn't look like a full domain name (e.g. example.com).")

    if port not in ALLOWED_PORTS:
        raise BlockedURLError("Only standard web ports (80, 443, 8080, 8443) are allowed.")

    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


async def _resolve(host: str, port: int) -> list[str]:
    """Resolve a hostname to its IP addresses (separate function so tests can fake DNS)."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise WebsiteUnreachableError(
            "We couldn't find that domain. Please check the spelling of the URL."
        ) from None
    return sorted({info[4][0] for info in infos})


async def assert_public_url(url: str) -> None:
    """Raise BlockedURLError if the URL would connect to a non-public address."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = parts.port or (443 if parts.scheme == "https" else 80)

    if _parse_ip(host):
        addresses = [host]
    else:
        addresses = await _resolve(host, port)

    if not addresses or not all(is_public_ip(addr) for addr in addresses):
        raise BlockedURLError()
