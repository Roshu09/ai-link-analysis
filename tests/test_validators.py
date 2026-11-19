import asyncio

import pytest

from app.exceptions import BlockedURLError, InvalidURLError
from app.utils.validators import assert_public_url, is_public_ip, normalize_url


class TestNormalizeUrl:
    def test_adds_https_and_cleans(self):
        assert normalize_url("  Example.COM/Path#frag ") == "https://example.com/Path"

    def test_keeps_query_and_adds_root_path(self):
        assert normalize_url("https://example.com?a=1") == "https://example.com/?a=1"

    def test_unicode_domain_is_converted(self):
        assert normalize_url("https://bücher.example/").startswith("https://xn--")

    @pytest.mark.parametrize(
        "bad",
        ["", "   ", "ftp://example.com", "http://user:pass@example.com", "not a url",
         "http://example", "http://2130706433", "javascript:alert(1)", "http://" + "a" * 3000 + ".com"],
    )
    def test_rejects_invalid(self, bad):
        with pytest.raises((InvalidURLError, BlockedURLError)):
            normalize_url(bad)

    @pytest.mark.parametrize(
        "internal",
        [
            "http://localhost:4000",
            "http://127.0.0.1/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5",
            "http://192.168.1.1",
            "http://172.16.0.1",
            "http://100.64.0.1",
            "http://0.0.0.0",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[fe80::1]/",
            "http://metadata.google.internal/",
            "http://printer.local/",
            "http://example.com:22/",
        ],
    )
    def test_blocks_internal_targets(self, internal):
        with pytest.raises(BlockedURLError):
            normalize_url(internal)

    def test_public_ip_literal_allowed(self):
        assert normalize_url("http://93.184.216.34/") == "http://93.184.216.34/"


class TestIsPublicIp:
    @pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
    def test_public(self, ip):
        assert is_public_ip(ip)

    @pytest.mark.parametrize(
        "ip", ["127.0.0.1", "10.1.2.3", "192.168.0.1", "169.254.1.1", "::1", "fc00::1", "::ffff:10.0.0.1", "224.0.0.1"]
    )
    def test_not_public(self, ip):
        assert not is_public_ip(ip)


class TestAssertPublicUrl:
    def test_public_host_passes(self):
        asyncio.run(assert_public_url("https://example.com/"))

    def test_host_resolving_to_private_ip_is_blocked(self):
        with pytest.raises(BlockedURLError):
            asyncio.run(assert_public_url("https://intranet.example.com/"))

    def test_host_resolving_to_loopback_is_blocked(self):
        with pytest.raises(BlockedURLError):
            asyncio.run(assert_public_url("https://sneaky.example.com/"))
