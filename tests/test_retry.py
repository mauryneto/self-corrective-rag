import httpx
import pytest

from rag.retry import with_retries


def _status_error(code):
    req = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError("e", request=req, response=httpx.Response(code, request=req))


def test_retries_transient_500_then_succeeds():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _status_error(500)
        return "ok"

    assert with_retries(fn, base_delay_s=0) == "ok" and len(calls) == 3


def test_does_not_retry_client_errors():
    calls = []

    def fn():
        calls.append(1)
        raise _status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        with_retries(fn, base_delay_s=0)
    assert len(calls) == 1
