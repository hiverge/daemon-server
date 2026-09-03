"""Tests for the daemon server's outbound SSE client."""

import asyncio
import json

import pytest

import main


class _FakeResponse:
  """Minimal async context manager standing in for an aiohttp response."""

  def __init__(self, status=200, body=None, lines=None):
    self.status = status
    self._body = body if body is not None else {}
    self._lines = lines or []

  async def __aenter__(self):
    return self

  async def __aexit__(self, *exc):
    return False

  def raise_for_status(self):
    if self.status >= 400:
      raise RuntimeError(f"HTTP {self.status}")

  async def json(self):
    return self._body

  @property
  def content(self):
    async def _gen():
      for line in self._lines:
        yield line

    return _gen()


class _FakeSession:
  """Records POST/GET calls and returns queued fake responses."""

  def __init__(self, post_responses=None, get_response=None):
    self._post_responses = list(post_responses or [])
    self._get_response = get_response
    self.post_calls = []
    self.get_calls = []

  def post(self, url, json=None, data=None, auth=None, headers=None):
    self.post_calls.append(
      {"url": url, "json": json, "data": data, "auth": auth, "headers": headers}
    )
    return self._post_responses.pop(0)

  def get(self, url, headers=None):
    self.get_calls.append({"url": url, "headers": headers})
    return self._get_response


class _FakeTokenProvider:
  """Returns preset tokens and records force_refresh requests."""

  def __init__(self, tokens):
    self._tokens = list(tokens)
    self.calls = []

  async def get(self, force_refresh):
    self.calls.append(force_refresh)
    return self._tokens.pop(0) if self._tokens else "token"


@pytest.fixture(name="_release_lock", autouse=True)
def _release_lock_fixture():
  """
  Ensures the module-level sandbox lock is released around every test, so a
  test that leaves it held cannot leak into the next.
  """
  # given a freshly released lock before each test
  if main.sandbox_lock.locked():
    main.sandbox_lock.release()
  yield
  if main.sandbox_lock.locked():
    main.sandbox_lock.release()


class TestRunCommandLocally:
  """
  Tests for the `run_command_locally` function.
  """

  def test_wraps_runner_result(self):
    """
    A known runner result is returned unchanged for its router.
    """
    # given a fake runner returning a known success dict
    async def fake_runner(payload):
      return {"status": "success", "result": {"fitness": 7}}

    runners = {"run_code": fake_runner}

    # when the command is run locally
    result = asyncio.run(
      main.run_command_locally("run_code", {"code": "x"}, runners)
    )

    # then the runner's result is returned verbatim
    assert result == {"status": "success", "result": {"fitness": 7}}

  def test_returns_busy_when_lock_held(self):
    """
    A held sandbox lock yields a busy response without invoking the runner.
    """
    # given the sandbox lock is already held
    main.sandbox_lock.acquire()
    invoked = False

    async def fake_runner(payload):
      nonlocal invoked
      invoked = True
      return {"status": "success"}

    # when a command is run locally
    result = asyncio.run(
      main.run_command_locally("run_code", {}, {"run_code": fake_runner})
    )

    # then it reports busy and never calls the runner
    assert result == {"status": "busy"}
    assert invoked is False

  def test_unknown_router_fails(self):
    """
    An unknown router name yields a clean failure.
    """
    # given an empty runner mapping
    # when an unknown router is run
    result = asyncio.run(main.run_command_locally("nope", {}, {}))

    # then it reports the unknown router
    assert result == {"status": "failed", "error": "Unknown router: nope"}

  def test_agent_router_stub_fails_cleanly(self):
    """
    The real agent runner returns a clean not-implemented failure.
    """
    # given the real runner mapping
    runners = main.build_runners()

    # when the agent router is run
    result = asyncio.run(main.run_command_locally("agent", {}, runners))

    # then it fails cleanly rather than hanging
    assert result == {"status": "failed", "error": "agent not implemented"}


class TestHandleCommand:
  """
  Tests for the `handle_command` function.
  """

  def test_posts_response_with_bearer_header(self):
    """
    The command's result is POSTed to /daemon/response with a Bearer header and
    the originating request_id.
    """
    # given a fake runner and a session that accepts the response POST
    async def fake_runner(payload):
      return {"status": "success", "result": {"fitness": 1.0}}

    session = _FakeSession(post_responses=[_FakeResponse(status=200)])
    token_provider = _FakeTokenProvider(tokens=["tok-abc"])
    cmd = {
      "type": "command",
      "request_id": "req-1",
      "router": "run_code",
      "payload": {"code": "x = 1"},
    }

    # when the command is handled
    asyncio.run(
      main.handle_command(
        session,
        "http://coord",
        token_provider,
        cmd,
        {"run_code": fake_runner},
      )
    )

    # then exactly one POST carried the request_id, result and Bearer token
    assert len(session.post_calls) == 1
    call = session.post_calls[0]
    assert call["url"] == "http://coord/daemon/response"
    assert call["json"] == {
      "request_id": "req-1",
      "response": {"status": "success", "result": {"fitness": 1.0}},
    }
    assert call["headers"] == {"Authorization": "Bearer tok-abc"}

  def test_refreshes_token_and_retries_on_401(self):
    """
    A 401 on the first POST triggers a token refresh and a second POST.
    """
    # given a session that 401s once then accepts
    async def fake_runner(payload):
      return {"status": "success"}

    session = _FakeSession(
      post_responses=[_FakeResponse(status=401), _FakeResponse(status=200)]
    )
    token_provider = _FakeTokenProvider(tokens=["stale", "fresh"])
    cmd = {"request_id": "req-2", "router": "run_code", "payload": {}}

    # when the command is handled
    asyncio.run(
      main.handle_command(
        session,
        "http://coord",
        token_provider,
        cmd,
        {"run_code": fake_runner},
      )
    )

    # then it POSTed twice, forcing a refresh on the retry
    assert len(session.post_calls) == 2
    assert token_provider.calls == [False, True]
    assert session.post_calls[0]["headers"] == {"Authorization": "Bearer stale"}
    assert session.post_calls[1]["headers"] == {"Authorization": "Bearer fresh"}


class TestSubscribeLoop:
  """
  Tests for the `subscribe_loop` function.
  """

  def test_parses_command_line_and_dispatches(self):
    """
    A single `data:` command line is parsed and dispatched, POSTing its result.
    """
    # given a stream that yields one command event then ends
    command = {
      "type": "command",
      "request_id": "req-3",
      "router": "run_code",
      "payload": {"code": "y = 2"},
    }
    lines = [
      b": keepalive\n",
      f"data: {json.dumps(command)}\n".encode("utf-8"),
    ]

    async def fake_runner(payload):
      return {"status": "success", "result": {"fitness": 2}}

    get_response = _FakeResponse(status=200, lines=lines)
    session = _FakeSession(
      post_responses=[_FakeResponse(status=200)], get_response=get_response
    )
    token_provider = _FakeTokenProvider(tokens=["tok"])

    # when the subscribe loop runs to stream end
    asyncio.run(
      main.subscribe_loop(
        session, "http://coord", token_provider, {"run_code": fake_runner}
      )
    )

    # then it subscribed once and POSTed the one command's result
    assert len(session.get_calls) == 1
    assert session.get_calls[0]["url"] == "http://coord/daemon/subscribe"
    assert len(session.post_calls) == 1
    assert session.post_calls[0]["json"]["request_id"] == "req-3"


class TestRunForever:
  """
  Tests for the `run_forever` reconnect loop.
  """

  def test_reconnects_after_drop(self):
    """
    A dropped connection is retried up to the reconnect bound.
    """
    # given a subscribe fn that always raises, and a no-op sleep
    attempts = {"count": 0}

    async def failing_subscribe():
      attempts["count"] += 1
      raise ConnectionError("dropped")

    sleeps = []

    async def fake_sleep(delay):
      sleeps.append(delay)

    # when run_forever runs with a bound of 3 reconnects
    asyncio.run(
      main.run_forever(failing_subscribe, fake_sleep, max_reconnects=3)
    )

    # then it retried exactly 3 times and slept between each
    assert attempts["count"] == 3
    assert len(sleeps) == 3


class TestGetToken:
  """
  Tests for the `get_token` function.
  """

  def test_sends_basic_auth_and_client_credentials(self):
    """
    The token request uses BasicAuth and the client_credentials grant, and
    returns the access token from the response body.
    """
    # given a session returning a token body
    response = _FakeResponse(
      status=200, body={"access_token": "the-token", "expires_in": 300}
    )
    session = _FakeSession(post_responses=[response])

    # when a token is fetched
    token = asyncio.run(
      main.get_token(
        session, "http://kc", "hiverge", "daemon-client", "secret"
      )
    )

    # then the token is returned and the request was well-formed
    assert token == "the-token"
    assert len(session.post_calls) == 1
    call = session.post_calls[0]
    assert (
      call["url"]
      == "http://kc/realms/hiverge/protocol/openid-connect/token"
    )
    assert call["data"] == {"grant_type": "client_credentials"}
    assert call["auth"].login == "daemon-client"
    assert call["auth"].password == "secret"


class TestBuildCoordinatorBaseUrl:
  """
  Tests for the `build_coordinator_base_url` function.
  """

  @pytest.mark.parametrize(
    "gateway_url,experiment_id,expected",
    [
      pytest.param(
        "https://gw.example.com",
        "exp-123",
        "https://gw.example.com/coordinator/exp-123",
        id="Simple experiment id",
      ),
      pytest.param(
        "https://gw.example.com/",
        "exp-123",
        "https://gw.example.com/coordinator/exp-123",
        id="Trailing slash on gateway url is trimmed",
      ),
    ],
  )
  def test_builds_experiment_scoped_url(
    self, gateway_url, experiment_id, expected
  ):
    """
    The base URL nests the experiment id under /coordinator on the gateway.
    """
    # given a gateway url and experiment id
    # when the base url is built
    result = main.build_coordinator_base_url(gateway_url, experiment_id)

    # then it is the experiment-scoped coordinator path
    assert result == expected

  @pytest.mark.parametrize(
    "experiment_id",
    [
      pytest.param("", id="Empty"),
      pytest.param("Exp_123", id="Uppercase and underscore"),
      pytest.param("exp/../other", id="Path traversal"),
      pytest.param("-leading-hyphen", id="Leading hyphen"),
    ],
  )
  def test_rejects_invalid_experiment_id(self, experiment_id):
    """
    An experiment id that is not a valid DNS label is rejected.
    """
    # given an invalid experiment id
    # when the base url is built, then it raises
    with pytest.raises(RuntimeError, match="Invalid EXPERIMENT_ID"):
      main.build_coordinator_base_url("https://gw", experiment_id)

  def test_composed_subscribe_and_response_urls(self):
    """
    subscribe_loop and handle_command target the experiment-scoped daemon
    endpoints when given the built base URL.
    """
    # given the experiment-scoped base url and a one-command stream
    base = main.build_coordinator_base_url("https://gw", "exp-9")
    command = {"type": "command", "request_id": "r1", "router": "run_code", "payload": {}}
    lines = [f"data: {json.dumps(command)}\n".encode("utf-8")]

    async def fake_runner(payload):
      return {"status": "success"}

    session = _FakeSession(
      post_responses=[_FakeResponse(status=200)],
      get_response=_FakeResponse(status=200, lines=lines),
    )
    token_provider = _FakeTokenProvider(tokens=["tok", "tok"])

    # when the subscribe loop runs
    asyncio.run(
      main.subscribe_loop(session, base, token_provider, {"run_code": fake_runner})
    )

    # then subscribe and response both hit the experiment-scoped path
    assert (
      session.get_calls[0]["url"]
      == "https://gw/coordinator/exp-9/daemon/subscribe"
    )
    assert (
      session.post_calls[0]["url"]
      == "https://gw/coordinator/exp-9/daemon/response"
    )
