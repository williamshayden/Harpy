"""Bounded, one-attempt text transports for an explicitly configured local bridge.

Commands are trusted operator configuration, not model-provided actions. These
transports do not provide an OS sandbox or retry an uncertain generation.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import signal
import socket
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_REQUEST_BYTES = 262_144
DEFAULT_RESPONSE_BYTES = 1_048_576
_SHELLS = {
    "sh",
    "bash",
    "dash",
    "zsh",
    "ksh",
    "fish",
    "cmd",
    "powershell",
    "pwsh",
}
_SECRET_ARGUMENT = re.compile(
    r"^--?(?:api[-_]?key|token|authorization|password|secret|access[-_]?token)(?:=|$)",
    re.IGNORECASE,
)


class TransportError(RuntimeError):
    """A safe diagnostic that never includes response bodies, tokens, or stderr."""


def _limits(timeout_seconds: float, max_request_bytes: int, max_response_bytes: int) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 600
    ):
        raise TransportError("Transport timeout must be positive and at most 600 seconds.")
    for value in (max_request_bytes, max_response_bytes):
        if type(value) is not int or not 1 <= value <= 16_777_216:
            raise TransportError("Transport byte limits must be positive and at most 16 MiB.")


def _request_bytes(model: str, messages: list[dict], limit: int) -> bytes:
    if not isinstance(model, str) or not model or len(model) > 128:
        raise TransportError("A bounded configured model alias is required.")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise TransportError("Supply between one and 128 text messages.")
    dialogue = False
    copied = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "developer", "user", "assistant")
            or not isinstance(message["content"], str)
            or len(message["content"]) > 100_000
        ):
            raise TransportError("Only bounded plain-text role/content messages are supported.")
        if message["role"] in ("system", "developer"):
            if dialogue:
                raise TransportError("Instructions must precede dialogue.")
        else:
            dialogue = True
        copied.append({"role": message["role"], "content": message["content"]})
    if copied[-1]["role"] != "user" or not copied[-1]["content"].strip():
        raise TransportError("The last message must be a nonempty user message.")
    try:
        content = json.dumps(
            {"model": model, "messages": copied, "stream": False},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (UnicodeError, ValueError, TypeError):
        raise TransportError("The text request cannot be encoded as JSON.") from None
    if len(content) > limit:
        raise TransportError("The text request exceeds the configured byte limit.")
    return content


def _decode(content: bytes) -> dict:
    def reject_constant(_value):
        raise ValueError

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        result = json.loads(
            content.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object
        )
    except (ValueError, UnicodeError, RecursionError):
        raise TransportError("The local completion response is not valid bounded JSON.") from None
    if not isinstance(result, dict):
        raise TransportError("The local completion response must be an object.")
    return result


def _metadata(value: object, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise TransportError("The completion metadata is invalid.")
    return value


def _usage(value: object) -> dict | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or not value
        or set(value) - {"prompt_tokens", "completion_tokens", "total_tokens"}
        or any(type(count) is not int or not 0 <= count <= 2**53 - 1 for count in value.values())
    ):
        raise TransportError("The completion usage metadata is invalid.")
    return dict(value)


def _normalized(document: dict, *, chat_completion: bool) -> dict:
    if chat_completion:
        choices = document.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise TransportError("The local bridge must return exactly one completed text choice.")
        choice = choices[0]
        message = choice.get("message")
        if (
            choice.get("finish_reason") != "stop"
            or not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") != "assistant"
        ):
            raise TransportError("The local bridge did not return a completed text response.")
        content = message.get("content")
        response_id, response_model = document.get("id"), document.get("model")
    else:
        if set(document) - {"content", "usage", "response_id", "response_model"}:
            raise TransportError("The command returned an unsupported completion shape.")
        content = document.get("content")
        response_id, response_model = document.get("response_id"), document.get("response_model")
    if not isinstance(content, str):
        raise TransportError("The local completion content must be plain text.")
    return {
        "content": content,
        "usage": _usage(document.get("usage")),
        "response_id": _metadata(response_id, 256),
        "response_model": _metadata(response_model, 128),
    }


class HttpTransport:
    """Use literal HTTP loopback only, without proxies, redirects, or retries."""

    def __init__(
        self,
        base_url: str,
        *,
        token_file: Path | None = None,
        token_env: str | None = None,
        timeout_seconds: float = 120,
        max_request_bytes: int = DEFAULT_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_RESPONSE_BYTES,
    ) -> None:
        _limits(timeout_seconds, max_request_bytes, max_response_bytes)
        try:
            if (
                not isinstance(base_url, str)
                or len(base_url) > 2048
                or any(ord(character) <= 32 or ord(character) >= 127 for character in base_url)
                or "?" in base_url
                or "#" in base_url
            ):
                raise ValueError
            url = urlsplit(base_url)
            if (
                url.scheme != "http"
                or url.hostname not in {"127.0.0.1", "::1"}
                or url.username is not None
                or url.password is not None
                or url.path not in {"", "/", "/v1", "/v1/"}
                or (url.port is not None and not 1 <= url.port <= 65535)
            ):
                raise ValueError
            self.host, self.port = url.hostname, url.port or 80
        except (ValueError, TypeError):
            raise TransportError(
                "Use a literal HTTP loopback base URL with optional /v1."
            ) from None
        if (token_file is None) == (token_env is None):
            raise TransportError(
                "Select exactly one explicit local token file or environment name."
            )
        if token_env is not None and (
            not isinstance(token_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env) is None
        ):
            raise TransportError("The local token environment name is invalid.")
        self.token_file = None if token_file is None else Path(token_file)
        self.token_env = token_env
        self.timeout_seconds = timeout_seconds
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes

    def _token(self) -> str:
        try:
            if self.token_file is not None:
                if self.token_file.is_symlink() or not self.token_file.is_file():
                    raise ValueError
                with self.token_file.open("rb") as stream:
                    raw = stream.read(1025)
                if len(raw) > 1024:
                    raise ValueError
                token = raw.decode("ascii").strip()
            else:
                token = os.environ.get(self.token_env, "")
            if not token or len(token) > 1024 or any(not 33 <= ord(char) <= 126 for char in token):
                raise ValueError
            return token
        except (OSError, ValueError, UnicodeError):
            raise TransportError("Local bridge authentication is unavailable or invalid.") from None

    def complete(self, *, model: str, messages: list[dict]) -> dict:
        body = _request_bytes(model, messages, self.max_request_bytes)
        token = self._token()
        deadline = time.monotonic() + self.timeout_seconds
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_seconds)
        peer = None
        timer = None
        expired = threading.Event()

        def expire():
            expired.set()
            if peer is not None:
                with suppress(OSError):
                    peer.shutdown(socket.SHUT_RDWR)
                peer.close()

        try:
            connection.connect()
            peer = connection.sock
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportError("The local bridge request timed out; it was not retried.")
            peer.settimeout(remaining)
            timer = threading.Timer(remaining, expire)
            timer.daemon = True
            timer.start()
            host = f"[{self.host}]" if self.host == "::1" else self.host
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Host": f"{host}:{self.port}",
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                raise TransportError(
                    f"Local bridge HTTP request failed (status {response.status})."
                )
            if response.getheader("Content-Encoding", "identity") != "identity":
                raise TransportError("Compressed local bridge responses are unsupported.")
            length = response.getheader("Content-Length")
            if length is not None and (
                not length.isdigit() or int(length) > self.max_response_bytes
            ):
                raise TransportError("The local bridge response exceeds the configured byte limit.")
            content = response.read(self.max_response_bytes + 1)
            if expired.is_set() or time.monotonic() >= deadline:
                raise TransportError("The local bridge request timed out; it was not retried.")
            if len(content) > self.max_response_bytes:
                raise TransportError("The local bridge response exceeds the configured byte limit.")
            return _normalized(_decode(content), chat_completion=True)
        except TransportError:
            raise
        except (OSError, http.client.HTTPException):
            if expired.is_set() or time.monotonic() >= deadline:
                raise TransportError(
                    "The local bridge request timed out; it was not retried."
                ) from None
            raise TransportError(
                "The local bridge connection failed; it was not retried."
            ) from None
        finally:
            if timer is not None:
                timer.cancel()
            connection.close()
            if peer is not None:
                peer.close()


class CommandTransport:
    """Exchange bounded JSON with one trusted native command, never through a shell."""

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        timeout_seconds: float = 120,
        max_request_bytes: int = DEFAULT_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_RESPONSE_BYTES,
    ) -> None:
        _limits(timeout_seconds, max_request_bytes, max_response_bytes)
        if (
            not isinstance(command, tuple)
            or not 1 <= len(command) <= 64
            or any(
                not isinstance(arg, str) or not arg or len(arg) > 4096 or "\0" in arg
                for arg in command
            )
        ):
            raise TransportError("Supply a bounded native executable argument tuple.")
        executable = command[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
        if (
            executable.removesuffix(".exe") in _SHELLS
            or executable.endswith((".cmd", ".bat", ".ps1"))
            or any(_SECRET_ARGUMENT.match(arg) for arg in command[1:])
        ):
            raise TransportError("Shell commands and direct secret arguments are unsupported.")
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes

    def complete(self, *, model: str, messages: list[dict]) -> dict:
        body = _request_bytes(model, messages, self.max_request_bytes)
        environment = dict(os.environ)
        environment["HARPY_LLM_TRANSPORT_TIMEOUT_MS"] = str(
            max(1, int(self.timeout_seconds * 1000))
        )
        environment["HARPY_LLM_MAX_REQUEST_BYTES"] = str(self.max_request_bytes)
        environment["HARPY_LLM_MAX_RESPONSE_BYTES"] = str(self.max_response_bytes)
        deadline = time.monotonic() + self.timeout_seconds
        options = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=False,
                env=environment,
                **options,
            )
        except (OSError, ValueError):
            raise TransportError("The local completion command could not start.") from None
        output = bytearray()
        overflow = threading.Event()
        io_failed = threading.Event()

        def stop():
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
            except (OSError, ProcessLookupError):
                pass

        def write_input():
            try:
                process.stdin.write(body)
                process.stdin.close()
            except (OSError, ValueError):
                io_failed.set()

        def read_output():
            try:
                while True:
                    chunk = process.stdout.read(
                        min(65536, self.max_response_bytes + 1 - len(output))
                    )
                    if not chunk:
                        return
                    output.extend(chunk)
                    if len(output) > self.max_response_bytes:
                        overflow.set()
                        stop()
                        return
            except (OSError, ValueError):
                io_failed.set()

        writer = threading.Thread(target=write_input, daemon=True)
        reader = threading.Thread(target=read_output, daemon=True)
        writer.start()
        reader.start()
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
            writer.join(timeout=max(0, deadline - time.monotonic()))
            reader.join(timeout=max(0, deadline - time.monotonic()))
            if overflow.is_set():
                raise TransportError(
                    "The local command response exceeds the configured byte limit."
                )
            if writer.is_alive() or reader.is_alive() or time.monotonic() >= deadline:
                raise TransportError("The local completion command timed out; it was not retried.")
            if process.returncode != 0 or io_failed.is_set():
                raise TransportError("The local completion command failed; it was not retried.")
            return _normalized(_decode(bytes(output)), chat_completion=False)
        except subprocess.TimeoutExpired:
            raise TransportError(
                "The local completion command timed out; it was not retried."
            ) from None
        finally:
            stop()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
            writer.join(timeout=1)
            reader.join(timeout=1)
            if not writer.is_alive():
                with suppress(OSError):
                    process.stdin.close()
            if not reader.is_alive():
                process.stdout.close()


__all__ = ["CommandTransport", "HttpTransport", "TransportError"]
