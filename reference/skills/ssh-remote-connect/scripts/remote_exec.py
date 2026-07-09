#!/usr/bin/env python3
"""Windows-friendly SSH command runner for the ssh-remote-connect skill."""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import sys
import threading
import time
from typing import Iterable

try:
    import paramiko
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise SystemExit(
        "Missing dependency: paramiko. Install it with `python -m pip install paramiko`."
    ) from exc


SENSITIVE_KEYS = {"SSH_PASSWORD", "SSH_PRIVATE_KEY", "SSH_KEY_PASSPHRASE"}


def default_config_path() -> pathlib.Path:
    script_dir = pathlib.Path(__file__).resolve().parent
    return pathlib.Path(os.environ.get("SSH_REMOTE_CONFIG", script_dir / "connection.local.env"))


def parse_env_file(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SSH config: {path}\nCreate it from: {path.with_name('connection.env.example')}"
        )

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            parts = shlex.split(raw_value, posix=True)
            value = parts[0] if parts else ""
        except ValueError:
            value = raw_value.strip().strip("'\"")
        values[key] = value
    return values


def require_config(values: dict[str, str], path: pathlib.Path) -> tuple[str, int, str, str]:
    missing = [name for name in ("SSH_HOST", "SSH_PORT", "SSH_USER") if not values.get(name)]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} in {path}")

    try:
        port = int(values["SSH_PORT"])
    except ValueError as exc:
        raise ValueError(f"SSH_PORT must be an integer in {path}") from exc

    return values["SSH_HOST"], port, values["SSH_USER"], values.get("SSH_PASSWORD", "")


def make_client(
    host: str,
    port: int,
    user: str,
    password: str,
    timeout: float,
) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password or None,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        look_for_keys=not bool(password),
        allow_agent=not bool(password),
    )
    return client


def run_command(
    client: paramiko.SSHClient,
    command: str,
    *,
    timeout: float,
    pty: bool,
) -> int:
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport is not available")

    channel = transport.open_session(timeout=timeout)
    if pty:
        channel.get_pty(term="xterm")
    channel.exec_command(command)

    deadline = time.monotonic() + timeout if timeout > 0 else None
    while True:
        while channel.recv_ready():
            sys.stdout.buffer.write(channel.recv(32768))
            sys.stdout.buffer.flush()
        while channel.recv_stderr_ready():
            sys.stderr.buffer.write(channel.recv_stderr(32768))
            sys.stderr.buffer.flush()

        if channel.exit_status_ready():
            break
        if deadline is not None and time.monotonic() > deadline:
            channel.close()
            raise TimeoutError(f"Remote command timed out after {timeout:.0f}s")
        time.sleep(0.05)

    while channel.recv_ready():
        sys.stdout.buffer.write(channel.recv(32768))
    while channel.recv_stderr_ready():
        sys.stderr.buffer.write(channel.recv_stderr(32768))
    sys.stdout.buffer.flush()
    sys.stderr.buffer.flush()
    return channel.recv_exit_status()


def interactive_shell(client: paramiko.SSHClient) -> int:
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport is not available")

    channel = transport.open_session()
    channel.get_pty(term=os.environ.get("TERM", "xterm"))
    channel.invoke_shell()

    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            if channel.recv_ready():
                data = channel.recv(32768)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            elif channel.exit_status_ready():
                break
            else:
                time.sleep(0.05)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for line in sys.stdin:
            channel.send(line)
    finally:
        stop.set()
        channel.close()
        thread.join(timeout=1)
    return 0


def command_from_args(parts: Iterable[str], bash: bool) -> str:
    command = " ".join(parts).strip()
    if command.startswith("-- "):
        command = command[3:].strip()
    if not command:
        return ""
    if bash:
        return f"bash -lc {shlex.quote(command)}"
    return command


def print_config(values: dict[str, str], path: pathlib.Path) -> None:
    print(f"CONFIG={path}")
    for key in sorted(values):
        if key in SENSITIVE_KEYS:
            status = "set" if values[key] else "empty"
            print(f"{key}=<{status}>")
        else:
            print(f"{key}={values[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SSH commands from connection.local.env")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="remote command to execute")
    parser.add_argument("--config", type=pathlib.Path, default=default_config_path())
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--raw", action="store_true", help="do not wrap the command with bash -lc")
    parser.add_argument("--pty", action="store_true", help="allocate a remote pseudo-terminal")
    parser.add_argument("--show-config", action="store_true", help="print non-sensitive config and exit")
    parser.add_argument("--interactive", action="store_true", help="open a line-mode interactive shell")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    values = parse_env_file(config_path)
    if args.show_config:
        print_config(values, config_path)
        return 0

    host, port, user, password = require_config(values, config_path)
    remote_command = command_from_args(args.command, bash=not args.raw)

    client = make_client(host, port, user, password, timeout=args.timeout)
    try:
        if args.interactive or not remote_command:
            return interactive_shell(client)
        return run_command(client, remote_command, timeout=args.timeout, pty=args.pty)
    finally:
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, TimeoutError, OSError, paramiko.SSHException) as exc:
        print(f"ssh-remote-connect: {exc}", file=sys.stderr)
        raise SystemExit(1)
