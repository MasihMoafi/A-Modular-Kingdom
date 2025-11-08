#!/usr/bin/env python3
"""Run a command while throttling it based on CPU package temperature.

The script monitors `sensors -j` output and pauses/resumes the wrapped
command using SIGSTOP/SIGCONT when the temperature exceeds a configurable
threshold. This provides a user-space solution for keeping thermals under
control without requiring root access or kernel-level power tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Sequence


def read_package_temperature(preferred_keys: Sequence[tuple[str, str, str]]) -> float:
    """Return the current CPU package temperature in Celsius."""
    try:
        result = subprocess.run(
            ["sensors", "-j"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on host
        raise RuntimeError("`sensors` command not available.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to read sensors output: {exc.stderr}") from exc

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Could not parse sensors JSON output.") from exc

    for chip_key, sensor_key, field_key in preferred_keys:
        try:
            return float(data[chip_key][sensor_key][field_key])
        except (KeyError, TypeError):
            continue

    raise RuntimeError("Unable to locate CPU package temperature in sensors output.")


def launch_command(cmd: Sequence[str], cpu_affinity: Sequence[int] | None) -> subprocess.Popen:
    """Start the wrapped command and optionally restrict CPU affinity."""
    preexec = None
    if cpu_affinity:
        affinity_mask = list(cpu_affinity)

        def set_affinity():  # pragma: no cover - requires OS support
            os.setsid()
            os.sched_setaffinity(0, affinity_mask)

        preexec = set_affinity
    else:
        def start_new_session():  # pragma: no cover - relies on OS
            os.setsid()

        preexec = start_new_session

    return subprocess.Popen(cmd, preexec_fn=preexec)


def throttle_process(proc: subprocess.Popen, *, threshold: float, resume_delta: float, interval: float,
                     preferred_keys: Sequence[tuple[str, str, str]]) -> int:
    """Monitor temperature and pause/resume the process accordingly."""
    paused = False
    resume_threshold = max(0.0, threshold - resume_delta)

    while True:
        retcode = proc.poll()
        if retcode is not None:
            return retcode

        try:
            temp_c = read_package_temperature(preferred_keys)
        except RuntimeError as exc:
            proc.terminate()
            proc.wait(timeout=5)
            raise RuntimeError(f"Thermal guard aborted: {exc}") from exc

        if temp_c >= threshold and not paused:
            os.killpg(proc.pid, signal.SIGSTOP)
            paused = True
            print(f"[thermal-runner] Paused process at {temp_c:.1f}°C", file=sys.stderr)
        elif paused and temp_c <= resume_threshold:
            os.killpg(proc.pid, signal.SIGCONT)
            paused = False
            print(f"[thermal-runner] Resumed process at {temp_c:.1f}°C", file=sys.stderr)

        time.sleep(interval)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute (prefix with -- to separate from script options)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=85.0,
        help="Temperature in °C at which the process will be paused (default: 85)",
    )
    parser.add_argument(
        "--resume-delta",
        type=float,
        default=5.0,
        help="How much the temperature must drop below the threshold before resuming (default: 5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Seconds between temperature checks (default: 1.5)",
    )
    parser.add_argument(
        "--affinity",
        type=str,
        default=None,
        help="Optional CPU core list (e.g. '0,1,2') to restrict the wrapped process",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if not args.command:
        print("No command specified. Use -- to separate script args from command.", file=sys.stderr)
        return 1

    affinity = None
    if args.affinity:
        try:
            affinity = [int(core.strip()) for core in args.affinity.split(',') if core.strip()]
        except ValueError as exc:
            print(f"Invalid affinity list: {exc}", file=sys.stderr)
            return 1

    preferred_keys = (
        ("coretemp-isa-0000", "Package id 0", "temp1_input"),
        ("acpitz-acpi-0", "temp1", "temp1_input"),
    )

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    proc = launch_command(cmd, affinity)

    try:
        return throttle_process(
            proc,
            threshold=args.threshold,
            resume_delta=args.resume_delta,
            interval=args.interval,
            preferred_keys=preferred_keys,
        )
    except KeyboardInterrupt:  # pragma: no cover
        os.killpg(proc.pid, signal.SIGTERM)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
