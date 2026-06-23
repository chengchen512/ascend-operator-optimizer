#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = ".operator-optim"
DEFAULT_KEY_FIELDS = ["case", "shape", "dtype", "mode"]
METRIC_FALLBACKS = (
    "custom",
    "candidate",
    "latency",
    "time",
    "duration",
    "custom_us",
    "candidate_us",
    "latency_us",
    "time_us",
    "duration_us",
)


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"top-level JSON must be object: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def slug(value: str) -> str:
    value = value.strip() or "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def resolve_workspace(config_path: Path, config: dict[str, Any]) -> Path:
    workspace = Path(str(config.get("workspace", "."))).expanduser()
    if not workspace.is_absolute():
        workspace = config_path.parent / workspace
    return workspace.resolve()


def resolve_run_root(workspace: Path, config: dict[str, Any]) -> Path:
    run_root = Path(str(config.get("run_root", DEFAULT_RUN_ROOT))).expanduser()
    if not run_root.is_absolute():
        run_root = workspace / run_root
    return run_root.resolve()


def required_command(config: dict[str, Any], name: str) -> str:
    commands = config.get("commands", {})
    if not isinstance(commands, dict):
        raise ValueError("config.commands must be an object")
    command = str(commands.get(name, "")).strip()
    if not command:
        raise ValueError(f"missing required command: commands.{name}")
    return command


def optional_command(config: dict[str, Any], name: str) -> str:
    commands = config.get("commands", {})
    if not isinstance(commands, dict):
        raise ValueError("config.commands must be an object")
    return str(commands.get(name, "")).strip()


def build_env(config: dict[str, Any], phase: str, label: str, workspace: Path, run_dir: Path, result_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    extra = config.get("environment", {})
    if extra:
        if not isinstance(extra, dict):
            raise ValueError("config.environment must be an object")
        env.update({str(key): str(value) for key, value in extra.items()})
    env.update(
        {
            "OPT_HARNESS_PHASE": phase,
            "OPT_HARNESS_LABEL": label,
            "OPT_HARNESS_ROOT": str(workspace),
            "OPT_HARNESS_RUN_DIR": str(run_dir),
            "OPT_HARNESS_RESULT": str(result_path),
        }
    )
    return env


def run_command(name: str, command: str, cwd: Path, env: dict[str, str], log_dir: Path) -> dict[str, Any]:
    if not command:
        return {"name": name, "command": "", "status": "skipped", "exit_code": None}

    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    started = dt.datetime.now(dt.timezone.utc)
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ended = dt.datetime.now(dt.timezone.utc)
    write_text(stdout_path, proc.stdout)
    write_text(stderr_path, proc.stderr)
    return {
        "name": name,
        "command": command,
        "status": "pass" if proc.returncode == 0 else "fail",
        "exit_code": proc.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def benchmark_options(config: dict[str, Any]) -> dict[str, Any]:
    options = config.get("benchmark", {})
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("config.benchmark must be an object")
    return options


def validate_benchmark_result(path: Path) -> dict[str, Any]:
    data = load_json(path)
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"benchmark JSON must contain non-empty cases array: {path}")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"benchmark case {index} must be an object")
    return data


def case_key(case: dict[str, Any], fields: list[str]) -> str:
    values: list[str] = []
    for field in fields:
        value = case.get(field, "")
        values.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return "|".join(values)


def metric_value(case: dict[str, Any], primary_metric: str) -> float:
    candidates = [primary_metric] if primary_metric else []
    candidates.extend(name for name in METRIC_FALLBACKS if name not in candidates)
    for name in candidates:
        value = case.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    raise ValueError(f"case has no numeric primary metric; tried {candidates}: {case}")


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    options = benchmark_options(config)
    key_fields = list(options.get("case_key_fields") or DEFAULT_KEY_FIELDS)
    primary_metric = str(options.get("primary_metric", "custom"))
    smaller_is_better = bool(options.get("smaller_is_better", True))

    baseline_cases = baseline.get("cases", [])
    candidate_cases = candidate.get("cases", [])
    baseline_by_key = {case_key(case, key_fields): case for case in baseline_cases}
    rows = []
    missing = []
    extra = []

    for case in candidate_cases:
        key = case_key(case, key_fields)
        base_case = baseline_by_key.get(key)
        if base_case is None:
            extra.append(case)
            continue
        base_value = metric_value(base_case, primary_metric)
        cand_value = metric_value(case, primary_metric)
        if smaller_is_better:
            ratio = base_value / cand_value if cand_value else None
            delta_percent = ((cand_value - base_value) / base_value * 100.0) if base_value else None
        else:
            ratio = cand_value / base_value if base_value else None
            delta_percent = ((cand_value - base_value) / base_value * 100.0) if base_value else None
        rows.append(
            {
                "key": key,
                "identity": {field: case.get(field, "") for field in key_fields},
                "baseline": base_value,
                "candidate": cand_value,
                "speedup": ratio,
                "delta_percent": delta_percent,
                "status": status_from_ratio(ratio),
            }
        )

    candidate_keys = {case_key(case, key_fields) for case in candidate_cases}
    for case in baseline_cases:
        key = case_key(case, key_fields)
        if key not in candidate_keys:
            missing.append(case)

    if missing or extra:
        raise ValueError(
            "benchmark cases are not comparable: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    speedups = [row["speedup"] for row in rows if isinstance(row["speedup"], (int, float))]
    mean_speedup = sum(speedups) / len(speedups) if speedups else None
    return {
        "primary_metric": primary_metric,
        "smaller_is_better": smaller_is_better,
        "case_key_fields": key_fields,
        "unit": candidate.get("unit", baseline.get("unit", "")),
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "mean_speedup": mean_speedup,
            "improved": sum(1 for row in rows if row["status"] == "improved"),
            "regressed": sum(1 for row in rows if row["status"] == "regressed"),
            "neutral": sum(1 for row in rows if row["status"] == "neutral"),
        },
    }


def status_from_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    if ratio > 1.01:
        return "improved"
    if ratio < 0.99:
        return "regressed"
    return "neutral"


def render_summary(run_record: dict[str, Any], comparison: dict[str, Any] | None) -> str:
    phase = run_record["phase"]
    label = run_record["label"]
    lines = [
        f"# Optimization Harness Run: {label}",
        "",
        "## Metadata",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Phase | {phase} |",
        f"| Status | {run_record['status']} |",
        f"| Run directory | `{run_record['run_dir']}` |",
        "",
        "## Commands",
        "",
        "| Step | Status | Exit |",
        "| --- | --- | ---: |",
    ]
    for command in run_record["commands"]:
        lines.append(f"| {command['name']} | {command['status']} | {command.get('exit_code', '')} |")
    if comparison:
        summary = comparison["summary"]
        lines.extend(
            [
                "",
                "## Comparison",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Cases | {summary['cases']} |",
                f"| Mean speedup | {format_optional(summary['mean_speedup'])} |",
                f"| Improved | {summary['improved']} |",
                f"| Neutral | {summary['neutral']} |",
                f"| Regressed | {summary['regressed']} |",
            ]
        )
    return "\n".join(lines) + "\n"


def format_optional(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "N/A"
    return str(value)


def run_harness(phase: str, args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = load_json(config_path)
        workspace = resolve_workspace(config_path, config)
        run_root = resolve_run_root(workspace, config)
        correctness_cmd = required_command(config, "correctness")
        benchmark_cmd = required_command(config, "benchmark")
        build_cmd = optional_command(config, "build")
    except ValueError as exc:
        return fail(str(exc))

    label = args.label or phase
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / "runs" / f"{timestamp}-{phase}-{slug(label)}"
    log_dir = run_dir / "logs"
    result_path = run_dir / "benchmark.json"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.json")

    env = build_env(config, phase, label, workspace, run_dir, result_path)
    commands = [
        run_command("build", build_cmd, workspace, env, log_dir),
        run_command("correctness", correctness_cmd, workspace, env, log_dir),
    ]
    status = "pass"
    comparison = None

    if commands[-1]["status"] != "pass" or commands[0]["status"] == "fail":
        status = "fail"
    else:
        commands.append(run_command("benchmark", benchmark_cmd, workspace, env, log_dir))
        if commands[-1]["status"] != "pass":
            status = "fail"
        else:
            try:
                candidate = validate_benchmark_result(result_path)
                if phase == "baseline":
                    write_json(run_root / "baseline.json", {"run_dir": str(run_dir), "result": candidate})
                else:
                    baseline_record = load_json(run_root / "baseline.json")
                    baseline = baseline_record.get("result")
                    if not isinstance(baseline, dict):
                        raise ValueError(f"invalid baseline record: {run_root / 'baseline.json'}")
                    comparison = compare_results(baseline, candidate, config)
                    write_json(run_dir / "comparison.json", comparison)
                    write_json(run_root / "latest-evaluation.json", {"run_dir": str(run_dir), "comparison": comparison})
            except ValueError as exc:
                status = "fail"
                write_text(run_dir / "error.txt", str(exc) + "\n")

    run_record = {
        "phase": phase,
        "label": label,
        "status": status,
        "workspace": str(workspace),
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "commands": commands,
    }
    write_json(run_dir / "run.json", run_record)
    write_text(run_dir / "summary.md", render_summary(run_record, comparison))

    print(f"run_dir: {run_dir}")
    print(f"status: {status}")
    if comparison:
        print(f"mean_speedup: {format_optional(comparison['summary']['mean_speedup'])}")
        print(f"comparison: {run_dir / 'comparison.json'}")
    return 0 if status == "pass" else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    root = Path(args.skill_dir).expanduser().resolve() if args.skill_dir else Path(__file__).resolve().parents[1]
    checks = [
        ("SKILL.md", root / "SKILL.md"),
        ("assets/harness.example.json", root / "assets" / "harness.example.json"),
        ("references/INDEX.md", root / "references" / "INDEX.md"),
        ("references/hardware.md", root / "references" / "hardware.md"),
        ("references/playbook.md", root / "references" / "playbook.md"),
        ("references/constraints.md", root / "references" / "constraints.md"),
        ("references/harness.md", root / "references" / "harness.md"),
        ("references/ascendc.md", root / "references" / "ascendc.md"),
        ("references/ascendc-examples.md", root / "references" / "ascendc-examples.md"),
        ("references/sources.md", root / "references" / "sources.md"),
        ("scripts/harness.py", root / "scripts" / "harness.py"),
    ]
    ok = True
    for name, path in checks:
        exists = path.is_file()
        ok = ok and exists
        print(f"{'OK' if exists else 'FAIL'}  {name}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic Ascend operator optimization harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="run build/correctness/benchmark and save baseline")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--label", default="baseline")
    baseline.set_defaults(func=lambda args: run_harness("baseline", args))

    evaluate = subparsers.add_parser("evaluate", help="run candidate evaluation and compare with baseline")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--label", required=True)
    evaluate.set_defaults(func=lambda args: run_harness("evaluate", args))

    doctor = subparsers.add_parser("doctor", help="check self-contained Skill layout")
    doctor.add_argument("--skill-dir", default="")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())



