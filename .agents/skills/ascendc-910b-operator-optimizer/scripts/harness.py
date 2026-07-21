#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = ".ascendc-opt"
DEFAULT_KEY_FIELDS = ["case", "shape", "dtype", "mode"]
ASCENDC_PROBE_REQUIRED_PROFILES = {
    "ascendc-msopgen-aclnn-dynamic",
    "msopgen-aclnn-dynamic",
}
DEFAULT_PROFILE_KERNEL_GLOB = "**/kernel_details.csv"
DEFAULT_PROFILE_OP_STAT_GLOB = "**/op_statistic.csv"
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


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


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


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config.{name} must be an object")
    return value


def implementation_profile(config: dict[str, Any]) -> dict[str, str]:
    implementation = config_section(config, "implementation")
    operator = config_section(config, "operator")
    return {
        "type": str(implementation.get("type", operator.get("framework", ""))).strip(),
        "launch_profile": str(implementation.get("launch_profile", "")).strip(),
        "op_name": str(implementation.get("op_name", operator.get("name", ""))).strip(),
    }


def validate_scope(config: dict[str, Any]) -> None:
    target = config_section(config, "target")
    implementation = implementation_profile(config)
    implementation_type = implementation["type"].lower().replace("-", "")
    if implementation_type not in {"ascendc", ""}:
        raise ValueError("this harness accepts only Ascend C implementations")
    soc_family = str(target.get("soc_family", "Ascend910B")).lower().replace(" ", "")
    if "910b" not in soc_family:
        raise ValueError(f"target.soc_family must be Ascend 910B, got {soc_family!r}")
    cann_version = str(target.get("cann_version", "9.0.0")).strip()
    if cann_version != "9.0.0":
        raise ValueError(f"target.cann_version must be 9.0.0, got {cann_version!r}")


def gate_enabled(config: dict[str, Any], name: str, default: bool) -> bool:
    gates = config_section(config, "gates")
    return bool(gates.get(name, default))


def launch_probe_is_required(config: dict[str, Any]) -> bool:
    gates = config_section(config, "gates")
    if "require_launch_probe" in gates:
        return bool(gates["require_launch_probe"])
    profile = implementation_profile(config).get("launch_profile", "")
    return profile in ASCENDC_PROBE_REQUIRED_PROFILES


def build_env(
    config: dict[str, Any],
    phase: str,
    label: str,
    workspace: Path,
    run_dir: Path,
    result_path: Path,
    probe_result_path: Path,
    profile_dir: Path,
    profile_result_path: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    extra = config.get("environment", {})
    if extra:
        if not isinstance(extra, dict):
            raise ValueError("config.environment must be an object")
        env.update({str(key): str(value) for key, value in extra.items()})
    implementation = implementation_profile(config)
    env.update(
        {
            "ASCENDC_OPT_PHASE": phase,
            "ASCENDC_OPT_LABEL": label,
            "ASCENDC_OPT_ROOT": str(workspace),
            "ASCENDC_OPT_RUN_DIR": str(run_dir),
            "ASCENDC_OPT_RESULT": str(result_path),
            "ASCENDC_OPT_PROBE_RESULT": str(probe_result_path),
            "ASCENDC_OPT_PROFILE_DIR": str(profile_dir),
            "ASCENDC_OPT_PROFILE_RESULT": str(profile_result_path),
            "ASCENDC_OPT_LAUNCH_PROFILE": implementation["launch_profile"],
            "ASCENDC_OPT_OP_NAME": implementation["op_name"],
            "OPT_HARNESS_PHASE": phase,
            "OPT_HARNESS_LABEL": label,
            "OPT_HARNESS_ROOT": str(workspace),
            "OPT_HARNESS_RUN_DIR": str(run_dir),
            "OPT_HARNESS_RESULT": str(result_path),
            "OPT_HARNESS_PROBE_RESULT": str(probe_result_path),
            "OPT_HARNESS_PROFILE_DIR": str(profile_dir),
            "OPT_HARNESS_PROFILE_RESULT": str(profile_result_path),
            "OPT_HARNESS_IMPLEMENTATION_TYPE": implementation["type"],
            "OPT_HARNESS_LAUNCH_PROFILE": implementation["launch_profile"],
            "OPT_HARNESS_OP_NAME": implementation["op_name"],
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


def validate_launch_probe_result(path: Path, required: bool, min_coverage: float) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ValueError(f"launch probe must write JSON to {path}")
        return None
    data = load_json(path)
    status = str(data.get("status", "")).lower()
    coverage = data.get("coverage", data.get("coverage_ratio", None))
    if required and status not in {"pass", "ok", "success"}:
        raise ValueError(f"launch probe failed or missing pass status: {path}")
    if isinstance(coverage, (int, float)) and not isinstance(coverage, bool):
        if required and float(coverage) < min_coverage:
            raise ValueError(f"launch probe coverage {coverage} is below required {min_coverage}")
    elif required:
        raise ValueError("launch probe JSON must contain numeric coverage or coverage_ratio")
    return data


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if path.is_file():
        return load_json(path)
    return None


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

    baseline_unit = baseline.get("unit", "")
    candidate_unit = candidate.get("unit", "")
    if baseline_unit and candidate_unit and baseline_unit != candidate_unit:
        raise ValueError(f"benchmark units are not comparable: baseline={baseline_unit} candidate={candidate_unit}")

    metadata_fields = list(options.get("metadata_fields") or [])
    baseline_metadata = baseline.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    if isinstance(baseline_metadata, dict) and isinstance(candidate_metadata, dict):
        for field in metadata_fields:
            base_value = baseline_metadata.get(field)
            candidate_value = candidate_metadata.get(field)
            if base_value is not None and candidate_value is not None and base_value != candidate_value:
                raise ValueError(
                    f"benchmark metadata is not comparable for {field}: "
                    f"baseline={base_value!r} candidate={candidate_value!r}"
                )

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


def candidate_decision(comparison: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    summary = comparison["summary"]
    threshold = float(benchmark_options(config).get("noise_threshold_percent", 1.0))
    minimum_speedup = 1.0 + threshold / 100.0
    mean_speedup = summary.get("mean_speedup")
    if not isinstance(mean_speedup, (int, float)):
        return False, "missing comparable speedup"
    if summary.get("regressed", 0) > 0:
        return False, "one or more protected cases regressed"
    if mean_speedup < minimum_speedup:
        return False, f"mean speedup {mean_speedup:.6g} is below {minimum_speedup:.6g}"
    return True, "all cases passed and mean speedup exceeded the noise threshold"


def command_status(commands: list[dict[str, Any]], name: str) -> str:
    for command in commands:
        if command.get("name") == name:
            return str(command.get("status", "unknown"))
    return "skipped"


def normalize_csv_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({normalize_csv_key(str(k)): str(v or "").strip() for k, v in row.items()})
        return rows


def first_value(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value
    return default


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return default


def find_profile_files(profile_dir: Path, pattern: str) -> list[Path]:
    if not profile_dir.exists():
        return []
    return sorted(path for path in profile_dir.glob(pattern) if path.is_file())


def profile_row_name(row: dict[str, str]) -> str:
    return first_value(row, ("name", "kernelname", "opname", "taskname"), "unknown")


def profile_row_type(row: dict[str, str]) -> str:
    return first_value(row, ("tasktype", "type", "acceleratorcore", "coretype"), "unknown")


def profile_row_start(row: dict[str, str]) -> float | None:
    value = first_value(row, ("starttimeus", "starttime", "startus", "timestampus", "ts"), "")
    return parse_float(value) if value else None


def profile_row_duration(row: dict[str, str]) -> float:
    return parse_float(
        first_value(
            row,
            ("durationus", "taskdurationus", "elapsedtimeus", "totaltimeus", "executiontimeus", "dur"),
            "0",
        )
    )


def profile_row_wait(row: dict[str, str]) -> float:
    return parse_float(first_value(row, ("waittimeus", "taskwaittimeus", "waitus"), "0"))


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end < start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def profile_task_rows(kernel_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in kernel_files:
        for raw in load_csv_dicts(path):
            start_us = profile_row_start(raw)
            duration_us = profile_row_duration(raw)
            wait_us = profile_row_wait(raw)
            rows.append(
                {
                    "name": profile_row_name(raw),
                    "task_type": profile_row_type(raw),
                    "stream": first_value(raw, ("streamid", "stream", "tid"), ""),
                    "start_us": start_us,
                    "end_us": start_us + duration_us if start_us is not None else None,
                    "duration_us": duration_us,
                    "wait_us": wait_us,
                    "total_cost_us": duration_us + wait_us,
                    "source": str(path),
                }
            )
    return rows


def profile_op_stat_rows(op_stat_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in op_stat_files:
        for raw in load_csv_dicts(path):
            rows.append(
                {
                    "name": profile_row_name(raw),
                    "total_time_us": parse_float(first_value(raw, ("totaltimeus", "totalus", "timeus"), "0")),
                    "count": parse_float(first_value(raw, ("count", "calls", "executioncount"), "0")),
                    "source": str(path),
                }
            )
    return sorted(rows, key=lambda row: row["total_time_us"], reverse=True)


def compact_profile_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name", "unknown"),
        "task_type": row.get("task_type", "unknown"),
        "stream": row.get("stream", ""),
        "duration_us": row.get("duration_us", 0.0),
        "wait_us": row.get("wait_us", 0.0),
        "total_cost_us": row.get("total_cost_us", 0.0),
        "source": row.get("source", ""),
    }


def analyze_profile(profile_dir: Path, config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    options = config_section(config, "profiling")
    kernel_files = find_profile_files(profile_dir, str(options.get("kernel_details_glob", DEFAULT_PROFILE_KERNEL_GLOB)))
    op_stat_files = find_profile_files(profile_dir, str(options.get("op_statistic_glob", DEFAULT_PROFILE_OP_STAT_GLOB)))
    op_stats = profile_op_stat_rows(op_stat_files)
    if not kernel_files:
        analysis = {
            "status": "no_kernel_details",
            "profile_dir": str(profile_dir),
            "kernel_details_files": [],
            "op_statistic_files": [str(path) for path in op_stat_files],
            "top_ops_by_total_time": op_stats[:10],
            "tags": ["INSUFFICIENT_TRACE_DETAIL"],
            "certainty": "insufficient_evidence",
        }
        write_json(run_dir / "profile_analysis.json", analysis)
        write_profile_summary(run_dir / "profile_summary.md", analysis)
        return analysis

    rows = profile_task_rows(kernel_files)
    timed_rows = [row for row in rows if row["start_us"] is not None and row["end_us"] is not None]
    intervals = merge_intervals([(row["start_us"], row["end_us"]) for row in timed_rows])
    service_us = intervals[-1][1] - intervals[0][0] if intervals else 0.0
    busy_union_us = sum(end - start for start, end in intervals)
    gaps = [
        {"start_us": left[1], "end_us": right[0], "duration_us": right[0] - left[1]}
        for left, right in zip(intervals, intervals[1:])
        if right[0] > left[1]
    ]
    underfeed_us = max(0.0, service_us - busy_union_us)
    underfeed_ratio = underfeed_us / service_us if service_us else 0.0
    largest_gap_us = max((gap["duration_us"] for gap in gaps), default=0.0)
    wait_anchor_candidates = []
    for index, row in enumerate(sorted(rows, key=lambda item: item["total_cost_us"], reverse=True)[:10], start=1):
        total_cost = row["total_cost_us"]
        wait_ratio = row["wait_us"] / total_cost if total_cost else 0.0
        if wait_ratio > 0.95 and row["duration_us"] < 10.0:
            item = compact_profile_row(row)
            item.update({"rank": index, "wait_ratio": wait_ratio})
            wait_anchor_candidates.append(item)

    tags = []
    if underfeed_ratio >= float(options.get("underfeed_ratio_threshold", 0.10)):
        tags.append("DEVICE_IDLE_GAP_HEAVY")
    if largest_gap_us >= float(options.get("bubble_threshold_us", 50.0)):
        tags.append("INTERNAL_BUBBLE_HEAVY")
    if wait_anchor_candidates:
        tags.append("WAIT_ANCHOR_FALSE_HOTSPOT")
    if not tags:
        tags.append("NO_MAJOR_PROFILE_ANOMALY")

    analysis = {
        "status": "ok",
        "profile_dir": str(profile_dir),
        "kernel_details_files": [str(path) for path in kernel_files],
        "op_statistic_files": [str(path) for path in op_stat_files],
        "summary": {
            "service_us": service_us,
            "device_busy_union_us": busy_union_us,
            "kernel_sum_us": sum(row["duration_us"] for row in rows),
            "total_cost_sum_us": sum(row["total_cost_us"] for row in rows),
            "wait_sum_us": sum(row["wait_us"] for row in rows),
            "underfeed_us": underfeed_us,
            "underfeed_ratio": underfeed_ratio,
            "internal_bubble_count": len(gaps),
            "largest_internal_bubble_us": largest_gap_us,
        },
        "top_kernels_by_duration": [compact_profile_row(row) for row in sorted(rows, key=lambda item: item["duration_us"], reverse=True)[:10]],
        "top_ops_by_total_time": op_stats[:10],
        "top_internal_bubbles": sorted(gaps, key=lambda item: item["duration_us"], reverse=True)[:10],
        "wait_anchor_candidates": wait_anchor_candidates,
        "tags": tags,
        "certainty": "profile_only_soft_signal",
    }
    write_json(run_dir / "profile_analysis.json", analysis)
    write_profile_summary(run_dir / "profile_summary.md", analysis)
    return analysis


def write_profile_summary(path: Path, analysis: dict[str, Any]) -> None:
    summary = analysis.get("summary", {})
    lines = [
        "# Profile Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Status | {analysis.get('status', 'unknown')} |",
        f"| Tags | {', '.join(analysis.get('tags', []))} |",
        f"| Certainty | {analysis.get('certainty', 'unknown')} |",
        f"| Service(us) | {format_optional(summary.get('service_us'))} |",
        f"| Busy union(us) | {format_optional(summary.get('device_busy_union_us'))} |",
        f"| Underfeed ratio | {format_optional(summary.get('underfeed_ratio'))} |",
        f"| Largest bubble(us) | {format_optional(summary.get('largest_internal_bubble_us'))} |",
        "",
        "## Top Kernels",
        "",
        "| Name | Type | Duration(us) | Wait(us) |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in analysis.get("top_kernels_by_duration", [])[:10]:
        lines.append(
            f"| {row.get('name', '')} | {row.get('task_type', '')} | "
            f"{format_optional(row.get('duration_us'))} | {format_optional(row.get('wait_us'))} |"
        )
    lines.extend(
        [
            "",
            "Profile tags are soft diagnostic signals. Acceptance still depends on correctness and comparable benchmark results.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def collect_artifacts(config: dict[str, Any], workspace: Path, run_dir: Path) -> list[str]:
    artifacts = config_section(config, "artifacts")
    patterns = artifacts.get("collect", [])
    if not isinstance(patterns, list):
        raise ValueError("config.artifacts.collect must be an array")
    collected: list[str] = []
    output_root = run_dir / "collected"
    for pattern in patterns:
        for path in workspace.glob(str(pattern)):
            if not path.is_file():
                continue
            try:
                relative = path.resolve().relative_to(workspace)
            except ValueError:
                continue
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            collected.append(str(target))
    return collected


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

    artifacts = run_record.get("artifacts", {})
    if artifacts:
        lines.extend(["", "## Artifacts", "", "| Name | Path |", "| --- | --- |"])
        for name, value in artifacts.items():
            if isinstance(value, list):
                value = f"{len(value)} files"
            lines.append(f"| {name} | `{value}` |")

    diagnostics = run_record.get("diagnostics", {})
    profile_analysis = diagnostics.get("profile_analysis")
    if isinstance(profile_analysis, dict):
        profile_summary = profile_analysis.get("summary", {})
        lines.extend(
            [
                "",
                "## Profile",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Status | {profile_analysis.get('status', 'unknown')} |",
                f"| Underfeed ratio | {format_optional(profile_summary.get('underfeed_ratio'))} |",
                f"| Largest internal bubble(us) | {format_optional(profile_summary.get('largest_internal_bubble_us'))} |",
                f"| Tags | {', '.join(profile_analysis.get('tags', []))} |",
            ]
        )

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
        validate_scope(config)
        workspace = resolve_workspace(config_path, config)
        run_root = resolve_run_root(workspace, config)
        precision_cmd = optional_command(config, "precision") or optional_command(config, "correctness")
        if not precision_cmd:
            raise ValueError("missing required command: commands.precision")
        benchmark_cmd = required_command(config, "benchmark")
        env_smoke_cmd = optional_command(config, "env_smoke")
        build_cmd = optional_command(config, "build")
        launch_probe_cmd = optional_command(config, "launch_probe")
        profile_cmd = optional_command(config, "profile")
        if launch_probe_is_required(config) and not launch_probe_cmd:
            raise ValueError("commands.launch_probe is required for this launch_profile")
    except ValueError as exc:
        return fail(str(exc))

    label = args.label or phase
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / "runs" / f"{timestamp}-{phase}-{slug(label)}"
    log_dir = run_dir / "logs"
    result_path = run_dir / "benchmark.json"
    probe_result_path = run_dir / "launch_probe.json"
    profile_dir = run_dir / "profile"
    profile_result_path = run_dir / "profile.json"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.json")

    env = build_env(config, phase, label, workspace, run_dir, result_path, probe_result_path, profile_dir, profile_result_path)
    commands: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "implementation": implementation_profile(config),
    }
    artifacts: dict[str, Any] = {
        "benchmark": str(result_path),
        "logs": str(log_dir),
    }
    status = "pass"
    comparison = None

    if env_smoke_cmd:
        commands.append(run_command("env_smoke", env_smoke_cmd, workspace, env, log_dir))
        if commands[-1]["status"] != "pass":
            status = "fail"

    if status == "pass":
        commands.append(run_command("build", build_cmd, workspace, env, log_dir))
        if commands[-1]["status"] == "fail":
            status = "fail"

    if status == "pass" and launch_probe_cmd:
        commands.append(run_command("launch_probe", launch_probe_cmd, workspace, env, log_dir))
        required_probe = launch_probe_is_required(config)
        min_coverage = float(config_section(config, "probes").get("launch_coverage_min", 1.0))
        try:
            probe_result = validate_launch_probe_result(probe_result_path, required_probe, min_coverage)
            if probe_result is not None:
                diagnostics["launch_probe"] = probe_result
                artifacts["launch_probe"] = str(probe_result_path)
        except ValueError as exc:
            status = "fail"
            write_text(run_dir / "error.txt", str(exc) + "\n")
        if commands[-1]["status"] != "pass":
            status = "fail"

    if status == "pass":
        commands.append(run_command("precision", precision_cmd, workspace, env, log_dir))
        if commands[-1]["status"] != "pass":
            status = "fail"

    benchmark_ok = False
    if status == "pass":
        commands.append(run_command("benchmark", benchmark_cmd, workspace, env, log_dir))
        if commands[-1]["status"] != "pass":
            status = "fail"
        else:
            benchmark_ok = True
            try:
                candidate = validate_benchmark_result(result_path)
                if phase == "baseline":
                    write_json(
                        run_root / "baseline.json",
                        {"label": label, "run_dir": str(run_dir), "result": candidate},
                    )
                elif gate_enabled(config, "require_same_cases", True):
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

    if benchmark_ok and profile_cmd:
        commands.append(run_command("profile", profile_cmd, workspace, env, log_dir))
        artifacts["profile_dir"] = str(profile_dir)
        artifacts["profile_result"] = str(profile_result_path)
        if commands[-1]["status"] != "pass" and gate_enabled(config, "require_profile", False):
            status = "fail"
        try:
            profile_result = maybe_load_json(profile_result_path)
            if profile_result is not None:
                diagnostics["profile"] = profile_result
        except ValueError as exc:
            status = "fail"
            write_text(run_dir / "error.txt", str(exc) + "\n")

    if benchmark_ok and (profile_cmd or config_section(config, "profiling").get("analyze")):
        try:
            diagnostics["profile_analysis"] = analyze_profile(profile_dir, config, run_dir)
            artifacts["profile_analysis"] = str(run_dir / "profile_analysis.json")
            artifacts["profile_summary"] = str(run_dir / "profile_summary.md")
        except Exception as exc:
            diagnostics["profile_analysis_error"] = str(exc)
            write_text(run_dir / "profile_analysis_error.txt", str(exc) + "\n")

    try:
        collected = collect_artifacts(config, workspace, run_dir)
        if collected:
            artifacts["collected"] = collected
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
        "artifacts": artifacts,
        "diagnostics": diagnostics,
    }
    write_json(run_dir / "run.json", run_record)
    write_text(run_dir / "summary.md", render_summary(run_record, comparison))

    if phase == "evaluate" and comparison is not None:
        accepted, reason = candidate_decision(comparison, config)
        rows = comparison.get("rows", [])
        baseline_values = [row.get("baseline") for row in rows if isinstance(row.get("baseline"), (int, float))]
        candidate_values = [row.get("candidate") for row in rows if isinstance(row.get("candidate"), (int, float))]
        history_record = {
            "candidate": label,
            "parent": "best" if (run_root / "best.json").is_file() else "baseline",
            "hypothesis": str(getattr(args, "hypothesis", "") or "not recorded"),
            "changes": list(getattr(args, "change", []) or []),
            "build": command_status(commands, "build"),
            "precision": command_status(commands, "precision"),
            "baseline_metric": sum(baseline_values) / len(baseline_values) if baseline_values else None,
            "candidate_metric": sum(candidate_values) / len(candidate_values) if candidate_values else None,
            "speedup": comparison["summary"].get("mean_speedup"),
            "accepted": accepted,
            "decision_reason": reason,
            "run_dir": str(run_dir),
            "timestamp": timestamp,
        }
        append_jsonl(run_root / "history.jsonl", history_record)
        write_json(run_dir / "decision.json", history_record)
        if accepted:
            write_json(
                run_root / "best.json",
                {"candidate": label, "run_dir": str(run_dir), "result": candidate, "comparison": comparison},
            )
        run_record["decision"] = history_record
        write_json(run_dir / "run.json", run_record)
        write_text(run_dir / "summary.md", render_summary(run_record, comparison))

    print(f"run_dir: {run_dir}")
    print(f"status: {status}")
    if comparison:
        print(f"mean_speedup: {format_optional(comparison['summary']['mean_speedup'])}")
        print(f"accepted: {run_record.get('decision', {}).get('accepted', False)}")
        print(f"comparison: {run_dir / 'comparison.json'}")
    if "profile_analysis" in diagnostics:
        print(f"profile_analysis: {run_dir / 'profile_analysis.json'}")
    return 0 if status == "pass" else 1


def cmd_init(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    state_dir = workspace / DEFAULT_RUN_ROOT
    output_spec = state_dir / "spec.json"
    if output_spec.exists() and not args.force:
        return fail(f"state already exists: {output_spec}; use --force to replace the spec")
    try:
        config = load_json(spec_path)
        validate_scope(config)
    except ValueError as exc:
        return fail(str(exc))
    config["workspace"] = str(workspace)
    config["run_root"] = DEFAULT_RUN_ROOT
    for directory in ("candidates", "runs", "reports"):
        (state_dir / directory).mkdir(parents=True, exist_ok=True)
    write_json(output_spec, config)
    if not (state_dir / "history.jsonl").exists():
        write_text(state_dir / "history.jsonl", "")
    if args.cases:
        cases_path = Path(args.cases).expanduser().resolve()
        target_cases = state_dir / "benchmark-cases.jsonl"
        target_cases.write_text(cases_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"state_dir: {state_dir}")
    print(f"config: {output_spec}")
    return 0


def environment_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "ASCEND_AICPU_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    )
    tools = ("bisheng", "opc", "msprof", "npu-smi", "python", "python3")
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "target": config_section(config, "target"),
        "environment": {key: os.environ.get(key, "") for key in keys},
        "tools": {name: shutil.which(name) for name in tools},
    }


def run_single_phase(phase: str, args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = load_json(config_path)
        validate_scope(config)
        workspace = resolve_workspace(config_path, config)
        run_root = resolve_run_root(workspace, config)
    except ValueError as exc:
        return fail(str(exc))

    command_name = {
        "env": "env_smoke",
        "build": "build",
        "precision": "precision",
        "benchmark": "benchmark",
        "profile": "profile",
    }[phase]
    command = optional_command(config, command_name)
    if phase == "precision" and not command:
        command = optional_command(config, "correctness")
    if phase != "env" and not command:
        return fail(f"missing required command: commands.{command_name}")

    label = getattr(args, "label", "") or phase
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = run_root / "runs" / f"{timestamp}-{phase}-{slug(label)}"
    log_dir = run_dir / "logs"
    result_path = run_dir / "benchmark.json"
    probe_result_path = run_dir / "launch_probe.json"
    profile_dir = run_dir / "profile"
    profile_result_path = run_dir / "profile.json"
    run_dir.mkdir(parents=True, exist_ok=False)
    env = build_env(
        config,
        phase,
        label,
        workspace,
        run_dir,
        result_path,
        probe_result_path,
        profile_dir,
        profile_result_path,
    )
    commands: list[dict[str, Any]] = []
    status = "pass"
    diagnostics: dict[str, Any] = {"implementation": implementation_profile(config)}
    artifacts: dict[str, Any] = {"logs": str(log_dir)}

    if command:
        commands.append(run_command(command_name, command, workspace, env, log_dir))
        if commands[-1]["status"] != "pass":
            status = "fail"

    if phase == "env":
        snapshot = environment_snapshot(config)
        snapshot["env_smoke"] = commands[-1] if commands else {"status": "skipped"}
        write_json(run_root / "environment.json", snapshot)
        write_json(run_dir / "environment.json", snapshot)
        artifacts["environment"] = str(run_root / "environment.json")
    elif phase == "benchmark" and status == "pass":
        try:
            result = validate_benchmark_result(result_path)
            write_json(run_root / "latest-benchmark.json", {"run_dir": str(run_dir), "result": result})
            artifacts["benchmark"] = str(result_path)
        except ValueError as exc:
            status = "fail"
            write_text(run_dir / "error.txt", str(exc) + "\n")
    elif phase == "profile" and status == "pass":
        try:
            diagnostics["profile_analysis"] = analyze_profile(profile_dir, config, run_dir)
            artifacts["profile_dir"] = str(profile_dir)
            artifacts["profile_analysis"] = str(run_dir / "profile_analysis.json")
        except Exception as exc:
            if gate_enabled(config, "require_profile", False):
                status = "fail"
            diagnostics["profile_analysis_error"] = str(exc)

    run_record = {
        "phase": phase,
        "label": label,
        "status": status,
        "workspace": str(workspace),
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "commands": commands,
        "artifacts": artifacts,
        "diagnostics": diagnostics,
    }
    write_json(run_dir / "run.json", run_record)
    write_text(run_dir / "summary.md", render_summary(run_record, None))
    print(f"run_dir: {run_dir}")
    print(f"status: {status}")
    return 0 if status == "pass" else 1


def cmd_compare(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    try:
        config = load_json(config_path)
        validate_scope(config)
        workspace = resolve_workspace(config_path, config)
        run_root = resolve_run_root(workspace, config)
        baseline_record = load_json(Path(args.baseline).expanduser().resolve()) if args.baseline else load_json(run_root / "baseline.json")
        baseline = baseline_record.get("result", baseline_record)
        if not isinstance(baseline, dict):
            raise ValueError("baseline record does not contain a result object")
        candidate_record = load_json(candidate_path)
        candidate = candidate_record.get("result", candidate_record)
        if not isinstance(candidate, dict):
            raise ValueError("candidate record does not contain a result object")
        comparison = compare_results(baseline, candidate, config)
    except ValueError as exc:
        return fail(str(exc))
    output = Path(args.output).expanduser().resolve() if args.output else candidate_path.parent / "comparison.json"
    write_json(output, comparison)
    print(f"mean_speedup: {format_optional(comparison['summary']['mean_speedup'])}")
    print(f"comparison: {output}")
    return 0


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid history JSON on line {line_number}: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def cmd_report(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = load_json(config_path)
        validate_scope(config)
        workspace = resolve_workspace(config_path, config)
        run_root = resolve_run_root(workspace, config)
        history = load_history(run_root / "history.jsonl")
        baseline = maybe_load_json(run_root / "baseline.json") or {}
        best = maybe_load_json(run_root / "best.json") or {}
    except ValueError as exc:
        return fail(str(exc))

    operator = config_section(config, "operator")
    target = config_section(config, "target")
    lines = [
        "# Ascend C Optimization Report",
        "",
        "## Contract",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Operator | {operator.get('name', 'N/A')} |",
        f"| SoC | {target.get('soc_version') or target.get('soc_family', 'Ascend910B')} |",
        f"| CANN | {target.get('cann_version', '9.0.0')} |",
        f"| Baseline run | {baseline.get('run_dir', 'N/A')} |",
        f"| Best candidate | {best.get('candidate', 'N/A')} |",
        "",
        "## Candidate History",
        "",
        "| Candidate | Hypothesis | Build | Precision | Speedup | Accepted |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in history:
        lines.append(
            f"| {row.get('candidate', '')} | {row.get('hypothesis', '')} | "
            f"{row.get('build', '')} | {row.get('precision', '')} | "
            f"{format_optional(row.get('speedup'))} | {row.get('accepted', False)} |"
        )
    if not history:
        lines.append("| N/A | No evaluated candidates | N/A | N/A | N/A | N/A |")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- State directory: `{run_root}`",
            f"- Baseline: `{run_root / 'baseline.json'}`",
            f"- Best: `{run_root / 'best.json'}`",
            f"- History: `{run_root / 'history.jsonl'}`",
            "",
            "Complete the final data path, resource budget, precision summary, and remaining bottleneck from the accepted design and run evidence.",
        ]
    )
    output = Path(args.output).expanduser().resolve() if args.output else run_root / "reports" / "final-report.md"
    write_text(output, "\n".join(lines) + "\n")
    print(f"report: {output}")
    return 0


def cmd_profile_analyze(args: argparse.Namespace) -> int:
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    config: dict[str, Any] = {}
    if args.config:
        try:
            config = load_json(Path(args.config).expanduser().resolve())
        except ValueError as exc:
            return fail(str(exc))
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else profile_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        analysis = analyze_profile(profile_dir, config, output_dir)
    except Exception as exc:
        return fail(str(exc))
    print(f"status: {analysis.get('status', 'unknown')}")
    print(f"profile_analysis: {output_dir / 'profile_analysis.json'}")
    print(f"profile_summary: {output_dir / 'profile_summary.md'}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    root = Path(args.skill_dir).expanduser().resolve() if args.skill_dir else Path(__file__).resolve().parents[1]
    checks = [
        ("SKILL.md", root / "SKILL.md"),
        ("agents/openai.yaml", root / "agents" / "openai.yaml"),
        ("assets/operator-spec.example.yaml", root / "assets" / "operator-spec.example.yaml"),
        ("assets/benchmark-cases.example.jsonl", root / "assets" / "benchmark-cases.example.jsonl"),
        ("assets/optimization-report-template.md", root / "assets" / "optimization-report-template.md"),
        ("assets/project-scaffold/CMakeLists.txt", root / "assets" / "project-scaffold" / "CMakeLists.txt"),
        ("assets/operator-templates/elementwise/manifest.yaml", root / "assets" / "operator-templates" / "elementwise" / "manifest.yaml"),
        ("references/INDEX.md", root / "references" / "INDEX.md"),
        ("references/workflow-contract.md", root / "references" / "workflow-contract.md"),
        ("references/environment-cann-900.md", root / "references" / "environment-cann-900.md"),
        ("references/hardware-ascend910b.md", root / "references" / "hardware-ascend910b.md"),
        ("references/operator-classification.md", root / "references" / "operator-classification.md"),
        ("references/api-selection.md", root / "references" / "api-selection.md"),
        ("references/tiling-general.md", root / "references" / "tiling-general.md"),
        ("references/memory-and-alignment.md", root / "references" / "memory-and-alignment.md"),
        ("references/data-movement.md", root / "references" / "data-movement.md"),
        ("references/pipeline-scheduling.md", root / "references" / "pipeline-scheduling.md"),
        ("references/host-code-rules.md", root / "references" / "host-code-rules.md"),
        ("references/kernel-code-rules.md", root / "references" / "kernel-code-rules.md"),
        ("references/precision-validation.md", root / "references" / "precision-validation.md"),
        ("references/profiling-metrics.md", root / "references" / "profiling-metrics.md"),
        ("references/optimization-playbook.md", root / "references" / "optimization-playbook.md"),
        ("references/failure-patterns.md", root / "references" / "failure-patterns.md"),
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
    parser = argparse.ArgumentParser(description="Deterministic Ascend C 910B operator optimization harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize .ascendc-opt state in an operator workspace")
    init.add_argument("--workspace", required=True)
    init.add_argument("--spec", required=True, help="JSON or JSON-compatible YAML operator specification")
    init.add_argument("--cases", default="", help="optional benchmark JSONL file")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    for phase in ("env", "build", "precision", "benchmark", "profile"):
        phase_parser = subparsers.add_parser(phase, help=f"run only the {phase} phase")
        phase_parser.add_argument("--config", required=True)
        phase_parser.add_argument("--label", default=phase)
        phase_parser.set_defaults(func=lambda args, selected=phase: run_single_phase(selected, args))

    baseline = subparsers.add_parser("baseline", help="run build/correctness/benchmark and save baseline")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--label", default="baseline")
    baseline.set_defaults(func=lambda args: run_harness("baseline", args))

    evaluate = subparsers.add_parser("evaluate", help="run candidate evaluation and compare with baseline")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--label", required=True)
    evaluate.add_argument("--hypothesis", default="")
    evaluate.add_argument("--change", action="append", default=[])
    evaluate.set_defaults(func=lambda args: run_harness("evaluate", args))

    compare = subparsers.add_parser("compare", help="compare a benchmark JSON result with the saved baseline")
    compare.add_argument("--config", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline", default="")
    compare.add_argument("--output", default="")
    compare.set_defaults(func=cmd_compare)

    report = subparsers.add_parser("report", help="render a final report from state records")
    report.add_argument("--config", required=True)
    report.add_argument("--output", default="")
    report.set_defaults(func=cmd_report)

    profile_analyze = subparsers.add_parser("profile-analyze", help="analyze Ascend profiler CSV artifacts")
    profile_analyze.add_argument("--profile-dir", required=True)
    profile_analyze.add_argument("--config", default="")
    profile_analyze.add_argument("--output-dir", default="")
    profile_analyze.set_defaults(func=cmd_profile_analyze)

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



