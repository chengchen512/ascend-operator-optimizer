from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "ascendc-910b-operator-optimizer"
HARNESS = SKILL_DIR / "scripts" / "harness.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("ascendc_optimizer_harness", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load harness: {HARNESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SkillLayoutTests(unittest.TestCase):
    def test_frontmatter_and_flat_references(self) -> None:
        text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        frontmatter = text.split("---", 2)[1]
        keys = [line.split(":", 1)[0] for line in frontmatter.splitlines() if ":" in line]
        self.assertEqual(keys, ["name", "description"])
        self.assertIn("name: ascendc-910b-operator-optimizer", frontmatter)
        self.assertIn("Ascend 910B", frontmatter)
        self.assertIn("CANN 9.0.0", frontmatter)
        self.assertIn("sole Ascend C operator workflow", text)
        self.assertIn("Do not invoke or depend on separate AscendC", text)

        references = SKILL_DIR / "references"
        self.assertFalse(any(path.is_dir() for path in references.iterdir()))
        for filename in (
            "INDEX.md",
            "environment-cann-900.md",
            "hardware-ascend910b.md",
            "workflow-contract.md",
            "optimization-playbook.md",
            "profiling-metrics.md",
        ):
            self.assertTrue((references / filename).is_file(), filename)

    def test_openai_metadata_matches_skill(self) -> None:
        text = (SKILL_DIR / "agents" / "openai.yaml").read_text(encoding="utf-8")
        self.assertIn("Ascend C 910B Operator Optimizer", text)
        self.assertIn("$ascendc-910b-operator-optimizer", text)

    def test_skill_reference_links_resolve(self) -> None:
        for document in (SKILL_DIR / "SKILL.md", SKILL_DIR / "references" / "INDEX.md"):
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                self.assertTrue((document.parent / target).resolve().is_file(), f"{document}: {target}")

    def test_skill_cases_reference_existing_files(self) -> None:
        for case_path in sorted((REPO_ROOT / "tests" / "skill-cases").glob("*.yaml")):
            case = json.loads(case_path.read_text(encoding="utf-8"))
            for reference in case["expected_references"]:
                self.assertTrue((SKILL_DIR / "references" / reference).is_file(), reference)


class HarnessTests(unittest.TestCase):
    def test_doctor(self) -> None:
        result = subprocess.run(
            [sys.executable, str(HARNESS), "doctor"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_comparison_rejects_case_drift(self) -> None:
        harness = load_harness()
        config = {"benchmark": {"primary_metric": "candidate"}}
        baseline = {
            "unit": "us",
            "cases": [{"case": "a", "shape": [1], "dtype": "float16", "mode": "auto", "candidate": 10.0}],
        }
        candidate = {
            "unit": "us",
            "cases": [{"case": "b", "shape": [1], "dtype": "float16", "mode": "auto", "candidate": 8.0}],
        }
        with self.assertRaisesRegex(ValueError, "not comparable"):
            harness.compare_results(baseline, candidate, config)

    def test_baseline_evaluate_state_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            benchmark_script = workspace / "benchmark_stub.py"
            benchmark_script.write_text(
                "import json, os\n"
                "path = os.environ['ASCENDC_OPT_RESULT']\n"
                "latency = float(os.environ['TEST_LATENCY'])\n"
                "data = {'unit':'us','metadata':{'warmup':5,'repeat':30,'statistic':'mean'},"
                "'cases':[{'case':'main','shape':[16,256,128],'dtype':'float16','mode':'auto','candidate':latency}]}\n"
                "with open(path, 'w', encoding='utf-8') as handle: json.dump(data, handle)\n",
                encoding="utf-8",
            )
            source_spec = workspace / "source-spec.json"
            source_spec.write_text(
                json.dumps(
                    {
                        "operator": {"name": "contract_test"},
                        "target": {"soc_family": "Ascend910B", "cann_version": "9.0.0"},
                        "implementation": {"type": "ascendc", "launch_profile": "ascendc-standard"},
                        "commands": {
                            "build": "",
                            "precision": f'"{sys.executable}" -c "import sys; sys.exit(0)"',
                            "benchmark": f'"{sys.executable}" "{benchmark_script}"',
                            "profile": "",
                        },
                        "environment": {"TEST_LATENCY": "10.0"},
                        "benchmark": {
                            "primary_metric": "candidate",
                            "case_key_fields": ["case", "shape", "dtype", "mode"],
                            "metadata_fields": ["warmup", "repeat", "statistic"],
                            "noise_threshold_percent": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            init = subprocess.run(
                [sys.executable, str(HARNESS), "init", "--workspace", str(workspace), "--spec", str(source_spec)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(init.returncode, 0, init.stdout + init.stderr)
            config_path = workspace / ".ascendc-opt" / "spec.json"

            baseline = subprocess.run(
                [sys.executable, str(HARNESS), "baseline", "--config", str(config_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)

            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["environment"]["TEST_LATENCY"] = "8.0"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            evaluate = subprocess.run(
                [
                    sys.executable,
                    str(HARNESS),
                    "evaluate",
                    "--config",
                    str(config_path),
                    "--label",
                    "candidate-001",
                    "--hypothesis",
                    "reduce call overhead",
                    "--change",
                    "tile: 1 -> 2",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(evaluate.returncode, 0, evaluate.stdout + evaluate.stderr)
            state = workspace / ".ascendc-opt"
            self.assertTrue((state / "baseline.json").is_file())
            self.assertTrue((state / "best.json").is_file())
            history = [json.loads(line) for line in (state / "history.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(history), 1)
            self.assertTrue(history[0]["accepted"])
            self.assertGreater(history[0]["speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
