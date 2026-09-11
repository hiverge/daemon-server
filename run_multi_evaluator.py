#!/usr/bin/env python3
"""Run a multi-evaluator by hand, mirroring the coordinator's contract.

    run_multi_evaluator.py HIVE_YAML

Reads repo.{evaluation_script,evaluation_arguments,aggregation_script} from
HIVE_YAML, runs each sub-evaluation in sequence, then feeds their results to the
aggregator. Script paths are resolved relative to the current directory, the
repo root: the evaluator and aggregator are only meaningful there. Run it from
the repo root (in a hive shell that is $REPO_DIR, /app).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    sys.exit(
        "PyYAML is required to parse the hive YAML. Install it into the shell "
        "with: pip install pyyaml"
    )

PYTHON_EXECUTABLE = os.environ.get("PYTHON_EXECUTABLE", "python3")


def canonicalize_arguments(entry):
    """A scalar is the single-argument argv `[str(scalar)]`; a list is argv."""
    if isinstance(entry, (str, int, float, bool)):
        return [str(entry)]
    return [str(arg) for arg in entry]


def run(script, args):
    """Run `python script *args` in the current directory, returning its `to_response`."""
    proc = subprocess.run(
        [PYTHON_EXECUTABLE, script, *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"status": "failed", "result": None, "error": proc.stderr.strip()}
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return {"status": "failed", "result": None, "error": "no output"}
    try:
        resp = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {"status": "failed", "result": None, "error": f"invalid JSON: {e}"}
    return {
        "status": resp.get("status"),
        "result": resp.get("result"),
        "error": resp.get("error"),
    }


def main(argv):
    if len(argv) != 1:
        sys.exit(f"usage: {sys.argv[0]} HIVE_YAML")

    hive_yaml = Path(argv[0])

    repo = yaml.safe_load(hive_yaml.read_text())["repo"]
    evaluation_script = repo["evaluation_script"]
    aggregation_script = repo.get("aggregation_script")
    raw_arguments = repo.get("evaluation_arguments") or []

    if not raw_arguments or not aggregation_script:
        sys.exit(
            "not a multi-evaluator: both repo.evaluation_arguments and "
            "repo.aggregation_script are required."
        )

    arguments = [canonicalize_arguments(entry) for entry in raw_arguments]

    results = []
    for pos, args in enumerate(arguments):
        print(f"Sub-evaluation {pos} running: {evaluation_script} {' '.join(args)}", file=sys.stderr)
        result = run(evaluation_script, args)
        print(f"Sub-evaluation {pos} {result['status']}: {result['result'] or result['error']}", file=sys.stderr)
        results.append(result)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        results_file = f.name

    print(f"Aggregating {len(results)} results.", file=sys.stderr)
    aggregated = run(aggregation_script, [results_file])
    print(json.dumps(aggregated))
    sys.exit(0 if aggregated["status"] == "success" else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
