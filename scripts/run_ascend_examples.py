"""Run CPU-only synthetic Ascend examples with equal two-device budgets.

This command explicitly opts into synthetic profiles and nearest lookup.
Its outputs validate simulator wiring; they are not NPU performance estimates.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_examples(output_dir, selected_case="both"):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "aggregate": ["start_state=ascend_aggregate", "applications.0.scheduler=token_jsq"],
        "pd": ["start_state=ascend_pd", "applications.0.scheduler=ascend_kv"],
    }
    if selected_case != "both":
        cases = {selected_case: cases[selected_case]}
    input_count = len(read_rows(REPO / "examples" / "ascend_trace.csv"))
    manifest = {
        "evidence": "synthetic_cpu_simulation",
        "notice": "Arithmetic demo only; no Ascend device, measured latency, power or speedup claim.",
        "devices_per_case": 2,
        "cases": [],
    }
    for case, overrides in cases.items():
        case_dir = output_dir / case
        command = [
            sys.executable, "run.py", "--config-name", "config_ascend",
            "performance_model.allow_synthetic=True", "performance_model.lookup_mode=nearest",
            *overrides, f"output_dir={case_dir.as_posix()}",
        ]
        run = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        (output_dir / f"{case}.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"{case} simulation failed; see {output_dir / (case + '.log')}\n{run.stderr}")
        requests = read_rows(case_dir / "detailed" / "0.csv")
        nodes = read_rows(case_dir / "request_nodes.csv")
        if len(requests) != input_count:
            raise RuntimeError(f"{case}: completed {len(requests)}/{input_count} requests")
        flow_count = sum(row["node_type"] == "KVCacheTransfer" for row in nodes)
        if flow_count != (input_count if case == "pd" else 0):
            raise RuntimeError(f"{case}: unexpected KV transfer count: {flow_count}")
        if (case_dir / "oom.csv").exists():
            raise RuntimeError(f"{case}: simulator recorded OOM; see {case_dir / 'oom.csv'}")
        record = {
            "case": case, "command": command, "input_requests": input_count,
            "completed_requests": len(requests), "kv_transfers": flow_count,
            "mean_ttft_seconds": statistics.mean(float(row["ttft_times"]) for row in requests),
            "mean_e2e_seconds": statistics.mean(float(row["response_times"]) for row in requests),
            "output_dir": str(case_dir),
        }
        manifest["cases"].append(record)
        print(f"{case}: {len(requests)}/{input_count} requests, {flow_count} KV transfers; {case_dir}")
    (output_dir / "demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "results" / "ascend_examples")
    parser.add_argument("--case", choices=("both", "aggregate", "pd"), default="both")
    args = parser.parse_args()
    print("CPU simulation using explicitly permitted synthetic profiles and nearest lookup.")
    print("The numbers do not describe measured Ascend hardware performance.")
    run_examples(args.output_dir, args.case)


if __name__ == "__main__":
    main()
