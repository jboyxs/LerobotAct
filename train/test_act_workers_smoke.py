"""实际训练并检查 num_workers=4 在两种预取配置下的退出行为。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERROR_PATTERN = re.compile(
    r"terminate called without an active exception|"
    r"Exception ignored in:|RuntimeError: DataLoader worker|"
    r"Traceback \(most recent call last\):|Aborted \(core dumped\)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000, help="每组实际训练步数，默认 1000")
    parser.add_argument("--output-dir", type=Path, help="测试结果目录，必须尚不存在")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps 必须大于 0")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output_dir or PROJECT_ROOT / "outputs/worker_tests" / f"{stamp}_smoke"
    root = root.expanduser().resolve()
    if not root.name.endswith("_smoke"):
        root = root.with_name(f"{root.name}_smoke")
    if root.exists():
        parser.error(f"测试目录已存在，请使用新目录：{root}")
    root.mkdir(parents=True)

    results = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    for name, persistent, prefetch in (
        ("workers4_current_smoke", False, 1),
        ("workers4_original_smoke", True, 4),
    ):
        run_dir = root / name
        log_path = root / f"{name}.log"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train/train_act_smoke.py"),
            "--num-workers", "4",
            "--persistent-workers" if persistent else "--no-persistent-workers",
            "--prefetch-factor", str(prefetch),
            "--steps", str(args.steps),
            "--save-freq", str(max(1, args.steps // 2)),
            "--output-dir", str(run_dir),
        ]
        print(f"START {name}: steps={args.steps}, log={log_path}", flush=True)
        started = time.monotonic()
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

        text = log_path.read_text(errors="replace")
        error_lines = [line for line in text.splitlines() if ERROR_PATTERN.search(line)]
        checkpoint = run_dir / "checkpoints/last"
        state_path = checkpoint / "training_state/training_step.json"
        config_path = checkpoint / "pretrained_model/train_config.json"
        saved_step = json.loads(state_path.read_text()).get("step") if state_path.is_file() else None
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        weights = checkpoint / "pretrained_model/model.safetensors"
        checkpoint_ok = saved_step == args.steps and weights.is_file() and weights.stat().st_size > 0
        config_ok = (
            config.get("num_workers") == 4
            and config.get("persistent_workers") is persistent
            and config.get("prefetch_factor") == prefetch
        )
        ended = "End of training" in text
        passed = completed.returncode == 0 and ended and checkpoint_ok and config_ok and not error_lines
        result = {
            "case": name,
            "command": command,
            "num_workers": 4,
            "persistent_workers": persistent,
            "prefetch_factor": prefetch,
            "steps": args.steps,
            "elapsed_s": round(time.monotonic() - started, 2),
            "exit_code": completed.returncode,
            "end_of_training": ended,
            "saved_step": saved_step,
            "checkpoint_ok": checkpoint_ok,
            "config_ok": config_ok,
            "error_lines": error_lines,
            "passed": passed,
            "log": str(log_path),
        }
        results.append(result)
        (root / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"{'PASS' if passed else 'FAIL'} {name}: exit={completed.returncode}, "
              f"saved_step={saved_step}, error_lines={len(error_lines)}", flush=True)

    print(f"SUMMARY {root / 'summary.json'}", flush=True)
    sys.exit(0 if all(result["passed"] for result in results) else 1)


if __name__ == "__main__":
    main()
