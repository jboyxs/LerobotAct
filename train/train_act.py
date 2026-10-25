"""从 JSON 配置启动 ALOHA Transfer Cube ACT 训练，默认使用提速配置。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import prod
import os
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config_act_fast.json")


def get_nested(config: dict, key: str):
    value = config
    for part in key.split("."):
        value = value[part]
    return value


def set_nested(config: dict, key: str, value) -> None:
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def check_shared_memory(config: dict, dataset_root: Path, parser: argparse.ArgumentParser) -> None:
    """检查当前 RGB uint8 预取窗口；估算不包含动作张量和 IPC 等额外开销。"""
    if config["num_workers"] == 0 or not Path("/dev/shm").is_dir():
        return
    metadata = json.loads((dataset_root / "meta/info.json").read_text())
    image_bytes = sum(
        prod(feature["shape"])
        for feature in metadata["features"].values()
        if feature.get("dtype") == "video"
    )
    required = image_bytes * config["batch_size"] * config["num_workers"] * config["prefetch_factor"]
    stat = os.statvfs("/dev/shm")
    available = stat.f_bavail * stat.f_frsize
    mib = 1024**2
    if required > available:
        parser.error(
            f"图像预取窗口估算需要 {required / mib:.1f} MiB，共享内存 /dev/shm "
            f"仅可用 {available / mib:.1f} MiB（尚未包含其他张量和 IPC 开销）。"
            "请降低 --num-workers / --prefetch-factor，或使用 --num-workers 0；"
            "如需保持高并发，应在容器启动时增大 --shm-size。"
        )
    print(f"共享内存预检：图像预取约 {required / mib:.1f} MiB，"
          f"/dev/shm 可用 {available / mib:.1f} MiB；仍需为其他开销留余量。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="LeRobot JSON 配置文件，默认 train/config_act_fast.json；其余选项覆盖文件中的值",
    )
    # 不设默认值，避免用脚本默认参数覆盖所选 JSON 中的配置。
    integer_options = {
        "steps": "steps",
        "batch-size": "batch_size",
        "seed": "seed",
        "save-freq": "save_freq",
        "log-freq": "log_freq",
        "env-eval-freq": "env_eval_freq",
        "eval-episodes": "eval.n_episodes",
        "eval-batch-size": "eval.batch_size",
        "num-workers": "num_workers",
        "prefetch-factor": "prefetch_factor",
    }
    for option in integer_options:
        parser.add_argument(f"--{option}", type=int, default=None, help="覆盖配置文件中的对应值")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="只显示命令，不启动训练")
    args = parser.parse_args()

    config_path = project_path(args.config)
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"无法读取 JSON 配置 {config_path}：{exc}")
    if not isinstance(config, dict):
        parser.error("配置文件的顶层必须是 JSON 对象")

    overrides = {}
    for option, key in integer_options.items():
        value = getattr(args, option.replace("-", "_"))
        if value is not None:
            overrides[key] = value
    for key, value in (
        ("policy.device", args.device),
        ("policy.use_amp", args.use_amp),
        ("persistent_workers", args.persistent_workers),
    ):
        if value is not None:
            overrides[key] = value

    effective = deepcopy(config)
    try:
        for key, value in overrides.items():
            set_nested(effective, key, value)
        # JSON 内相对数据/输出路径以项目根目录为基准。
        # 命令行传入的相对路径以调用者工作目录为基准。
        dataset_root = (
            args.dataset_root.expanduser().resolve() if args.dataset_root is not None
            else project_path(get_nested(effective, "dataset.root"))
        )
        output_dir = (
            args.output_dir.expanduser().resolve() if args.output_dir is not None
            else project_path(effective["output_dir"])
        )
        for key in ("steps", "batch_size", "eval.n_episodes", "eval.batch_size", "prefetch_factor"):
            if get_nested(effective, key) <= 0:
                parser.error(f"{key} 必须大于 0")
        for key in ("num_workers", "env_eval_freq", "log_freq"):
            if get_nested(effective, key) < 0:
                parser.error(f"{key} 不能小于 0")
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"配置字段缺失或类型错误：{exc}")
    if not (dataset_root / "meta/info.json").is_file():
        parser.error(f"未找到本地数据集的 meta/info.json：{dataset_root}")

    overrides["dataset.root"] = str(dataset_root)
    overrides["output_dir"] = str(output_dir)
    if args.output_dir is not None:
        overrides["job_name"] = output_dir.name
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={config_path}",
        *[f"--{key}={json.dumps(value) if not isinstance(value, str) else value}"
          for key, value in overrides.items()],
    ]
    print(f"MUJOCO_GL=egl {shlex.join(command)}", flush=True)
    if not args.dry_run:
        check_shared_memory(effective, dataset_root, parser)
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
