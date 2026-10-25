"""使用本地 ALOHA 数据集执行 ACT 冒烟训练。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000, help="训练总步数，默认 1000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0, help="默认 0，避免多进程退出异常")
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=False,
        help="是否保留数据加载 worker，默认关闭；用于复测退出异常",
    )
    parser.add_argument("--prefetch-factor", type=int, default=1, help="每个 worker 预取的 batch 数")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "aloha_sim_transfer_cube_human",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "train" / "act_transfer_smoke",
        help="训练输出目录，目录名自动补齐 _smoke 后缀",
    )
    parser.add_argument("--dry-run", action="store_true", help="只显示训练命令，不启动训练")
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps 和 batch-size 必须大于 0")
    if args.num_workers < 0:
        parser.error("num-workers 不能小于 0")
    if args.prefetch_factor <= 0:
        parser.error("prefetch-factor 必须大于 0")

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.name.endswith("_smoke"):
        output_dir = output_dir.with_name(f"{output_dir.name}_smoke")
    if not (dataset_root / "meta" / "info.json").is_file():
        parser.error(f"未找到本地数据集的 meta/info.json：{dataset_root}")

    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        "--policy.chunk_size=100",
        "--policy.n_action_steps=100",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--persistent_workers={str(args.persistent_workers).lower()}",
        f"--prefetch_factor={args.prefetch_factor}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        "--env_eval_freq=0",
        "--wandb.enable=false",
        f"--seed={args.seed}",
        f"--output_dir={output_dir}",
        f"--job_name={output_dir.name}",
    ]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        # 替换当前进程，使训练直接接收终端信号并返回原始退出状态。
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
