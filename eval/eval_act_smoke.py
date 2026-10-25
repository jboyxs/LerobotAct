"""在 ALOHA 双臂传递方块仿真中执行 ACT 冒烟评估。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/train/act_transfer_smoke/checkpoints/last/pretrained_model",
        help="本地模型目录，默认使用 1,000 步训练的 last checkpoint",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--episodes", type=int, default=5, help="评估回合数，默认 5")
    parser.add_argument("--batch-size", type=int, default=1, help="并行仿真环境数，默认 1")
    parser.add_argument("--episode-length", type=int, default=400, help="每回合最多执行的环境步数")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/eval/act_transfer_smoke",
        help="评估输出目录，目录名自动补齐 _smoke 后缀",
    )
    parser.add_argument("--dry-run", action="store_true", help="只显示评估命令，不启动仿真")
    args = parser.parse_args()

    if args.episodes <= 0 or args.batch_size <= 0 or args.episode_length <= 0:
        parser.error("episodes、batch-size 和 episode-length 必须大于 0")

    policy_path = args.policy_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.name.endswith("_smoke"):
        output_dir = output_dir.with_name(f"{output_dir.name}_smoke")
    for filename in ("config.json", "model.safetensors"):
        if not (policy_path / filename).is_file():
            parser.error(f"模型目录缺少 {filename}：{policy_path}")

    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={policy_path}",
        f"--policy.device={args.device}",
        "--env.type=aloha",
        "--env.task=AlohaTransferCube-v0",
        f"--env.episode_length={args.episode_length}",
        f"--eval.n_episodes={args.episodes}",
        f"--eval.batch_size={args.batch_size}",
        f"--seed={args.seed}",
        f"--output_dir={output_dir}",
    ]
    print(f"MUJOCO_GL=egl {shlex.join(command)}", flush=True)
    if not args.dry_run:
        # 在导入 MuJoCo 前设置渲染后端，并直接传递终端信号和退出状态。
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
