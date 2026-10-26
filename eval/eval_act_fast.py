"""评估 act_transfer_fast 的 ACT checkpoint，保存成功率指标和执行视频。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PROJECT_ROOT / "outputs/train/act_transfer_fast/checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--checkpoint", default="last", help="checkpoint 名称，例如 last、020000、100000")
    model.add_argument("--policy-path", type=Path, help="直接指定其他模型的 pretrained_model 目录")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--episodes", type=int, default=50, help="评估回合数，默认 50")
    parser.add_argument("--batch-size", type=int, default=1, help="并行仿真环境数，默认 1")
    parser.add_argument("--episode-length", type=int, default=400)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, help="评估输出目录，默认按 checkpoint 名称区分")
    parser.add_argument("--dry-run", action="store_true", help="只显示评估命令，不启动仿真")
    args = parser.parse_args()

    if args.episodes <= 0 or args.batch_size <= 0 or args.episode_length <= 0:
        parser.error("episodes、batch-size 和 episode-length 必须大于 0")

    if args.policy_path is not None:
        policy_path = args.policy_path.expanduser().resolve()
        run_name = f"act_transfer_fast_{policy_path.parent.name}"
    else:
        policy_path = (CHECKPOINTS / args.checkpoint / "pretrained_model").resolve()
        run_name = "act_transfer_fast" if args.checkpoint == "last" else f"act_transfer_fast_{args.checkpoint}"

    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None
        else PROJECT_ROOT / "outputs/eval" / run_name
    )
    for filename in (
        "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
    ):
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
        "--eval.use_async_envs=false",
        f"--seed={args.seed}",
        f"--output_dir={output_dir}",
    ]
    print(f"MUJOCO_GL=egl {shlex.join(command)}", flush=True)
    if not args.dry_run:
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
