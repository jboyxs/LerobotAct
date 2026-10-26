### 任务一：实现双臂交接物体
1. 拉取完整的现有数据集
2. 阅读训练代码
3. 实施训练
4. 验证训练结果

### 环境与数据准备

使用 Python 3.12 和 uv 安装锁定的依赖：

```bash
uv sync --locked
```

Linux 仿真需要可用的 OpenGL/EGL 和视频依赖，Ubuntu 可安装：

```bash
sudo apt-get install -y libgl1 libopengl0 libegl1 ffmpeg
```

下载本项目使用的公开数据集 v3.0：

```bash
uv run hf download lerobot/aloha_sim_transfer_cube_human \
  --repo-type dataset \
  --revision v3.0 \
  --local-dir data/aloha_sim_transfer_cube_human
```

仓库保留训练代码、JSON 配置和 `uv.lock`。本地数据集、`outputs/` 中的
checkpoint/视频/测试报告、模型权重、虚拟环境、日志和缓存均由 `.gitignore` 排除。

### ACT 冒烟训练

在项目目录运行，默认使用本地数据集、CUDA、batch size 8、单进程加载，训练 1,000 步：

```bash
uv run python train/train_act_smoke.py
```

只查看生成的命令：

```bash
uv run python train/train_act_smoke.py --dry-run
```

调整冒烟训练步数、保存频率和输出目录：

```bash
uv run python train/train_act_smoke.py \
  --steps 2000 \
  --save-freq 500 \
  --output-dir outputs/train/act_transfer_2k_smoke
```

默认数据目录为 `data/aloha_sim_transfer_cube_human`，默认输出目录为
`outputs/train/act_transfer_smoke`。LeRobot 会自动创建输出目录；新训练需使用尚不存在的输出目录。
自定义输出目录名会自动补齐 `_smoke` 后缀，已有该后缀时不会重复添加。
运行 `uv run python train/train_act_smoke.py --help` 查看可调整的参数。

#### 四 worker 退出故障复测

冒烟脚本支持调整 `--persistent-workers` / `--no-persistent-workers` 和
`--prefetch-factor`，默认仍为关闭常驻 worker、预取数 1。
例如，用当时出错的数据加载配置进行一次独立训练：

```bash
uv run python train/train_act_smoke.py \
  --num-workers 4 \
  --persistent-workers \
  --prefetch-factor 4 \
  --output-dir outputs/train/act_workers4_original_smoke
```

自动对照测试会顺序执行两组实际训练，每组默认 1,000 步、batch size 8：

```bash
uv run python train/test_act_workers_smoke.py
```

两组均使用 4 个 worker，分别测试：

- 当前配置：`persistent_workers=false`、`prefetch_factor=1`。
- 原故障配置：`persistent_workers=true`、`prefetch_factor=4`。

每次测试自动创建独立的 `outputs/worker_tests/<UTC时间戳>_smoke/` 目录，
保存两组完整日志、checkpoint 和 `summary.json`。测试同时检查进程退出码、
`End of training`、最终 checkpoint 步数、实际保存的 worker 配置，以及
`terminate called without an active exception`、`Exception ignored in:` 等错误；
即使进程退出码为 0，出现这些退出异常也会判为失败。
运行 `uv run python train/test_act_workers_smoke.py --help` 查看选项。

2026-10-25 本机实测：两组各运行 1,000 步，分别耗时 186.13 秒和 186.34 秒，
均正常退出（退出码 0）、保存到第 1,000 步，未检测到上述退出异常。
报告位于 `outputs/worker_tests/workers4_1000_smoke/summary.json`，同目录保存完整日志。
这次未复现原故障；每组仅执行一次，不能据此排除间歇性问题。冒烟脚本默认仍使用 0 个 worker。

### ACT 正式训练（JSON 配置）

`train/train_act.py` 使用 LeRobot 原生 JSON 配置，默认加载提速配置。
两份完整配置可以单独修改、保存和复现实验：

| 配置文件 | batch size | worker 数 | AMP | 输出目录 |
| --- | --- | --- | --- | --- |
| `train/config_act_baseline.json` | 8 | 4 | 关闭 | `outputs/train/act_transfer` |
| `train/config_act_fast.json` | 8 | 4 | 开启 | `outputs/train/act_transfer_fast` |

原配置固化了此前运行的参数，参考[官方同任务模型训练配置](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/blob/main/train_config.json)。
提速配置针对当前 64 MiB 共享内存限制，使用 batch size 8、4 个 worker、
`prefetch_factor=1`、`persistent_workers=false` 并启用 AMP。
原配置保留 4 个 worker、预取数 2、开启常驻 worker 和关闭 AMP。
两份配置均保留 100,000 步、seed 1000，
AdamW 学习率 `1e-5`、权重衰减 `1e-4`、梯度裁剪 10.0 和无学习率调度器。
网络保持 ResNet18、chunk size 100、KL 权重 10。
每 20,000 步保存并仿真评估一次，每次 50 回合、逐回合执行，每 200 步记录指标。
保留本地数据路径，关闭 W&B 和模型上传，使用 EGL 渲染。
旧官方配置中的 `eval_freq` 对应当前版本的 `env_eval_freq`。

选择提速配置，或者省略 `--config` 使用它作为默认值：

```bash
uv run python train/train_act.py --config train/config_act_fast.json
uv run python train/train_act.py
```

选择原配置，或预览选中的命令：

```bash
uv run python train/train_act.py --config train/config_act_baseline.json
uv run python train/train_act.py --config train/config_act_fast.json --dry-run
```

命令行优先于配置文件，可覆盖步数、batch size、worker 数、AMP、预取数等。
例如，先做一组 3,000 步验证，覆盖跨轮 worker 复用和退出阶段：

```bash
uv run python train/train_act.py \
  --config train/config_act_fast.json \
  --steps 3000 \
  --save-freq 1000 \
  --env-eval-freq 0 \
  --output-dir outputs/train/act_fast_check
```

`--no-use-amp` 可关闭 AMP，`--num-workers 0` 可使用主进程加载，
`--no-persistent-workers` 可关闭常驻 worker。
配置文件中的相对数据/输出路径，以及 `--config` 的相对路径，均以项目根目录为基准；
显式传入的 `--dataset-root` / `--output-dir` 相对路径以调用者工作目录为基准。
两套配置均从头训练，新训练需使用尚不存在的输出目录。
提速配置通过了 200 步短训练验证，尚未经过完整训练和任务成功率验证。
应同时比较吞吐、总耗时和任务成功率，不能预先保证总训练时间更短。
运行 `uv run python train/train_act.py --help` 查看全部选项。

#### 共享内存限制

本环境 `/dev/shm` 为 64 MiB，独立于系统内存和 GPU 显存。
训练管线以 uint8 传输 640×480 RGB 图像，batch size 16 的图像张量约 14.1 MiB。
原提速配置（8 个 worker、预取数 2）的图像预取窗口约 225 MiB，
会有耗尽共享内存的风险，已实际触发 `unable to allocate shared memory(shm)`。
当前提速配置（batch size 8、4 个 worker、预取数 1）将该窗口降为约 28.1 MiB。
这些估算尚不包括其他张量、在途 batch 和进程间通信开销。

实际启动时，脚本会检查图像预取窗口是否超过 `/dev/shm` 的可用容量，
超出时在启动 worker 前报告具体数值和可调整选项；`--dry-run` 仅预览命令。
如需完全绕开 DataLoader 的跨进程共享内存，可添加 `--num-workers 0`。
AMP 主要影响 GPU 计算精度，不能解决 CPU 共享内存不足。

如需恢复 8 个 worker，应由启动容器的一侧配置足够的共享内存，
例如 Docker 的 `--shm-size=2g` 或 Compose 的 `shm_size: "2gb"`，
然后通过 `--batch-size 16 --num-workers 8 --prefetch-factor 2` 进行独立测试。
当前容器缺少 `CAP_SYS_ADMIN`，无法在内部重新挂载扩容 `/dev/shm`。

2026-10-25 对照测试（每组 200 步、开启 AMP）：

| batch size | worker 数 | 预取数 | 常驻 worker | 稳定阶段吞吐 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 16 | 2 | 1 | 开启 | 约 27 样本/秒 | 正常退出 |
| 8 | 4 | 1 | 开启 | 约 54 样本/秒 | 退出时复现 `terminate called`，测试失败 |
| 8 | 4 | 1 | 关闭 | 约 54–55 样本/秒 | 正常退出，无共享内存错误 |

最终选用第三组作为当前提速配置。报告和日志位于 `outputs/worker_tests/`，
最终报告为 `fast_shm64_batch8_workers4_nonpersistent_200_smoke_summary.json`。
单次短测试不能排除间歇性退出问题；200 步未覆盖整轮数据读取后的 worker 重建。

### act_transfer_fast 正式评估

`eval/eval_act_fast.py` 专门评估 `outputs/train/act_transfer_fast/checkpoints/` 下的模型。
默认加载 `last/pretrained_model`，使用 CUDA 和 EGL，逐回合评估 50 次，
每回合最多 400 步，seed 为 123。

```bash
uv run python eval/eval_act_fast.py
uv run python eval/eval_act_fast.py --dry-run
```

默认结果保存在 `outputs/eval/act_transfer_fast/`：
`eval_info.json` 的 `overall.pc_success` 为成功率百分比，`videos/` 下保存执行视频。
可先运行少量回合，再进行完整评估，使用不同输出目录保留两次结果：

```bash
uv run python eval/eval_act_fast.py --episodes 5 --output-dir outputs/eval/act_transfer_fast_5
uv run python eval/eval_act_fast.py --episodes 50 --output-dir outputs/eval/act_transfer_fast_50
```

选择中间 checkpoint 作对照：

```bash
uv run python eval/eval_act_fast.py --checkpoint 020000
```

此时默认输出目录为 `outputs/eval/act_transfer_fast_020000`。
也可通过 `--policy-path` 直接指定一个 `pretrained_model` 目录；
该选项不能与 `--checkpoint` 同时使用。
运行 `uv run python eval/eval_act_fast.py --help` 查看全部选项。

### ACT 冒烟评估

默认加载 `act_transfer_smoke` 的 last checkpoint，使用 EGL 渲染和 CUDA，
在 ALOHA 传递方块环境中评估 5 个回合，每回合最多 400 步：

```bash
uv run python eval/eval_act_smoke.py
```

只查看生成的命令，或修改评估回合数和输出目录：

```bash
uv run python eval/eval_act_smoke.py --dry-run
uv run python eval/eval_act_smoke.py \
  --episodes 50 \
  --output-dir outputs/eval/act_transfer_50_smoke
```

使用 `--policy-path` 指定其他 checkpoint 的 `pretrained_model` 目录。
默认结果写入 `outputs/eval/act_transfer_smoke`：`eval_info.json` 包含评估指标
（`overall.pc_success` 为成功率百分比），`videos/` 下保存执行视频。
为保留历史结果，不同评估使用不同的输出目录。
自定义输出目录名同样会自动补齐 `_smoke` 后缀。
运行 `uv run python eval/eval_act_smoke.py --help` 查看全部参数。
