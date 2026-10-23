# SO-101 第一轮 Value 训练准备记录（2026-10-23）

本轮完成数据准备与训练命令记录，正式训练尚未启动。用户已确定 batch size 为 32。

## 数据集

- 输出：`/home/xiaozhi/so101_collect_data/success_valueModelTrain`。
- 共 103 条成功轨迹、120,582 帧、25 FPS，约 80.39 分钟。
- 来源：`pi05_demo_57_v1` 的全部 57 条示范 + `selfRoll_episode1` 至 `selfRoll_episode52` 中的 46 条成功轨迹。
- 未纳入的 failure：14、15、22、24、34、36；原数据保留。
- 人工标记帧 13,792，论文奖励计入的 policy → human 接管事件 52 次。
- 本轮按要求收集全部成功数据，没有擅自切出验证集。正式训练若需要验证集，应按完整 episode 切分，并保存独立清单；当前训练器没有内置验证循环。

来源和索引映射、SHA-256、排除原因、校验结果均在输出目录的 `merge_manifest.json`。
视频文件逐个复制，SHA-256 与源文件相同；没有裁剪、重编码或拼接视频。
数据中的时间戳、动作、状态和接管标记逐值不变，仅合并全局索引和文件位置映射。
每个 episode 的视频偏移不变，使用 LeRobot/PyAV 读取首、中、末帧共 309 个样本成功。
真实目标构建函数已通过全部数据，task scale 为 2381，归一化目标范围为 [-1, 0]。

原始示范的 `is_intervention` 全零，保留原始记录。它们并非自主 rollout，
必须通过 manifest 的 `source_type=demonstration` 识别，不能用此字段筛选自主来源。
全零示范不会产生 policy → human 接管惩罚。

## 用户确认的标签修正

- 6、19、20、29：此前已改为 `success`，备份清单为
  `/home/xiaozhi/so101_collect_data/_label_backup_20261023/relabel_manifest.json`。
- 33：本轮按用户确认改为 `success`，备份清单为
  `/home/xiaozhi/so101_collect_data/_success_label_backup_2867vx88/relabel_manifest.json`。

`scripts/prepare_success_value_dataset.py` 可复现整理过程，拒绝覆盖已存在的输出。
在全新输出目录复现时运行：

```bash
/home/xiaozhi/miniconda3/envs/evo-rl/bin/python \
  scripts/prepare_success_value_dataset.py \
  --max-rollout=52 \
  --output-name=success_valueModelTrain_v2
```

标签已修正，无须再次传 `--confirm-success`。该选项只用于用户已经确认的轨迹。

## 参数解释

| 参数 | 本次设置的含义 |
| --- | --- |
| `reward.mode=time_takeover` | 监督目标用包含时间成本和人工接管成本的累计未来奖励构造。每个非终止帧基础奖励 -1；0→1 接管发生前的策略帧额外扣分；最后一帧奖励 0。当前实现要求完整且明确标为 success 的轨迹。 |
| `reward.takeover_cost=150` | 每次接管额外 -150，因此该帧原始奖励为 -151；不是每个人工帧都扣 150。25 FPS 下等价于 6 秒时间成本。初始即人工控制、持续人工控制、进入最终观测帧均不算新接管。 |
| `reward.intervention_field=complementary_info.is_intervention` | 指定读取哪一列识别控制来源和 0→1 转换；rollout 中 0 为策略、1 为人工。 |
| `targets.success_field=episode_success` | 从 episode 元数据的该字段读取最终任务成败；这不是逐帧标签，也不表示必须全程无人介入。 |
| `batch_size=32` | 单 GPU 每次梯度更新最多取 32 帧图像/目标对，不是 32 条轨迹。 |
| `steps=16000` | 16,000 次优化更新，不是帧数，也不是 16,000 个 epoch。约处理 512,000 个样本，相当于本轮数据约 4.25 遍；最后不满的 batch 会略微影响该估算。 |
| `num_workers=4` | 4 个 CPU 数据加载子进程负责读取/解码；不是 4 张 GPU。 |
| `log_freq=50` | 每 50 次更新记录一组训练指标及 W&B 曲线，约 320 个日志点。日志包含 loss、梯度范数、学习率及耗时，不是验证集准确率。 |
| `save_checkpoint=true` | 保存本地模型、配置、处理器和优化器等训练状态，支持后续恢复。 |
| `save_freq=4000` | 在 4,000、8,000、12,000、16,000 步保存；最终步也会保存，`last` 指向最新 checkpoint。 |
| `wandb.enable=true` | 启用 W&B 上传训练配置和指标，需要已登录。 |
| `wandb.project=evo-rl-value` | W&B 的项目分类名称；一个项目可以包含多个训练 run。 |
| `wandb.disable_artifact=true` | 禁止向 W&B 上传模型 checkpoint artifact；曲线和配置照常记录，本地 checkpoint 照常保存。 |
| `output_dir=outputs/value_train/pistar06_pi05_time_takeover_v1` | 保存本轮训练产物的本地目录，相对于仓库根目录；脚本解析为绝对路径。 |
| `job_name=pistar06_pi05_time_takeover_v1` | 实验显示名称，也用作 W&B run 名称；不会自动更改模型结构或数据选择。 |

`takeover_cost` 作用于原始奖励，随后按训练任务的 scale 统一归一化。
当前数据的额外接管成本对应 `150 / 2381 ≈ 0.0630` 的归一化 return 差值。
这是同一未来轨迹其余条件相同时的目标差异，不保证任意两帧的预测 Value/advantage 都呈固定排序。

Value 使用 `pistar06 + vision_only`，初始化 SigLIP-SO400M 并训练视觉编码器及 Value head，
不是继续训练 PI0.5 动作模型。精度 `bfloat16`，梯度检查点开启，数据图像增强关闭。
建议学习率 5e-5、warmup 1000、余弦衰减到 1e-6（16000 步）、weight decay 1e-5、梯度裁剪 10。
`value.push_to_hub=false` 保证模型不推到 Hugging Face；W&B artifact 设置与 HF 推送是两个独立选项。

16,000 步是首轮实验预算：batch 32 时的样本暴露量约等于默认 batch 64 × 8,000。
这不是收敛保证，改变 batch 与更新次数也不等价于完全相同的优化过程。

## GPU 冒烟测试记录（先前已执行）

GPU：RTX 5070 Ti，总显存 16,303 MiB。输入前视相机 1280×720，模型内部调整到 384×384。

| batch | 步数 | 结果 | 采样到的整卡显存最大值 | 稳定更新耗时 |
| --- | ---: | --- | ---: | ---: |
| 8 | 2 | 完成 | 未在更新时采样 | 约 0.39 秒/步 |
| 16 | 20 | 完成 | 8,744 MiB | 约 0.73 秒/步 |
| 32 | 20 | 完成 | 11,201 MiB | 约 1.49 秒/步 |

上表是 `nvidia-smi` 每秒采样的整卡用量，包含桌面及其他进程，不等于 PyTorch 精确瞬时峰值。
GPU 训练时还保留约 5 GiB 余量。训练预算粗估 6–7 小时，合并数据随机读视频与保存 checkpoint 会影响实际时间。

## 训练入口与验证

训练脚本：`scripts/train_success_value_model.sh`。已固定 batch=32，其余值采用上述待正式开训的配置。

```bash
# 仅查看完整命令，不启动训练
bash scripts/train_success_value_model.sh --dry-run

# 确定参数后执行正式训练
bash scripts/train_success_value_model.sh
```

脚本记录执行命令、Git revision、数据 manifest 哈希，日志保存在
`outputs/value_train/pistar06_pi05_time_takeover_v1.log`。
输出目录已存在时会停止，避免误覆盖；恢复训练应使用明确的 resume 配置。

本轮相关代码验证：RTC、processor checkpoint 兼容、Value rewards/推理和 ACP prompt 测试共 68 passed。
另完成真实数据合并全量逐值校验、视频哈希校验、309 帧加载检查及 reward target 构建。

## 提交名称记录

本轮按功能拆分提交，便于定位和恢复；具体短哈希见本目录的 `local_commit_record_20261023.md`。

1. `fix(processor): load checkpoints with unavailable disabled steps`
2. `feat(recording): add RTC rollout and numbered SO101 collection`
3. `feat(training): log CUDA memory usage`
4. `feat(data): prepare success_valueModelTrain and value training recipe`

本地采集数据、视频、权重与缓存不纳入 Git，只提交代码、脚本和准备记录。
