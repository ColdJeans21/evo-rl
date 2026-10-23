# 2026-10-23 本地提交与数据准备记录

本次在 `main` 分支整理现有改动，并完成 Value 训练数据准备。对应可恢复标签：
`value-data-ready-20261023`。标签指向本记录和训练准备代码所在的提交。

| 提交/标签 | 提交名称 | 内容 |
| --- | --- | --- |
| `4891bc5` | `fix(processor): load checkpoints with unavailable disabled steps` | 兼容检查点中显式关闭且本机不可用的处理器步骤，并补充测试。 |
| `19e9c28` | `feat(recording): add RTC rollout and numbered SO101 collection` | RTC 异步分块、人工接管时清队列、配置检查、使用文档和 selfRoll 编号递增脚本。 |
| `b7b88fa` | `feat(training): log CUDA memory usage` | policy 训练时记录当前/峰值 allocated、reserved 显存，并写入训练指标。 |
| `value-data-ready-20261023` | `feat(data): prepare success_valueModelTrain and value training recipe` | 成功数据集构建校验脚本、batch 32 训练入口、参数及数据准备记录。 |

查看标签对应的确切提交：

```bash
git show --no-patch --format=fuller value-data-ready-20261023
```

本轮为本地提交；没有执行远程 push。数据与权重位于 Git 目录外或被忽略的输出目录中。

## 数据快照

- 名称：`success_valueModelTrain`
- 路径：`/home/xiaozhi/so101_collect_data/success_valueModelTrain`
- 103 episodes = 57 demonstrations + 46 successful rollouts；120,582 frames；25 FPS。
- `episode33` 按用户确认标为 success，合并后 `episode_index=85`。
- Manifest SHA-256：`b2af442860ac3d18cc956d7302c396f410202ecdc191b1d27db9b4e617dce00c`。
- 81 个源视频文件逐字节保留（源示范本身已经是合并数据），309 个首/中/末图像样本成功解码。
- 校验完整帧索引、逐值数据一致性、视频偏移和 source SHA-256；reward targets 构建成功。
- `time_takeover`、接管成本 150 时 task scale = 2381，目标范围 [-1, 0]。

详细解释和训练命令参见 `success_value_model_train_20261023.md`。

## 验证

以下命令结果为 `68 passed`：

```bash
/home/xiaozhi/miniconda3/envs/evo-rl/bin/pytest -q -rs \
  tests/test_recording_rtc.py \
  tests/processor/test_pipeline_from_pretrained_helpers.py \
  tests/value \
  tests/training/test_acp_dataset_stats.py \
  tests/training/test_acp_pi05_prompt_pipeline.py
```

另通过 shell 语法检查、训练命令 dry-run、训练配置解析、最终数据路径加载与 reward target 检查。
本轮没有启动正式训练。
