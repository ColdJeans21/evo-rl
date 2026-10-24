# Value Model 历史帧实验记录（2026-10-24）

## 1. 单帧基线结论

训练输出：
`outputs/value_train/pistar06_pi05_time_takeover_ModelValueTrain_v1`。

在固定的 16 条留出轨迹上，两个 checkpoint 的离线结果为：

| checkpoint | MAE | RMSE | Pearson correlation |
| --- | ---: | ---: | ---: |
| 008000 | 0.081897 | 0.126267 | 0.802913 |
| 016000 | 0.082513 | 0.127157 | 0.799721 |

训练 loss 已明显下降，说明模型成功学到了任务进度信号；但 16000-step
相对 8000-step 没有提升留出集误差，且验证 distributional loss 从
8.345385 上升到 9.742940。后续实验因此以 008000 为初始化点。

## 2. 单帧观察混淆证据

使用 `scripts/analyze_value_observation_aliasing.py` 在留出集进行跨轨迹近邻检索。
排除每条轨迹前 100 帧后，仍能找到视觉表示相似、目标价值差异很大的状态：

- episode 92/frame 1165 与 episode 93/frame 170：目标差 0.5183，预测差 0.0025；
- episode 91/frame 260 与 episode 93/frame 130：目标差 0.4393，预测差 0.0018；
- episode 89/frame 1080 与 episode 92/frame 315：目标差 0.5556，预测差 0.0007。

008000 checkpoint 在前 20 个高混淆样本对上的预测差恢复率中位数只有约
0.38%，价值顺序正确率为 11/20。这支持“当前图像看起来相似，但任务阶段和
近期运动不同”的观察混淆假设。

诊断产物位于：
`outputs/value_analysis/single_frame_aliasing_midtrajectory_v1`。该目录包含
`REPORT.md`、结构化 JSON/CSV 和人工复核 montage，不提交模型或数据到 Git。

## 3. 历史两帧实现

Pistar06 新增可选参数 `history_frame_offsets`。默认空列表，旧 checkpoint
保持单帧行为；实验配置为 `[-25, -12]`，在 25 FPS 下分别表示 1.00 秒前、
0.48 秒前以及当前帧。

实现约束：

- 各时间点共享同一个视觉编码器；
- 先按相机聚合，再用带时间位置编码的 Transformer 融合；
- episode 边界的历史 padding 使用可学习 missing-history token；
- 采用零初始化残差门，从单帧 checkpoint 加载时预测严格不变；
- Value 离线推理自动按 checkpoint 配置加载历史帧；
- 验证日志同时记录 CE loss、MAE、RMSE 和 Pearson correlation。

真实 checkpoint 检查中，单帧 008000 模型与刚初始化的历史模型最大预测差为
`0.0`。一轮完整 forward/backward/optimizer 冒烟测试通过。RTX 5070 Ti 上
batch size 32 的峰值 allocated/reserved 显存约为 8.84/11.68 GiB。

## 4. 受控 A/B 实验

两组都从同一个 008000 checkpoint 开始，并保持数据划分、batch size、学习率、
训练步数和随机种子一致：

```bash
scripts/train_value_history_ablation.sh control --dry-run
scripts/train_value_history_ablation.sh history2 --dry-run
```

确认命令后删除 `--dry-run`。推荐先后运行两组，避免把“继续训练 8000 步”的收益
误认为历史帧的收益。主要判断指标是留出集 MAE、RMSE、相关系数，以及混淆样本
对的 gap recovery 和排序正确率；distributional CE 只作为辅助指标。
