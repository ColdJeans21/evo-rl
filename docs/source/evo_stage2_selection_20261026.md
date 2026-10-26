# SO-101 Evo-RL 第二阶段筛选训练记录（2026-10-26）

本轮选择论文式数据筛选方案（方案 B），而不是把 ACP indicator 注入策略 prompt。
Value 模型只负责计算价值与优势；策略继续训练时使用显式片段选择和分来源加权采样。

## 固定输入

- 来源数据：`/home/xiaozhi/so101_collect_data/success_valueModelTrain`
- 独立工作副本：`/home/xiaozhi/so101_collect_data/stage2_evo_pi05_r1_v008k`
- Value checkpoint：`outputs/value_train/pistar06_pi05_time_takeover_ModelValueTrain_v1/checkpoints/008000/pretrained_model`
- 初始策略：`outputs/train/pi05_base1_bc57_v1/checkpoints/030000/pretrained_model`
- 评分对象：57 条示范（合并 episode 0--56）和 7 条纯自主成功轨迹（57、65、66、75、79、91、94）
- 人工介入成功轨迹不参与优势排序，只提取其中 `is_intervention=true` 的连续人工纠正片段。

原始 `success_valueModelTrain` 不被修改。Value 推理只向独立工作副本写入以下字段：

- `complementary_info.value_evo_r1_v008k_n20_top10`
- `complementary_info.advantage_evo_r1_v008k_n20_top10`
- `complementary_info.acp_indicator_evo_r1_v008k_n20_top10`

## 选择规则

- `n_step=20`，每个被选起点对应向前 20 帧窗口（0.8 秒，25 FPS）。
- 在每条可评分轨迹内选 top 10% 且严格大于 0 的 advantage 起点。
- 重叠或相邻窗口合并；窗口不会越过 episode 末端。
- 57 条示范同时保留完整基础池，并把高优势片段加入增强池。
- 7 条纯自主成功轨迹只使用高优势片段。
- 39 条成功介入轨迹只使用连续人工控制片段；策略自主部分不进入该池。
- 片段尾部不足 PI0.5 50-step action horizon 的动作由 `action_is_pad` 屏蔽，不向未选中的后续动作泄漏监督。

生成的可审计计划：
`outputs/value_infer/pi05_evo_r1_v008k_n20_top10/selection_plan.json`。
其中保存源数据哈希、字段名、帧范围、池统计和选择参数。

## 实际数据池

Value 推理成功完成 1,910 batches；预测值范围 `[-0.864147, 0]`，均值 `-0.214676`，
标准差 `0.150106`。要求正 advantage 后，实际起点选择比例为 `0.1000`。

| 数据池                         | episode 数 | 合并后片段数 | 有效帧数 | 每次抽样概率 |
| ------------------------------ | ---------: | -----------: | -------: | -----------: |
| 完整示范 `demo_base`           |         57 |           57 |   54,113 |          50% |
| 连续人工纠正 `intervention`    |         39 |           52 |   13,792 |          25% |
| 示范高优势增强 `demo_selected` |         57 |          614 |   21,402 |        12.5% |
| 自主成功高优势 `auto_selected` |          7 |           92 |    3,069 |        12.5% |

示范与自主池分别产生 5,498 和 613 个正优势起点。采样使用有放回的
`WeightedRandomSampler`；池内每个有效帧等概率，四个池的总概率严格按上表设置。
每个虚拟 epoch 抽取 120,582 个样本，与源合并数据总帧数一致。

## 策略训练参数

- policy：从 PI0.5 base1 的 30,000-step checkpoint 继续训练
- `batch_size=8`
- `steps=15000`
- `optimizer_lr=1e-5`
- warmup 500 steps，余弦衰减至 `1e-6`
- `gradient_checkpointing=true`
- `train_expert_only=true`
- `mask_action_padding_loss=true`
- 不使用图像增强
- ACP prompt conditioning 关闭
- 每 5,000 steps 保存本地 checkpoint
- W&B project：`evo-rl-pi05-stage2`，artifact 上传关闭
- Hugging Face 推送关闭

执行入口：

```bash
# 默认只检查并打印命令
scripts/run_stage2_evo_selection.sh train

# 实际执行
scripts/run_stage2_evo_selection.sh train --execute
```

完整流水线也可分别执行 `prepare`、`infer`、`select`、`train`，每个写操作阶段均需显式
`--execute`。已有输出时脚本拒绝覆盖。

## 验证与运行状态

1-step 真实 GPU 冒烟测试已通过：loss `0.085`、梯度范数 `2.156`，峰值 allocated
显存 `11.17 GiB`，batch 8 在 16 GiB GPU 上有安全余量。

正式训练输出：`outputs/train/pi05_base2_selective_evo_r1_v008k_n20_top10`。
W&B run id：`c0b1wdut`。前 50 steps：loss `0.082`、梯度范数 `1.854`、
学习率 `5.3e-7`（处于 warmup），峰值 allocated 显存 `11.17 GiB`，无 NaN 或 OOM。

代码测试覆盖配置校验、选择片段边界屏蔽和采样池概率；PI0.5/训练相关测试亦通过。
