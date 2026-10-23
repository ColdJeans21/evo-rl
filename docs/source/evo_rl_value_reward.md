# Evo-RL Value Rewards and Advantage Selection

This repository supports the intervention-aware value targets described in
Evo-RL. The implementation is opt-in so existing value checkpoints retain
their legacy target semantics.

## Data contract

Paper-style value training requires complete, explicitly successful episodes.
Each frame must contain a binary controller-source field (by default
`complementary_info.is_intervention`):

- `0`: the deployed policy controlled the robot;
- `1`: the human operator controlled the robot.

Frames must be contiguous and numbered from zero within every episode. Failed,
partial, or unlabelled episodes are rejected instead of being silently assigned
a reward that is not defined by the paper's successful-trajectory equation.

## Reward and return

A takeover occurs on a policy-to-human transition:

```text
b[t+1] = 1 if m[t] == 0 and m[t+1] == 1 else 0
```

For each nonterminal frame, the reward is `-1 - takeover_cost * b[t+1]`.
The terminal reward is zero. Initial human control is not a takeover, staying
under human control does not add repeated penalties, and entry into the terminal
observation is not penalized (`b[T] = 0`).

Returns are normalized with a task-specific scale fitted on the value-training
episodes. The scale is saved in the value checkpoint and reused for inference;
inference refuses to refit it or silently clip an out-of-range trajectory.

## Value training

Use a success-only dataset view containing demonstrations and complete
successful intervention/autonomous trajectories:

```bash
lerobot-value-train \
  --dataset.repo_id=<VALUE_DATASET_ID> \
  --dataset.root=<VALUE_DATASET_ROOT> \
  --value.type=pistar06 \
  --value.mode=vision_only \
  --value.dtype=bfloat16 \
  --reward.mode=time_takeover \
  --reward.takeover_cost=150 \
  --reward.intervention_field=complementary_info.is_intervention \
  --targets.success_field=episode_success \
  --output_dir=outputs/value_train/<RUN_NAME> \
  --job_name=<RUN_NAME>
```

`takeover_cost` is measured in frame-cost units. At 25 FPS, the default value
of 150 assigns one takeover the same cost as six seconds of execution. Treat it
as an experiment parameter and record it with each run.

## Advantage inference for Evo-RL selection

Run advantage inference on demonstrations and successful autonomous rollouts,
not on intervention episodes. Human correction segments are added to policy
fine-tuning through their controller-source labels, independently of advantage
selection.

```bash
lerobot-value-infer \
  --dataset.repo_id=<SELECTION_DATASET_ID> \
  --dataset.root=<SELECTION_DATASET_ROOT> \
  --inference.checkpoint_path=outputs/value_train/<RUN_NAME> \
  --runtime.device=cuda \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.1 \
  --acp.require_positive_advantage=true \
  --acp.force_intervention_positive=false \
  --acp.value_field=complementary_info.value_<TAG> \
  --acp.advantage_field=complementary_info.advantage_<TAG> \
  --acp.indicator_field=complementary_info.acp_indicator_<TAG>
```

`positive_ratio=0.1` implements the paper's top-10% cutoff, while
`require_positive_advantage=true` prevents a negative value from being selected
merely because it falls in the top quantile. `n_step` must not exceed the policy
action-chunk horizon; the current PI0.5 checkpoint uses a 50-step chunk.

The legacy ACP prompt-conditioning path remains available for backward
compatibility. It is not equivalent to Evo-RL's selective SFT mixture: original
demonstrations, selected successful autonomous segments, and contiguous human
correction segments must still be sampled as separate sources when performing
the subsequent policy update.
