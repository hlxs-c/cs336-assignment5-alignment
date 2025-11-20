import torch
import numpy as np

from typing import Callable, Literal
from einops import rearrange

def compute_group_normalized_rewards(
  reward_fn: Callable[[str, str], dict[str, float]],
  rollout_responses: list[str],
  repeated_ground_truths: list[str],
  group_size: int,
  advantage_eps: float,
  normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
  """
  Compute rewards for each group of rollout responses, normalized by the group size.
  Args:
    reward_fn (Callable[[str, str], dict[str, float]): Scores the rollout responses against the ground truths,
      producing a dict with keys "rewards", "format_reward", and "answer_reward".
    rollout_responses (list[str]): Rollouts from the policy. 
      The length of this list is `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
    repeated_ground_truths (list[str]): The ground truths for the examples.
      The length of this list is `rollout_batch_size`, because the ground truth for each example is repeated `group_size` times.
    group_size (int): Number of reponses per question (group).
    advantage_eps (float): Small constant to avoid division by zero in normalization.
    normalize_by_std (bool): If True, divide by the per-group standard deviation; otherwise subtract only the group mean.
  Returns:
    tuple[torch.Tensor, torch.Tensor, dict[str, float]]: 
      advantages (torch.Tensor): shape (rollout_batch_size). Group-normalized rewards for each rollout response.
      raw_rewards (torch.Tensor): shape (rollout_batch_size). Unnormalized rewards for each rollout response. 
      metadata (dict[str, float]): other statistics to log. (e.g. mean, std, max/min of rewards).    
  """
  groups = len(rollout_responses) // group_size
  # 1. 遍历所有的（rollout_response, ground_truth）对，使用 reward_fn 对其进行打分（reward）
  raw_rewards = []
  for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths):
    reward_res = reward_fn(rollout_response, ground_truth)
    raw_rewards.append(reward_res["reward"])
  
  # 2. 将 raw_rewards 转换为张量并变换形状为(groups, group_size)
  raw_rewards = torch.tensor(raw_rewards) # shape: (rollout_batch_size,)
  raw_rewards = rearrange(raw_rewards, "(groups group_size) -> groups group_size", groups=groups, group_size=group_size)  # shape: (groups, group_size)

  # 3. 计算统计值
  group_mean_rewards = raw_rewards.mean(dim=-1, keepdim=True)
  group_std_rewards = raw_rewards.std(dim=-1, keepdim=True)

  # 4. 计算优势advantage并进行标准化
  advantages = raw_rewards - group_mean_rewards  # shape: (groups, groups_size)
  if normalize_by_std:
    advantages = advantages / (group_std_rewards + advantage_eps)

  metadata = {
    "mean": group_mean_rewards,
    "std": group_std_rewards,
    "max": raw_rewards.max(dim=-1, keepdim=True),
    "min": raw_rewards.min(dim=-1, keepdim=True)
  }

  # 5. 最后将 raw_advantages 和 advantages 的形状变换回 (rollout_batch_size,)
  raw_rewards = rearrange(raw_rewards, "groups group_size -> (groups group_size)", groups=groups, group_size=group_size)
  advantages = rearrange(advantages, "groups group_size -> (groups group_size)", groups=groups, group_size=group_size)

  # return advantages, raw_rewards, metadata
  return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
  raw_rewards_or_advantages: torch.Tensor,
  policy_log_probs: torch.Tensor,
) -> torch.Tensor:
  """
  Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either the raw reward or an already-normalized advantage.
  Args:
    raw_rewards_or_advantages (torch.Tensor): Shape (batch_size, 1), scalar reward/advantage for each rollout response.
    policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), logprobs for each token.
  Returns:
    torch.Tensor: Shape (batch_size, sequence_length), the per-token policy-gradient loss 
      (to be aggregated across the batch and sequence dimensions in the training loop).
  
  Note:
    With question `q`, response `o`, and response token `o_t`, the naive per-token policy gradient loss = 
      -A_t · log p_{\theta}(o_t|q, o_{<t})
  """
  return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
  advantages: torch.Tensor,
  policy_log_probs: torch.Tensor,
  old_log_probs: torch.Tensor,
  cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """
  Compute the per-token GRPO-Clip loss.
  Args:
    advantages (torch.Tensor): Shape (batch_size, 1), per-example advantages A.
    policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log probs from the policy being trained.
    old_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log probs from the old policy.
    cliprange (float): clip parameter epsilon (e.g. 0.2).
  Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]]:
      loss (torch.Tensor): Shape (batch_size, sequence_length), the per-token clipped loss.
      metadata (dict[str, torch.Tensor]): dict containing whatever you want to log.
        Suggest logging whether each token was clipped or not, i.e., whether the clipped policy gradient loss on RHS of the min was lower thah the LHS.
  Note:
    The per-token GRPO-Clip loss is:
      -min(ratio_t · A_t, clip(ratio_t, 1-epsilon, 1+epsilon) · A_t)
    where ratio_t = policy_prob_t / old_prob_t
  """
  # 1. 计算重要性比率（概率比率）
  ratio = torch.exp(policy_log_probs - old_log_probs)
  
  # 2. 对 ratio 进行裁剪
  clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

  # 3. 计算损失
  unclipped_loss = ratio * advantages
  clipped_loss = clipped_ratio * advantages

  # 4. 逐元素取最小值（保守更新）
  loss = -torch.minimum(unclipped_loss, clipped_loss)

  # 5. 记录哪些token被裁剪了
  is_clipped = clipped_ratio != ratio
  metadata = {
    "is_clipped": is_clipped,
    "clipped_fraction": is_clipped.float().mean(),
    "ratio_mean": ratio.mean(),
    "ratio_std": ratio.std()
  }

  return loss, metadata


def compute_policy_gradient_loss(
  policy_log_probs: torch.Tensor,
  loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
  raw_rewards: torch.Tensor | None = None,
  advantages: torch.Tensor | None = None,
  old_log_probs: torch.Tensor | None = None,
  cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """
  This is a convenience wrapper that dispatches to the correct loss routine (no_baseline, reinforce_with_baseline, grpr_clip)
  and returns both the per-token loss and any auxiliary statistics.
  Args:
    policy_log_probs (torch.Tensor): shape (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
    loss_type (Literal): One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
    raw_rewards (torch.Tensor | None): Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages (torch.Tensor | None): Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
    old_log_probs (torch.Tensor | None): Required for "grpo_clip"; shape (batch_size, sequence_length).
    cliprange (float | None): Required for "grpo_clip"; scalar epsilon used for clipping.
  Returns:
    loss (torch.Tensor): shape (batch_size, sequence_length), per-token loss.
    metadata (dict[str, torch.Tensor]): statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
  """
  metadata = None
  if loss_type == "no_baseline":
    assert raw_rewards is not None, "raw_rewards is required for no_baseline."
    loss = compute_naive_policy_gradient_loss(
      raw_rewards_or_advantages=raw_rewards,
      policy_log_probs=policy_log_probs
    )
  elif loss_type == "reinforce_with_baseline":
    assert advantages is not None, "advantages is required for reinforce_with_baseline."
    loss = compute_naive_policy_gradient_loss(
      raw_rewards_or_advantages=advantages,
      policy_log_probs=policy_log_probs
    )
  elif loss_type == "grpo_clip":
    assert advantages is not None, "advantages is required for grpo_clip."
    assert old_log_probs is not None, "old_log_probs is required for grpo_clip."
    assert cliprange is not None, "cliprange is required for grpo_clip."
    loss, metadata = compute_grpo_clip_loss(
      advantages=advantages,
      policy_log_probs=policy_log_probs,
      old_log_probs=old_log_probs,
      cliprange=cliprange
    )
  
  return loss, metadata


if __name__ == "__main__":
  # 仿照 test_compute_group_normalized_rewards 编写了以下测试代码用于调试（最终错误在于 np.std 和 torch.std 的默认计算区别）
  def dummy_reward_fn(response, ground_truth):
    import hashlib
    # Use SHA-256 which is deterministic
    response_hash = int(hashlib.sha256(response.encode()).hexdigest(), 16)
    reward = (response_hash % 10) / 10.0
    return {
        "reward": reward,
        "format_reward": reward,
        "answer_reward": reward,
    }
  num_rollout_responses = 8
  rollout_responses = [f"hmm I think ths answer is {i}" for i in range(num_rollout_responses)]
  repeated_ground_truths = ["42"] * num_rollout_responses
  advantage_eps = 1e-6
  group_size = int(num_rollout_responses / 2)

  advantages, raw_rewards, metadata = compute_group_normalized_rewards(
    reward_fn=dummy_reward_fn,
    rollout_responses=rollout_responses,
    repeated_ground_truths=repeated_ground_truths,
    group_size=group_size,
    advantage_eps=advantage_eps,
    normalize_by_std=True
  )

  print(f'---------------- raw_rewards ---------------------')
  print(raw_rewards)
  print(f'---------------- advantages ---------------------')
  print(advantages)