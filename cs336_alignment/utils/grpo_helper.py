import torch
import numpy as np

from typing import Callable
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