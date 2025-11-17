import torch
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_sequence

from transformers import  PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.utils.softmax import log_softmax

def tokenize_prompt_and_output(
  prompt_strs: list[str], 
  output_strs: list[str], 
  tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
  """
  Tokenize the prompt and output strings, 
  and counstruct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding.)
  Args:
    prompt_strs (list[str]): List of prompt strings.
    output_strs (list[str]): List of output strings.
    tokenizer (PreTrainedTokenizer): Tokenizer to use for tokenization.
  
  Returns:
    dict[str, torch.Tensor]: Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings.
      Then the returned dictionary have the following keys:
      
      input_ids (torch.Tensor): shape (batch_size, max(prompt_and_output_lens) - 1), 
        the tokenized prompt and output strings, with the final token sliced off.
      labels (torch.Tensor): shape (batch_size, max(prompt_and_output_lens) - 1),
        shifted input ids, i.e., the input ids without the first token.
      response_mask (torch.Tensor): shape (batch_size, max(prompt_and_output_lens) - 1),
        a mask on the response tokens in the labels.
  """
  # 1.编码为token IDs
  prompt_token_idss = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs]
  output_token_idss = [tokenizer.encode(output, add_special_tokens=False) for output in output_strs]

  # 2.计算长度
  prompt_lens = [len(ids) for ids in prompt_token_idss]
  output_lens = [len(ids) for ids in output_token_idss]

  # 3.拼接和填充
  concat_sequences = [p + o for p, o in zip(prompt_token_idss, output_token_idss)]
  concat_tensors = [torch.tensor(seq, dtype=torch.long) for seq in concat_sequences]
  padded_sequences = pad_sequence(concat_tensors, batch_first=True, padding_value=tokenizer.pad_token_id) # 注意：这里应该使用 tokenizer.pad_token_id 来进行填充，而不是0.

  # 4.创建模型输入和对应的标签
  input_ids = padded_sequences[:, :-1]
  labels = padded_sequences[:, 1:]

  # 5.创建掩码
  batch_size, seq_len = input_ids.shape
  response_mask = torch.zeros_like(input_ids)

  for i, (p_len, o_len) in enumerate(zip(prompt_lens, output_lens)):
    start = p_len - 1
    end = min(start + o_len, seq_len)
    response_mask[i, start:end] = 1

  return {
    "input_ids": input_ids,
    "labels": labels,
    "response_mask": response_mask
  }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
  """
  Get the entropy of the next-token predictions (i.e., entropy over the vocabulary diemsion).
  Args:
    logits (torch.Tensor): Tensor of shape (batch_size, sequence_length, vocab_size)
      containing unnormalized logits.
  Returns:
    torch.Tensor, shape (batch_size, sequence_length). The entropy for each next-token prediction.
  Note:
    Entropy (H(p)) = - sum_x p(x)*log p(x)
  """
  # 1.使用 log_softmax 计算数值稳定性的对数概率: log(p(x))
  log_probs = log_softmax(x=logits, dim=-1)
  # 2.通过exp操作还原概率 p(x)
  probs = log_probs.exp()
  # 3.计算熵: 计算每个元素的 p(x) * log(p(x))，然后求和并取负数
  entropy = -torch.sum(probs * log_probs, dim=-1)
  return entropy


def get_response_log_probs(
  model: PreTrainedModel,
  input_ids: torch.Tensor,
  labels: torch.Tensor,
  return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
  """
  Args:
    model (PreTrainedModel): HuggingFace model used for scoring 
      (placed on the correct device and in inference mode if gradients should not be computed.)
    input_ids (torch.Tensor): shape (batch_size, sequence_length), concatenated prompt + response token as produced by tokenization method.
    labels (torch.Tensor): shape (batch_size, sequence_length), labels as produced by tokenization method.
    return_token_entropy (bool): If True, also return per-token entropy by calling compute_entropy.
  Returns:
    dict[str, torch.Tensor]:
      "log_probs": shape (batch_size, sequence_length), conditional log-probabilities
      "token_entropy": optional, shape (batch_size, sequence_length), per-token entropy for each position (present only if return_token_entropy=True)
  """
  # 1.获取模型的logits输出
  logits = model(input_ids).logits  # shape [batch_size, sequence_length, vocab_size]
  
  # 2.计算log_softmax得到对数概率
  log_probs = log_softmax(logits, dim=-1) # shape [batch_size, sequence_length, vocab_size]

  # 3.使用gather选择标签对应位置的概率，并压缩最后一个维度
  log_probs = torch.gather(input=log_probs, dim=-1, index=labels.unsqueeze(-1)) # shape [batch_size, sequence_length]
  
  # 4.计算熵（如果需要）
  token_entropy = compute_entropy(logits=logits) if return_token_entropy else None

  return {
    "log_probs": log_probs.squeeze(-1),
    "token_entropy": token_entropy
  }


def masked_normalize(
  tensor: torch.Tensor,
  mask: torch.Tensor,
  normalize_constant: float,
  dim: int | None = None,
) -> torch.Tensor:
  """
  Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.
  Args:
    tensor (torch.Tensor): The tensor to sum and normalize.
    mask (torch.Tensor): Same shape as tensor; positions with 1 are included in the sum.
    normalize_constant (float): float the constant to divide by for normalization.
    dim: int | None the dimension to sum along before normalization. If None, sum over all dimensions.
  Returns:
    torch.Tensor: the normalized sum, where masked elements (mask == 0) don't contribute to the sum.
  """
  masked_tensor = tensor.masked_fill(mask == 0, 0)
  sum_tensor = torch.sum(masked_tensor, dim=dim)
  normalized_sum = sum_tensor / normalize_constant
  return normalized_sum


def sft_microbatch_train_step(
  policy_log_probs: torch.Tensor,
  response_mask: torch.Tensor,
  gradient_accumulation_steps: int,
  normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """
  Execute a forward-and-backward pass on a microbatch.
  Args:
    policy_log_probs (torch.Tensor): (batch_size, sequence_length), per-predicted-token log-probabilities from SFT policy beging trained.
    response_mask (torch.Tensor): (batch_size, sequence_length), `1` for response tokens, `0` for prompt/padding.
    gradient_accumulation_steps (int): Number of microbatches per optimizer step.
    normalize_constant: The constant by which to divide the sum. It is fine to leave this as 1.0.
  Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]]:
      loss (torch.Tensor): scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
      metadata (dict[str, torch.Tensor]): Dict with metadata from the underlying loss call, and any other statistics you might want to log.
  """
  # 1.计算每个token的损失，即 -log(p(x))
  per_token_loss = - policy_log_probs
  
  # 2.计算响应部分的损失：沿着序列维度求和，得到这个批次中每个样本的总响应损失
  response_loss = masked_normalize(
    tensor=per_token_loss,
    mask=response_mask,
    normalize_constant=normalize_constant,
    dim=-1
  ) # [batch_size]

  # 3.计算批次平均损失并为梯度累积做缩放
  loss = response_loss.mean() / gradient_accumulation_steps

  # 4. 反向传播
  loss.backward()

  metadata = {
    "policy_log_probs": policy_log_probs,
    "response_mask": response_mask,
    "gradient_accumulation_steps": gradient_accumulation_steps
  }

  return loss, metadata


if __name__ == "__main__":
  device = "cuda:1"

  prompts = [
    "Hello, world!",
    "This is a test.",
    "This is another test.",
  ]

  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

  tokenized_res = tokenize_prompt_and_output(prompt_strs=prompts, output_strs=prompts, tokenizer=tokenizer)
  print(tokenized_res["input_ids"])
  print(tokenized_res["labels"])
  print(tokenized_res["response_mask"])

  model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
  ).to(device)

  log_probs_res = get_response_log_probs(
    model=model,
    input_ids=tokenized_res["input_ids"].to(device),
    labels=tokenized_res["labels"].to(device),
  )

  sft_microbatch_train_step_res = sft_microbatch_train_step(
    policy_log_probs=log_probs_res["log_probs"].to(device),
    response_mask=tokenized_res["response_mask"].to(device),
    gradient_accumulation_steps=2,
    normalize_constant=1.0
  )
