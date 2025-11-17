import torch
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizerBase, AutoTokenizer

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

if __name__ == "__main__":
  prompts = [
    "Hello, world!",
    "This is a test.",
    "This is another test.",
  ]

  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

  res = tokenize_prompt_and_output(prompt_strs=prompts, output_strs=prompts, tokenizer=tokenizer)
  print(res["input_ids"])
  print(res["labels"])
  print(res["response_mask"])