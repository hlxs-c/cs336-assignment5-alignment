import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """
  Compute the softmax of the input tensor along the specified dimension.
  Args:
    x (torch.Tensor): input tensor
    dim (int): dimension to apply softmax on
  Returns:
    torch.Tensor: softmax of the input tensor.
      The output tensor has the same shape as the input tensor, but specified dimension is normalized.  
  """
  # get the max value along the specified dimension
  x_max = torch.max(x, dim=dim, keepdim=True).values
  # subtract the max value from the input tensor for numerical stability
  x = x - x_max
  return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)

def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """
  计算 log_softmax 函数 (保证数值稳定性)
  Args:
    x (torch.Tensor): 输入张量
    dim (int): 沿着哪个维度计算 softmax, 默认为最后一个维度
  Returns:
    log_softmax 结果, 形状与输入相同
  数学原理：
    log_softmax(x_i) = log (e^{x_i} / (∑exp(x_j))) = x_i - log(∑exp(x_j))
    数值稳定性版本: log_softmax(x_i) = (x_i - x_max) - log(∑exp(x_j - x_max))
  """
  # 1.计算最大值（用于数值稳定性）：在指定维度上找到最大值，keepdim=True 保持维度以便广播
  x_max = torch.max(x, dim=dim, keepdim=True).values

  # 2.数值稳定化处理：减去最大值，避免指数运算时数值过大导致溢出
  stable_x = x - x_max

  # 3.计算指数，对稳定化后的值进行指数运算
  exp_x = torch.exp(stable_x)

  # 4.计算指数和：在指定维度上求和，得到每个位置的归一化分母
  sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

  # 5.计算 log-sum-exp（数值稳定的 log(∑exp(x))），使用对数恒等式 log(∑exp(x)) = x_max + log(∑exp(x - x_max))
  log_sum_exp = x_max + torch.log(sum_exp)

  # 6.计算最终的 log_softmax: log_softmax(x) = x - log(∑exp(x)) = x_i - x_max) - log(∑exp(x_j - x_max))
  return x - log_sum_exp

if __name__ == "__main__":
  # shape: (seq_len, vocab_size)
  logits = torch.Tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.1, 0.2, 0.3, 0.4, 0.5]
  ])
  print(f'logits.shape: {logits.shape}')
  # shape: (seq_len, 1)
  targets = torch.Tensor([0, 1, 3]).to(dtype=torch.int64)
  print(f'targets.shape: {targets.shape}')
  log_probs = log_softmax(x=logits, dim=-1)
  print(log_probs)

  log_probs_pytorch = torch.nn.functional.log_softmax(logits, dim=-1)
  print(log_probs_pytorch)