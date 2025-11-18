import os
import torch
import logging
import wandb
import random
import numpy as np

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, get_cosine_schedule_with_warmup

from tqdm import tqdm

from cs336_alignment.sft.dataset import SFTDataset, SFTDataCollator
from cs336_alignment.utils.sft_helper import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTTrainer():
  def __init__(self, model_path: str, data_path: str, output_dir: str, device: str = None):
    # 设置设备
    self.device = device if device is not None else "cpu"
    print(f"Using device: {self.device}")

    # 设置保存路径
    self.output_dir = output_dir
    os.makedirs(self.output_dir, exist_ok=True)

    # 开启TF32加速
    if torch.cuda.is_available():
      torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32加速

    # 加载tokenizer和模型
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 确保 pad_token 存在
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # 加载模型到指定设备
    self.policy_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,
      device_map="auto" if self.device == "cuda" else None,
      trust_remote_code=True,
      attn_implementation="flash_attention_2",  # 开启Flash Attention 2
    )

    # 开启梯度检查点
    self.policy_model.gradient_checkpointing_enable()

    # 训练配置
    self.seed = 42
    self.batch_size = 2
    self.gradient_accumulation_steps = 16  # 总的batch_size = 2 * 16 = 32
    self.max_grad_norm = 1.0
    self.learning_rate = 2e-5 # SFT 通常使用较小的学习率

    # 数据加载
    self.dataset = SFTDataset(json_file_path=data_path)
    self.collator = SFTDataCollator(self.tokenizer)
    self.dataloader = DataLoader(
      dataset=self.dataset, 
      batch_size=self.batch_size, 
      shuffle=True,
      collate_fn=self.collator,
      num_workers=4,
      pin_memory=True
    )

    # 优化器
    self.optimizer = torch.optim.AdamW(
      self.policy_model.parameters(), 
      lr=self.learning_rate
    )

    # 固定随机数种子，保证实验可重复性
    self.fix_seeds(seed=self.seed)


  def train(self, num_epochs: int = 1):
    # 初始化wandb
    wandb.init(project="cs336-assignment5", config={
      "model": "Qwen2.5-Math-1.5B",
      "seed": self.seed,
      "batch_size": self.batch_size,
      "gradient_accumulation_steps": self.gradient_accumulation_steps,
      "base_learning_rate": self.learning_rate,
      "epochs": num_epochs
    })

    # 设置模型为训练模式
    self.policy_model.train()
    self.optimizer.zero_grad()

    # 设置学习率调度器（余弦退火策略调度器）
    total_steps = len(self.dataloader) * num_epochs // self.gradient_accumulation_steps
    self.scheduler = get_cosine_schedule_with_warmup(
      self.optimizer,
      num_warmup_steps=int(total_steps * 0.03),
      num_training_steps=total_steps
    )

    global_step = 0
    for epoch in range(num_epochs):
      print(f'Epoch {epoch} / {num_epochs}')

      process_bar = tqdm(self.dataloader, desc=f"Epoch: {epoch + 1}")

      for inner_step, batch in enumerate(self.dataloader):
        # 移动到设备（batch 是经过 dataloader 中的 collate_fn 处理之后的）
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        response_mask = batch["response_mask"].to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
          # 前向传播，获得对数概率
          response_result = get_response_log_probs(
            model=self.policy_model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=True
          )
          policy_log_probs = response_result["log_probs"]
          entropy = response_result["token_entropy"]

          # 计算损失以及反向传播
          loss, metadata = sft_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            normalize_constant=1.0
          )

        # 计算熵指标
        # 所有token的熵（平均）
        avg_token_entropy = entropy.mean().item()

        # 响应的熵（平均）—— 只计算response部分的熵
        response_entropy = entropy * response_mask
        response_entropy = entropy * response_mask
        response_entropy_sum = response_entropy.sum()
        response_token_count = response_mask.sum()
        avg_response_entropy = (response_entropy_sum / response_token_count).item() if response_token_count > 0 else 0.0

        wandb.log({
          "train/loss": loss.item() * self.gradient_accumulation_steps,
          "train/learning_rate": self.scheduler.get_last_lr()[0],
          "train/avg_token_entropy": avg_token_entropy,
          "train/avg_response_entropy": avg_response_entropy,
        }, step=global_step)

        # 更新进度条
        process_bar.set_postfix_str({
          "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
          "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
          "entropy": f"{avg_token_entropy:.4f}",
          "resp_entropy": f"{avg_response_entropy:.4f}"
        })

        # 梯度累积步骤
        if (inner_step + 1) % self.gradient_accumulation_steps == 0:
          clip_grad_norm_(self.policy_model.parameters(), max_norm=self.max_grad_norm)
          self.optimizer.step()
          self.optimizer.zero_grad()
          self.scheduler.step()
          global_step += 1

    self.save_model(os.path.join(self.output_dir, "final_model"))
  
  def save_checkpoint(self, step: int):
    path = os.path.join(self.output_dir, f"checkpoint-{step}")
    self.save_model(path)
  
  def save_model(self, path: str):
    print(f"Saving model to {path}")
    model_to_save = self.policy_model.module if hasattr(self.policy_model, 'module') else self.policy_model
    model_to_save.save_pretrained(path)
    self.tokenizer.save_pretrained(path)

  def fix_seeds(self, seed: int = 42):
    # 1. 设置Python、Numpy随机种子
    random.seed(seed)
    np.random.seed(seed=seed)

    # 2. 设置PyTorch的随机种子
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"

  sft_trainer = SFTTrainer(
    model_path="Qwen/Qwen2.5-Math-1.5B", 
    data_path="/home/op/cfc/llm/cs336-assignment5-alignment/data/gsm8k/train.jsonl",
    output_dir="output/sft_output",
    device=device
  )

  sft_trainer.train()