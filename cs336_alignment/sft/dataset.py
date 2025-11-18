import json
import torch

from torch.utils.data import Dataset, DataLoader
from typing import Callable
from transformers import PreTrainedTokenizerBase

from cs336_alignment.utils.sft_helper import tokenize_prompt_and_output

class SFTDataset(Dataset):
  def __init__(self, json_file_path: str):
    super().__init__()

    # 加载提示词模板
    with open("/home/op/cfc/llm/cs336-assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt", mode="r", encoding="utf-8") as f:
      prompt_template = f.read().replace("<think>", "")

    self.data = []
    # 加载数据
    with open(json_file_path, "r", encoding="utf-8") as f:
      for line in f:
        example = json.loads(line.strip())

        question = example["question"]
        prompt = prompt_template.replace("{question}", question)

        answer = example["answer"]
        response_parts = answer.split("#### ")
        response = f'<think> {response_parts[0]} </think> <answer> {response_parts[1]} </answer>'

        self.data.append({
          "prompt": prompt,
          "response": response
        })

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    return  self.data[index]["prompt"], self.data[index]["response"]
  
class SFTDataCollator:
  def __init__(self, tokenizer: PreTrainedTokenizerBase):
    self.tokenizer = tokenizer
  
  def __call__(self, batch: list[tuple[str, str]]):
    prompts = [example[0] for example in batch]
    outputs = [example[1] for example in batch]

    tokenized = tokenize_prompt_and_output(
      prompt_strs=prompts,
      output_strs=outputs,
      tokenizer=self.tokenizer
    )

    return tokenized
    
  
if __name__ == "__main__":
  gsm8k_dataset = SFTDataset(json_file_path="/home/op/cfc/llm/cs336-assignment5-alignment/data/gsm8k/train.jsonl")
  dataloader = DataLoader(gsm8k_dataset, batch_size=2, shuffle=True, num_workers=4)
  for idx, batch in enumerate(dataloader):
    # print(f'batch_prompts: {prompts}')
    # print(f'batch_response: {outputs}')
    prompts, outputs = batch
    print(f'type batch_prompts: {type(prompts)}')
    print(f'type batch_response: {type(outputs)}')
    print(f'response:\n {outputs}')
    break