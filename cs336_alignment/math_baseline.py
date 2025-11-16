import os
import json

from vllm import LLM, SamplingParams
from typing import Callable

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evalute_vllm(
  vllm_model: LLM,
  reward_fn: Callable[[str, str], dict[str, float]],
  prompts: list[str],
  answers: list[str],
  eval_sampling_params: SamplingParams,
  eval_output_path: str,
) -> None:
  """
  Evaluate a language model on a list of prompts, 
  compute evaluation metrics, and serialize results to disk.
  """
  # 为每个样本生成模型输出
  outputs = vllm_model.generate(prompts=prompts, sampling_params=eval_sampling_params)

  # 所有样本的评估结果列表
  rewards = []

  for i, output in enumerate(outputs):
    # 调用 reward_fn 计算评估指标
    cur_reward = reward_fn(output.outputs[0].text, answers[i])
    rewards.append(cur_reward)
  
  # 将样本数据、模型生成结果以及对应评估分数序列化保存至磁盘
  # 确保文件目录存在
  output_dir = os.path.dirname(eval_output_path)
  os.makedirs(output_dir, exist_ok=True)
  # 构造结果并写入文件
  with open(eval_output_path, "w", encoding="utf-8") as f:
    for i in range(len(prompts)):
      data = {
        "prompt": prompts[i],
        "answer": answers[i],
        "response": outputs[i].outputs[0].text,
        "reward": rewards[i]
      }
      json.dump(data, f, ensure_ascii=False)
      f.write("\n")

def format_prompts(prompt_template_path: str, data_path: str) -> tuple[list[str], list[str]]:
  # 读取prompt模版
  with open(prompt_template_path, mode="r", encoding="utf-8") as prompt_template_f:
    prompt_template = prompt_template_f.read()
  
  prompts = []  # 模型输入的prompts
  answers = []  # 每一个prompt中question对应的answer

  # 读取数据集
  with open(data_path, mode='r', encoding='utf-8') as data_f:
    for line in data_f:
      # 去除行尾的换行符，解析当前的 JSON 对象
      data = json.loads(line.strip())

      # 提取 question 和 answer
      question = data["question"]
      answer = data["answer"]

      # 将 question 插入到prompt模板，形成输入prompt
      prompt = prompt_template.replace("{question}", question)

      prompts.append(prompt)
      answers.append(answer)
  
  return prompts, answers
      

if __name__ == "__main__":
  # prompt_template_path = "/home/op/cfc/llm/cs336-assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
  # data_path = "/home/op/cfc/llm/cs336-assignment5-alignment/data/gsm8k/test.jsonl"

  prompt_template_path = "cs336_alignment/prompts/r1_zero.prompt"
  data_path = "data/gsm8k/test.jsonl"
  
  # 加载验证集数据，并使用 r1_zero 提示次词模板将样本格式化为模型输入的prompts
  prompts, answers = format_prompts(prompt_template_path=prompt_template_path, data_path=data_path)

  # 创建VLLM模型
  vllm_model = LLM(model="Qwen/Qwen2.5-Math-1.5B")
  sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)

  # 进行评估，并将评估结果序列化至磁盘
  eval_output_path = "output/gsm8k/test_eval.jsonl"
  evalute_vllm(
    vllm_model=vllm_model,
    reward_fn=r1_zero_reward_fn,
    prompts=prompts,
    answers=answers,
    eval_sampling_params=sampling_params,
    eval_output_path=eval_output_path
  )