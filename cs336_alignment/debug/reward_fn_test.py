import json

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

with open("/home/op/cfc/llm/cs336-assignment5-alignment/data/test_reward_fn.jsonl", mode="r", encoding="utf-8") as f:
  for line in f:
    data = json.loads(line.strip())
    print(f'--------------------- answer -----------------')
    print(data["answer"])
    print(f'--------------------- response ---------------')
    print(data["response"])

reward = r1_zero_reward_fn(response=data["response"], ground_truth=data["answer"])