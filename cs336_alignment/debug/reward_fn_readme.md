
## `r1_zero_reward_fn` 的详细工作流程
`r1_zero_reward_fn` 函数的设计目标是严格评估模型的输出，不仅要求答案正确，还要求输出的格式必须符合规定。

详细步骤：
1.**第一步：检查格式（Format Check）**：
- 函数首先检查 `response` 字符串中是否 **同时包含 `</think> <answer>` 和 `</answer>`** 这两个字符串。
- 它假定模型的输出应该是一种 “思维链（Chain-of-Thought）”的格式，即模型先输出思考过程（在 `<think></think>` 标签中），然后用 `<answer> ... </answer>` 标签包裹最终答案。
- **如果格式不满足，函数直接返回一个所有奖励都为0.0 的字典，流程结束。** 这意味着模型即使在某个地方输出了正确答案，但只要格式不正确，就得不到任何奖励。

2.**第二步：提取答案（Answer Extraction）**：
- 如果格式正确，代码会使用 `response.split("<answer>")[-1].replace("</answer>", "")` 来提取 `<answer>` 和 `</answer>` 之间的内容。这部分内容被认为是模型的最终答案 `model_answer`。

3.**处理 `\boxed{}`**：
- 接下来，代码检查提取出的 `model_answer` 是否包含 `\boxed` 命令。在学术界和数学任务中，`\boxed{}` 通常用来包含最终答案。
- 如果包含 `\boxed`，则调用 `extract_answer()` 函数，只提取 `\boxed{}` 内部的内容作为最终答案。
- **一个重要的失败情况是**：如果 `model_answer` 中有 `\boxed`，但 `extract_answer()` 函数提取失败（例如，括号不匹配 `\boxed{123`），那么函数会认为是一个格式错误，并返回0分。

4.**第四步：准备标准答案（Ground truth Preparation）**：
- 代码会检查 `ground_truth` 的数据类型。它可以是数字 （`int` 或 `float`），也可以是字符串（`str`），甚至可以是一个列表（`list`）——列表中的每个元素都是一个可能的正确答案（例如 `["1/2", "0.5"]`）。
- 如果 `ground_truth` 是数字，它会被转换为字符串以便后续比较。

5.**第五步：评分（Grading）**：
- 这是最核心的步骤。代码调用 `grade(model_answer, ground_truth, fast)` 函数。
- `grade` 函数会用多种策略（字符串比较、符号计算比较等）来判断 `model_answer` 和 `ground_truth` 是否等价。
- 如果 `ground_truth` 是一个列表， `grade` 函数会逐一比较 `model_answer` 和列表中的每一个正确答案，只要和其中任何一个匹配， `is_correct` 就为 `True`。

6.**第六步：返回奖励（Return Reward）**：
-  **如果 is_correct 为 True（答案正确）**:
    - 函数返回 `{"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}`。这表示格式正确，答案也正确，总奖励为 `1.0`。
- **如果 is_correct 为 False（答案错误）**:
    - 函数返回 `{"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}`。注意，这里的 `format_reward` 是 `1.0`，因为它成功通过了第一步的格式检查。但这是一种“安慰分”，最终的总奖励 `reward` 依然是 `0.0`。这种设计有助于区分“格式正确但答案错误”和“格式都错误”这两种情况，便于更细致地分析模型性能。

#### 响应（response）和标准答案（ground truth）的格式要求
想要利用 `r1_zero_reward_fn` 来评估响应是否正确，`response` 和 `ground_truth` 都需要满足一定的格式：

#### Response（模型输出）的正确格式
`response` 必须遵循一个严格的模板：
```python
<think>思考过程</think> <answer>最终答案</answer>
```
其中 `<think>` 会作为模型输入的 `prompt`的最后一个词，用于提示模型按照先思考再回答的推理过程来进行回答。

- 必须包含 `</think> <answer>` 和 `</answer>` 标签包裹。
- `最终答案` 部分通常推荐使用 `\boxed{}` 来包裹，例如 `\boxed{42}` 或 `\boxed{\frac{1}{2}}`，虽然不强制，但这是比较稳妥的做法。

正确格式的例子（模型的输出）：
```python
I need to calculate the sum of 10 and 32. 10 + 32 = 42. So the final answer is 42. </think> <answer>\boxed{42}</answer>
```

错误格式的例子：
- The answer is 42. (没有标签)
- `</think> The answer is \boxed{42}` (没有 `<answer>` 标签)
- `<answer>\boxed{42}</answer>` (没有 `</think>`)

#### Ground Truth（标准答案）的格式
ground_truth 的格式比较灵活，可以是以下几种类型：
1.**字符串 (str)**：最常见的格式，可以是数字、分数、LaTeX 表达式等。
- `'42'`
- `'\\frac{1}{2}'`
- `'x=5'`
2.**整数 (int) 或浮点数 (float)**：简单的数值答案。
- 42
- 0.5
3.**字符串列表 (list[str])**: 当一个问题有多个等价的正确答案时使用。
- `['0.5', '\\frac{1}{2}']`
- `['(4, 5)', '[4, 5]']` (例如，一个区间可以用不同括号表示)

#### Ground Truth的格式注意点
**标准答案（ `ground_truth`）中可以包含一些额外的字符，但有严格的限制。该评估代码只能处理预先定义好的、与数学答案紧密相关的附加文本（主要是单位和一些格式化词语）。**

`r1_rewawrd_fn` 中的评估核心是 **标准化和清洗（Normalization & Cleaning）**。它使用一系列预定义的规则和列表，用来删除和替换字符串中的特定部分。但它并不能“读懂”一句话，然后提取出里面的答案。

例如对于以下例子：
```python
ground_truth = "Therefore jack has 2 apple."
response = "xxx</think> <answer> 2 </answer>"
```
在这种情况下，**评估会失败，即判定response中的答案为错误**。

在上述例子中， `ground_truth`会被以下过程处理：
1. `grad()` 函数会调用 `grade_answer_mathd()` 或 `grade_answer_sympy()`
2. 这些函数会调用标准化函数，比如 `math_normalize_answer()`，它内部会调用 `_strip_string()`。
3. `_strip_string()` 函数会遍历 `unit_texts` 列表。这个列表里恰好包含了 `"apple"` 和 `"apples"`。
4. 因此，`"Therefore jack has 2 apple."` 中的 `"apple"` 会被移除，字符串可能变成 `"Therefore jack has 2 ."`。
5. 接下来，代码会移除空格，字符串会变为 `"Thereforejackhas2.`/

**最终，代码比较的是 `"Thereforejackhas2.` 和从 response 中提取并标准化后的 `"2"`**。这两个字符串不想等，因此评估结果为错误。

#### 最佳实践
为了让 `r1_zero_reward_fn` 这个奖励函数能够准确工作，`ground_truth` 的格式应该遵循以下原则：
- **核心必须是数学答案**：`ground_truth` 应该是数学上精确的最终答案。
- **可以包含常见的、可被移除的单位**：例如 `meters`, `dollars`等。
- **可以包含 Latex 格式**：特别是 `\boxed{}` 或 `\frac{}{}` 等。
- **不应该包含完整的、任意的自然语言句子**。