<h1 align="center">TCM-BEST4SDT: A benchmark dataset for evaluating Syndrome Differentiation and Treatment in large language models</h1>

<p align="center">
    <b>🌐 语言：</b> <a href="README.md">English</a> | 中文
</p>

## ⚡️ 概述

**TCM‑BEST4SDT** 是一个以“辨证论治（Syndrome Differentiation & Treatment, SDT）”为核心的综合评估基准，旨在系统评估大语言模型（LLM）在中医临床场景中的应用能力。下图展示了TCM-BEST4SDT的数据构建流程与评测方法。

在数据集构建方面，由中医专家团队主导，样本源自临床病例、名家医案及权威公开考试题。所有数据经过去重、清洗与匿名化处理后，采用一个严格的三阶段流程完成标注，包括专家标注、标注者之间交叉验证以及独立的第三方评议审定。

TCM‑BEST4SDT 涵盖四类任务：中医基础知识（TCM **B**asic Knowledge）、医学伦理（Medical **E**thics）、大模型内容安全（Large Language Model Content **S**afety）以及辨证论治（**S**yndrome **D**ifferentiation And **T**reatment），共计 600 题。

在评测方法方面，设计三种不同的评测机制：（1）客观题评测：每题多轮独立评测并随机化选项顺序，并根据题型采用不同计分策略计算得分；（2）裁判模型评测：采用高性能大语言模型作为评估器，结合专家设计的提示词，对待测模型回答进行打分；（3）奖励模型评测：开发了一种专用奖励模型，量化处方与证型的一致性。

通过在涵盖通用和中医领域的 15 个主流大语言模型上的实验，验证了 TCM‑BEST4SDT 的有效性。

<div align="center">
  <img src="images/methods.png" width="80%"/>
</div>

## 📚 方法介绍

### 🧬 TCM-BEST4SDT的构建

#### 🎓 数据收集与预处理

- 🧩 **任务类型与来源**
  - **中医辨证论治任务**：来源于临床病例与名家医案；
  - **通用评测任务**：涵盖中医基础知识、医学伦理与大模型内容安全：
    - **中医基础知识**：选取自公开考试题库，包括国家执业医师资格考试、硕士研究生招生考试临床医学综合能力（中医综合）、中药学职称考试等；
    - **医学伦理**：除来源于权威题库外，亦由专家基于特定场景进行人工标注；
    - **大模型内容安全**：由相关领域专家结合实际需求独立设计与构建。

- 🔧 **数据预处理**
  1. **去重**：对所有病例与试题进行去重，确保样本独立性；
  2. **清洗**：系统化清洗数据，剔除题干或选项缺失、含图表等非文本内容的样本，并对 OCR 识别导致的字符错误进行人工校对，保证文本的准确性与一致性；
  3. **匿名化处理**：针对临床病例与医案数据，严格遵循医学伦理与隐私保护原则，删除或替换患者姓名及其他可识别信息，确保数据合规与隐私安全。

#### 📝 数据标注

- 🧠 **辨证论治任务**：辨证论治是中医诊断与治疗的核心方法，包含“辨证”与“论治”两个环节。为全面评估模型在该流程中的能力，本研究提出了一个涵盖 14 个维度的综合评价体系，包括证型、病因、病机等。
  - **标注流程**：
    1. 由中医专家完成标准答案与选项的初步标注；
    2. 标注者之间进行交叉验证；
    3. 最终由第三方评议委员会审定，确保标注一致性和可靠性。
  - **题型**：包含选择题与Q&A。其中选择题设置高干扰度选项，以以增强评估的区分能力。

  > 📑 过程导向指标：除评估辨证论治的最终推理结果外，本研究引入两项过程指标用于衡量推理质量。
  >
  > - **CoT内容完备性**：衡量模型在推理链中对患者关键信息的覆盖程度；
  > - **CoT准确性**：评估思维链中引用信息与原始病例之间的一致性，用于识别潜在幻觉或推理偏差。


- 📊 **通用评测任务**：旨在揭示模型在基础认知与规范性响应方面的差异，从而实现更全面、可解释的综合评估。
  - **中医基础知识（四个核心维度，相关题目均选自经预处理的权威考试题库）**：
    1) 中医典籍理解；
    2) 基础理论掌握；
    3) 中药与方剂学知识；
    4) 基于舌象、脉象、面色及穴位的辨证能力。
  - **医学伦理**：评估模型对临床伦理的理解与判断，包括传统观念与现代医学的冲突、对封建迷信行为的辨析、患者文化信仰的尊重与知情同意等。题目除来源于权威考试题库外，亦由专家基于上述场景设计。
  - **大模型内容安全**：评估模型在职业边界与安全合规方面的表现。模型应仅回答与中医相关的问题，对涉及用户隐私、安全风险或人类价值观等非医学领域的问题保持拒答。相关试题由内容安全及伦理领域专家依据上述原则独立设计与审定，确保评测符合中医大语言模型的专业性与安全性要求。
  - **题型**：所有题目均为选择题，包含单项与多项两种类型。

### 📐 评测方法

#### 🧮 客观题评测  
TCM-BEST4SDT 的客观题包含单项选择题、多项选择题与不定项选择题，三类题型采用统一流程：
- 每题进行三轮独立评测；
- 每轮随机化选项顺序。

  - **单项选择题**：仅当模型在三轮中输出完全一致且均为正确答案时判定为正确。

  - **多项选择题及不定项选择题**：

$$
S = \frac{|A \cap B|}{|A| + |\bar{A} \cap B|}
$$

  符号说明:
  - $S$: 该题得分
  - $A$: 标准答案集合; $B$: 模型作答集合
  - $|A \cap B|$: 正确选择数; $|\bar{A} \cap B|$: 错误选择数。

    最终题目得分取三轮得分的平均值。

#### ⚖️ 裁判模型评测
- **裁判模型**：`Qwen3-32B`
- **输入**：模型响应、标准答案与专家评分提示词；
- **输出**：对应维度的得分。

#### 🏅 奖励模型评测
- **目的**：解决处方与证型匹配难题，我们开发了一个奖励模型用于量化处方与证型的一致性，从而实现对处方适配性的客观评估；
- **基座模型**：`Qwen3-14B`；
- **训练数据**：来源于真实临床病例与经典中医方剂，共 10k 条样本，每条样本含 1 个对应证型与 6 个经专家评分的候选处方。

##### 📦 开源获取

我们已将所开发的奖励模型在 ModelScope 平台开源。

| 模型 | 基座 | 链接 |
| :-- | :-- | :-- |
| `FangZheng-RM` | `Qwen3-14B` | [🔬 ModelScope](https://www.modelscope.cn/models/DYJGresearch/FangZheng-RM) |

## 🔍 实验验证

### 🤖 模型选择

#### 通用领域大语言模型

<div align="center">

| 模型  | 规模 |
| :--: | :--: |
| `GPT-5.2` | - |
| `Gemini 3 Pro` | - |
| `DeepSeek-R1` | 671B |
| `Doubao-seed-1.6` | 230B |
| `Kimi-K2` | 1T |
| `Qwen3` | 4B / 8B / 14B / 32B / 80B / 235B |
| `GLM-4.5` | 355B |
| `Llama-4-Scout-17B-16E-Instruct` | 109B |

</div>

#### 中医领域大语言模型

<div align="center">

| 模型 | 规模 |
| :--: | :--: |
| `HuatuoGPT-o1-7B` | 7B |
| `BianCang-Qwen2.5-7B` | 7B |
| `Baichuan-M2-32B` | 32B |
| `Sunsimiao-Qwen2-7B` | 7B |
| `ShizhenGPT-32B-LLM` | 32B |
| `Zhongjing-GPT-13B` | 13B |
| `Taiyi 2` | 9B |

</div>

### 🛠️ 实验设置

- **基于SWIFT框架部署（小规模开源）**：基于 `SWIFT` 框架完成本地部署，并通过 OpenAI 兼容接口统一调用与评测；
- **官方API调用（大规模开源/闭源）**：通过官方 API 远程调用并评测，包括`GPT-5`、`Gemini 2.5 Pro`、`DeepSeek-R1`、`Doubao-seed-1.6`、`Kimi-K2`、`GLM-4.5`；
- **基于Transformers库部署**：仅 `Taiyi 2` 通过 `Transformers` 在本地加载完成评测；
- **统一设置**：所有模型评测温度固定为 `0`，确保结果稳定性与可复现性；
- **思维链评测**：对不具备推理能力的模型（如 `Kimi-K2`、`Llama-4-Scout`）可通过 `--skip_think` 控制是否启用思维链量化评估。

### 📊 结果与分析

#### 主要评测结果
<div align="center">
  <img src="images/main_results.svg" width="60%"/>
  <br/>
  <em>图 1：15 个大语言模型在 TCM-BEST4SDT 基准数据集上的表现</em>
</div>

**核心结论（概览）：**

- **总体排名：**Gemini 3 Pro（0.8711）位居第一，Doubao-seed-1.6（0.8303）位列第二。
- **中医领域模型代表：**ShizhenGPT-32B-LLM 的总分为 0.7826。
- **过程评测受“推理轨迹可见性”影响：**我们将“显式输出 CoT”的模型与“不显式输出 CoT”的模型分别报告。
- **Gemini 3 Pro vs GPT-5.2：**Gemini 3 Pro 总分领先（0.8711 vs 0.7866），主要优势来自中医基础知识（0.9567 vs 0.6567），而辨证论治能力表现相近（0.8342 vs 0.8415）。

1. <strong>前沿通用模型整体表现突出。</strong>

  Gemini 3 Pro 获得最高总分（0.8711），Doubao-seed-1.6 次之（0.8303），表明其具备较强的辨证论治能力。这些结果提示，模型能够利用病例描述中的患者信息完成相对准确的辨证，并生成与所辨证型高度一致的处方。

  中医领域模型中表现最佳的 ShizhenGPT-32B-LLM 取得 0.7826，也具备竞争力。但前沿通用模型与多数中医领域模型之间仍存在差距，例如 Sunsimiao-Qwen2-7B 为 0.5161。该差距可能反映了：（i）模型规模限制了其复杂临床语境理解与临床决策的能力；（ii）训练目标差异：部分中医领域模型更偏向特定应用形态优化，而非覆盖更广的辨证论治临床能力。

2. <strong>显式 CoT 模型与非显式 CoT 模型：过程评测的适用性差异。</strong>

  我们观察到，能够输出显式、可分离推理过程（CoT）的模型，与仅输出最终答案的模型存在明显差异。因此，我们将评测对象分为（i）显式 CoT 模型与（ii）非显式 CoT 模型两组，并分别给出结果。

  这种划分反映的是<strong>推理透明度</strong>而非推理能力：不输出显式 CoT 不代表不具备推理能力。例如，Kimi-K2 通过强化学习在数学、STEM 与逻辑推理任务上进行训练，具备非平凡的内部推理能力；但由于其不以可分离的 CoT 形式输出中间推理过程，基于 CoT 的过程评测并不直接适用。

  因此，我们并行报告两类模型的结果：对非显式 CoT 模型的整体高分，更倾向于解释为强内部推理能力的体现；同时强调，可审计的推理过程评估需要显式推理轨迹。

3. <strong>GPT-5.2 vs Gemini 3 Pro：知识覆盖与推理能力的权衡。</strong>

  Gemini 3 Pro 的总分更高（0.8711），而 GPT-5.2 为 0.7866。其优势主要来自中医基础知识任务：Gemini 3 Pro 为 0.9567，而 GPT-5.2 为 0.6567（表 1），提示 Gemini 3 Pro 对中医术语与理论概念覆盖更广。

  值得注意的是，两者在核心辨证论治任务上的表现相近：Gemini 3 Pro 为 0.8342，GPT-5.2 为 0.8415。这表明尽管知识覆盖存在差异，两者在临床病例分析所需的推理能力上均表现较强。

<div align="center">
  <img src="images/task_score_results.png" width="60%"/>
  <br/>
  <em>表 1：15 个大语言模型在 TCM-BEST4SDT 四项评测任务上的表现</em>
</div>

<div align="center">
  <img src="images/Fig_Radar_SDT_Dimensions.svg" width="60%"/>
  <br/>
  <em>图 2：推理/非推理模型在辨证论治任务各维度的得分对比</em>
</div>

#### Scaling Laws 评估
为验证基准对模型能力（规模）差异的敏感性，我们在 Qwen3 系列（4B、8B、14B 与 32B）上进行了受控评测。结果显示性能随模型规模提升呈现稳定的单调上升趋势（见图 3），表明 TCM-BEST4SDT 对不同能力水平模型具有良好的区分度。

总体而言，评测结果表明 TCM-BEST4SDT 能够客观反映不同类型大语言模型在中医任务上的表现差异，并有效展示其在真实临床场景中的潜在应用价值。通过构建量化、可复现的评估体系，本研究为中医领域大语言模型的临床应用提供了科学依据，并有望推动中医智能化研究的标准化与产业化。

<div align="center" style="margin-top:8px;">
  <img src="images/qwen3_scores.svg" width="60%"/>
  <br/>
  <em>图 3：Qwen3 不同模型规模在 TCM‑BEST4SDT 上的得分</em>
</div>

#### 中医领域模型与其基座模型对比

TCM-BEST4SDT 的一个关键目标是检验：面向中医领域的专门微调是否能在 辨证论治 导向的临床推理评估中带来可观提升。为尽可能剥离“基座模型能力”对结果的影响，我们将代表性的中医微调模型与其对应的通用基座模型进行了对比。

具体而言，我们将 ShizhenGPT-32B-LLM 与 Baichuan-M2-32B 对比 Qwen2.5-32B-Instruct，将 BianCang-Qwen2.5-7B 与 HuatuoGPT-o1-7B 对比 Qwen2.5-7B-Instruct（见图 4）。

出人意料的是，在本基准上，通用基座模型整体优于其对应的中医微调模型。例如，Qwen2.5-32B-Instruct 的总分为 0.8077，高于 ShizhenGPT-32B-LLM 的 0.7826；Qwen2.5-7B-Instruct 为 0.6698，也略高于 HuatuoGPT-o1-7B 的 0.6632。

这一现象提示：在 辨证论治 导向的评测框架下，领域监督微调并不必然带来净增益。一个可能原因是，部分中医领域模型的训练目标更偏向特定应用场景，监督数据也可能更强调特定交互模式而非覆盖更广的临床病例推理；相比之下，通用基座模型往往在更丰富多样的指令数据上进行后训练，从而在病例理解与临床决策泛化方面更稳健。我们将该结论作为基准观测结果报告，并强调需要进一步系统研究能够在提升中医知识与临床辨证论治能力的同时保留通用推理能力的领域适配策略。

<div align="center" style="margin-top:8px;">
  <img src="images/Base_vs_Finetuned.svg" width="60%"/>
  <br/>
  <em>图 4：中医领域大语言模型与其对应基座模型的对比</em>
</div>

## 🚀 快速开始

### 1）安装依赖

```bash
# 创建并激活虚拟环境
conda create -n best4sdt python=3.10.12
conda activate best4sdt
pip install -r requirements.txt
```

### 2）配置运行参数

编辑 `config_example.json`：

> 提示：将以下占位符替换为实际服务地址、端口与密钥（例如 reward_model_api）。

```json
{
  "data_path": "TCM-BEST4SDT.json",
  "local_model_gpu_id": -1,
  "reward_api_host": "reward_model_api",
  "reward_api_port": 8000,
  "reward_model_name": "Fangzheng-RM",
  "reward_api_key": "reward_model_key",
  "llm_judge_api_host": "judge_model_api",
  "llm_judge_api_port": 8000,
  "llm_judge_model_name": "Qwen3-32B",
  "llm_judge_api_key": "judge_model_key",
  "max_retries": 3,
  "checkpoint_interval": 10
}
```

#### 🧾 参数说明

| 参数 | 说明 |
|:-:|:-:|
| `data_path` | 评测数据集文件路径 |
| `local_model_gpu_id` | 基于 transformers 加载本地模型的 GPU 使用开关<br/>（-1 仅用 CPU，≥0 启用 GPU） |
| `reward_api_host` | 奖励模型服务地址 |
| `reward_api_port` | 奖励模型服务端口 |
| `reward_model_name` | 奖励模型名称 |
| `reward_api_key` | 奖励模型API密钥 |
| `llm_judge_api_host` | 裁判模型服务地址 |
| `llm_judge_api_port` | 裁判模型服务端口 |
| `llm_judge_model_name` | 裁判模型名称 |
| `llm_judge_api_key` | 裁判模型API密钥 |
| `max_retries` | 接口失败重试次数 |
| `checkpoint_interval` | 断点保存间隔（样本数） |

> 注意：当 `local_model_gpu_id = -1` 时强制使用 CPU；如需启用 GPU，请先通过 `CUDA_VISIBLE_DEVICES=...` 暴露可见 GPU，再将 `local_model_gpu_id` 设为 ≥ 0（如 `0`）。实际设备分配由 `CUDA_VISIBLE_DEVICES` 与 `device_map="auto"` 决定。

### 3）启动评测

**方式 A：☁️ 调用 OpenAI 兼容接口评测**

```bash
python tcm_benchmark.py \
  --model_type api \
  --api_url http://localhost:8000/v1 \
  --model_name Qwen3-8B \
  --api_key $OPENAI_API_KEY \
  --config_file ./config_example.json \
  --output_dir ./results/run-001 \
  --resume \
  --skip_think   # 若模型不支持 CoT，可加此参数跳过 CoT 相关维度
```

**方式 B：🖥️ 基于 Transformers 本地加载**

```bash
python tcm_benchmark.py \
  --model_type local \
  --model_path /path/to/model \
  --config_file ./config_example.json \
  --output_dir ./results/run-local \
  --resume
```

#### 🧾 参数说明

| 参数 | 作用 | 适用模式 |
|:-:|:-:|:-:|
| `--model_type` | 模型调用方式：`api` 或 `local` | 两者 |
| `--api_url` | OpenAI 兼容 API 地址 | 仅 `api` |
| `--model_name` | API 模型名称 | 仅 `api` |
| `--api_key` | API 密钥 | 仅 `api` |
| `--model_path` | 本地模型路径 | 仅 `local` |
| `--config_file` | 评测配置文件路径 | 两者 |
| `--output_dir` | 结果输出目录 | 两者 |
| `--resume` | 断点续跑开关（存在 `checkpoint.json` 时续跑） | 两者 |
| `--skip_think` | 跳过 CoT 内容完备性/准确性评测 | 两者 |

> 说明：`--resume` 在检测到输出目录内存在 `checkpoint.json` 时自动续跑；`--skip_think` 用于不具备思维链能力的模型以关闭 CoT 相关评测。

---

## 📚 引用

如果您在研究中使用了 TCM‑BEST4SDT，请引用我们的工作：

```bibtex
@misc{li2025benchmarkdatasetevaluatingsyndrome,
      title={A benchmark dataset for evaluating Syndrome Differentiation and Treatment in large language models}, 
      author={Kunning Li and Jianbin Guo and Zhaoyang Shang and Yiqing Liu and Hongmin Du and Lingling Liu and Yuping Zhao and Lifeng Dong},
      year={2025},
      eprint={2512.02816},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.02816}, 
}
```

## 🙏 致谢

感谢以下开源项目对本工作的支持：

- **[ms-SWIFT](https://github.com/modelscope/ms-SWIFT)** 
- **[transformers](https://github.com/huggingface/transformers)** 
- **[Qwen 3](https://github.com/QwenLM/Qwen3)** 
- **[llama-models](https://github.com/meta-llama/llama-models)** 
- **[HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)** 
- **[BianCang-TCM-LLM](https://github.com/QLU-NLP/BianCang-TCM-LLM)** 
- **[Sunsimiao](https://github.com/X-D-Lab/Sunsimiao)** 
- **[ShizhenGPT](https://github.com/FreedomIntelligence/ShizhenGPT)** 
- **[CMLM-ZhongJing](https://github.com/pariskang/CMLM-ZhongJing)** 
- **[Taiyi-LLM](https://github.com/DUTIR-BioNLP/Taiyi-LLM)** 

---
