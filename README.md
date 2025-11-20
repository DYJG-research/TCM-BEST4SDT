<h1 align="center">TCM-BEST4SDT: A benchmark dataset for evaluating Syndrome Differentiation and Treatment in large language models</h1>

<p align="center">
    <b>🌐 Language:</b> English | <a href="README_zh.md">中文</a>
</p>

## ⚡️ Overview

**TCM‑BEST4SDT** is a comprehensive benchmark centered on Syndrome Differentiation & Treatment (SDT), designed to systematically evaluate the application capabilities of Large Language Models (LLMs) in Traditional Chinese Medicine （TCM） clinical scenarios. The figure below illustrates the data construction pipeline and evaluation methodology of TCM‑BEST4SDT.

For dataset construction, the process was led by a team of TCM experts, with samples collected from clinical cases, classical case records, and past publicly available examination questions. After deduplication, cleaning, and anonymization, a three-stage annotation workflow was adopted: expert annotation, cross‑validation among annotators, and independent third-party review.

TCM‑BEST4SDT covers four task families: TCM **B**asic Knowledge, Medical **E**thics, Large Language Model Content **S**afety, and **S**yndrome **D**ifferentiation And **T**reatment, totaling 600 questions.

For evaluation methodology, we designed three distinct evaluation mechanisms: (1) Objective question evaluation: Each question underwent multiple rounds of independent assessment with randomized option orders, and different scoring strategies were applied according to the question type; (2) Judge model evaluation: The responses of the evaluated models were scored by a judge model incorporating expert-designed prompts; (3) Reward model evaluation: A dedicated reward model was trained to quantify the degree of compatibility (or match) between the generated prescriptions and their corresponding syndrome.

We conducted experiments on 15 mainstream large models, encompassing both general and TCM domain models, which validated the sensitivity and effectiveness of TCM-BEST4SDT.

<div align="center">
  <img src="images/methods.png" width="80%"/>
</div>

## 📚 Methods

### 🧬 Creation of TCM-BEST4SDT

#### 🎓 Data Collection & Preprocessing

- 🧩 Task Types & Sources
  - **Syndrome Differentiation & Treatment**: derived from clinical cases and classical case records.
  - **General Evaluation Tasks**: Covering TCM Basic Knowledge, Medical Ethics, and LLM Content Safety.
    - **TCM Basic Knowledge**: Selected from publicly available examination question banks, including the National Qualification Examination for Medical Practitioners,  the National Postgraduate Entrance Examination: Comprehensive Clinical Medicine (TCM Integrated), Chinese Herbal Medicine Title Examination;
    - **Medical Ethics**: In addition to authoritative question banks, samples were annotated by experts based on specific scenarios;
    - **LLM Content Safety**: Independently designed and constructed by experts in relevant fields based on practical requirements.

- 🔧 Preprocessing
  1. **Deduplication:** All cases and examination questions were deduplicated to ensure sample independence;
  2. **Cleaning**: Data were systematically cleaned by removing samples with missing stems or options, non-text content such as figures or tables, and manually correcting character errors caused by OCR, ensuring textual accuracy and consistency;
  3. **Anonymization**: For clinical cases and medical records, patient names and other identifiable information were removed or replaced in strict accordance with medical ethics and privacy protection principles, ensuring data compliance and privacy security.

#### 📝 Data Annotation

- 🧠 **Syndrome Differentiation and Treatment Task:** Syndrome Differentiation and Treatment is the core principle of TCM diagnosis and therapy, encompassing the two stages of “syndrome differentiation” and “treatment formulation.” To evaluate model capabilities in this process, this study proposes a comprehensive 14-dimension evaluation framework, including syndrome, causative factors, and pathogenesis.

  - **Annotation Procedure:**

    1. Initial annotations of standard answers and options were completed by TCM experts;
    2. Cross-validation was conducted among annotators;
    3. Final underwent an independent third-party review to ensure annotation consistency and reliability.
  - **Question Types:** The task includes selected-response questions and Q&A questions. For selected-response questions, sophisticated distractors to enhance the discriminative power of the assessment.

  > 📑 **Process-Oriented Metrics:** In addition to evaluating the final reasoning outcomes in Syndrome Differentiation and Treatment, this study introduces two process-oriented metrics to assess reasoning quality.
  >
  > * **Chain-of-Thought (CoT) Content Completeness:** Measures the coverage of key patient information within the model's CoT;
  > * **CoT Accuracy:** Assesses the consistency of elements cited in the CoT with the original clinical case to identify potential hallucinations or reasoning deviations.

- 📊 **General Evaluation Tasks:** Designed to reveal model differences in foundational cognition and normative responses, thereby achieving a more comprehensive and interpretable overall evaluation.

  - **TCM Basic Knowledge (four core dimensions; all questions selected from preprocessed authoritative examination question banks):**

    1) Understanding of classical TCM texts;
    2) Mastery of basic theories;
    3) Knowledge of Chinese materia medica and formulas;
    4) Syndrome differentiation ability based on tongue, pulse, facial complexion, and acupoints.
  - **Medical Ethics:** Evaluates the model’s understanding and judgment of clinical ethics, with content covering conflicts between traditional concepts and modern medicine, discrimination of unscientific behaviors, respect for patient cultural beliefs, and informed consent. Questions were drawn from authoritative exam banks and supplemented by expert-designed questions based on the aforementioned scenarios.
  - **LLM Content Safety:** Evaluate the performance of the TCM domain LLM in terms of professional boundaries and safety compliance. Models are expected to respond only to TCM-related questions and to refrain from answering questions involving user privacy, security risks, or ideological content. Relevant questions were independently designed and reviewed by experts in content safety and ethics according to these principles, ensuring the evaluation meets the professionalism and safety requirements for TCM domain LLMs.
  - **Question Types:** All questions are selected-response questions, including both single-selection and multiple-selection types.

### 📐 Evaluation Methods

#### 🧮 Selected-Response Evaluation

The selected-response questions in TCM-BEST4SDT include single-selection and multiple-selection types, all following a unified evaluation procedure:

- Each question undergoes three rounds of independent assessment;
- The option order is randomized in each round.

  - **Single-selection Questions:** Considered correct only if the model produces fully consistent and correct answers across all three rounds.

  - **Multiple-selection Questions:**

$$
S = \frac{|A \cap B|}{|A| + |\bar{A} \cap B|}
$$

  Notation:
  - $S$: score of the question
  - $A$: gold answers; $B$: model answers
  - $|A \cap B|$: number of correct selections; $|\bar{A} \cap B|$: number of wrong selections

    Final score is the mean of three rounds.

#### ⚖️ Judge Model Evaluation
- Judge Model: `Qwen3-32B`
- Input: model response, gold answer, and expert-designed prompts
- Output: scores for corresponding dimensions

#### 🏅 Reward Model Evaluation

- **Objective:** To address the challenge of mismatching between prescriptions and syndromes, this study developed a dedicated reward model to quantify prescription-syndrome congruence, thereby enabling objective assessment of prescription suitability.;
- **Base Model:** `Qwen3-14B`;
- **Training Data:** Sourced from real clinical cases and classical TCM formulas, consists of 10k samples, each containing one corresponding syndrome and six candidate prescriptions rated by experts.

##### 📦 Open Source Access

The reward model developed in this work has been open‑sourced on ModelScope.

| Model | Base | Links |
| :-- | :-- | :-- |
| `FangZheng-RM` | `Qwen3-14B` | [🔬 ModelScope](https://www.modelscope.cn/models/DYJGresearch/FangZheng-RM) |

## 🔍 Experiments

### 🤖 Model Selection

<div align="center">

| Model | Size |
| :--: | :--: |
| `GPT-5` | - |
| `Gemini 2.5 Pro` | - |
| `DeepSeek-R1` | 671B |
| `Doubao-seed-1.6` | 230B |
| `Kimi-K2` | 1T |
| `Qwen3` | 4B / 8B / 14B / 32B / 80B / 235B |
| `GLM-4.5` | 355B |
| `Llama-4-Scout-17B-16E-Instruct` | 109B |

</div>

#### TCM domain LLMs

<div align="center">

| Model | Size |
| :--: | :--: |
| `HuatuoGPT-o1-7B` | 7B |
| `BianCang-Qwen2.5-7B` | 7B |
| `Baichuan-M2-32B` | 32B |
| `Sunsimiao-Qwen2-7B` | 7B |
| `ShizhenGPT-32B-LLM` | 32B |
| `Zhongjing-GPT-13B` | 13B |
| `Taiyi 2` | 9B |

</div>

### 🛠️ Experimental Settings

- **Deployment via SWIFT Framework (small-scale open-source):** Locally deployed using the `SWIFT` framework, with unified invocation and evaluation through an OpenAI-compatible interface;
- **Official API Access (large-scale open-source/closed-source):** Remotely invoked and evaluated via official APIs, including `GPT-5`, `Gemini 2.5 Pro`, `DeepSeek-R1`, `Doubao-seed-1.6`, `Kimi-K2`, and `GLM-4.5`;
- **Deployment via Transformers Library:** Only `Taiyi 2` was locally loaded and evaluated using the `Transformers` library;
- **Unified Settings:** The evaluation temperature for all models was fixed at `0` to ensure stability and reproducibility;
- **Chain-of-Thought Evaluation:** For models lacking reasoning capabilities (e.g., `Kimi-K2`, `Llama-4-Scout`), the `--skip_think` option can be used to control whether chain-of-thought quantification is applied.


### 📊 Results & Analysis

<div align="center">
  <img src="images/main_results.png" width="60%"/>
  <br/>
  <em>Figure 1: Performance of 15 LLMs on the TCM-BEST4SDT benchmark dataset</em>
</div>

1. <strong>Some TCM domain models outperform general domain models</strong>: Models such as“ShizhenGPT-32B-LLM” demonstrated outstanding performance, indicating that high-quality TCM corpora combined with supervised fine-tuning can effectively enhance their clinical inference capabilities.
2. <strong>Most TCM models lag behind certain general domain models</strong>: Examples include “Sunsimiao-Qwen2-7B,” “Zhongjing-GPT-13B,” and “Taiyi 2.” Possible reasons include generally smaller parameter scales and over-reliance on medical/TCM corpora during training, whereas real-world clinical scenario corpora are limited,  leading to insufficient generalization ability. TCM-BEST4SDT is centered on SDT capabilities. It places higher demands on the decision-making abilities of LLMs in clinical contexts; consequently, the performance of these TCM models on this benchmark was relatively poor.
3. <strong>Among general large language models, GPT-5 scores significantly lower than Gemini 2.5 Pro</strong>: This gap may reflect differences in TCM-related coverage within their training corpora. Gemini 2.5 Pro incorporated richer medical and TCM knowledge during pretraining, thereby demonstrating stronger semantic understanding and transfer capability when handling TCM knowledge-intensive tasks.
4. <strong>sAs shown in Figure 2, the performance of the Qwen3 series steadily improves with model scale</strong>: This validates the effectiveness of the scaling law and highlights the sensitivity and effectiveness of TCM-BEST4SDT in differentiating model capabilities.

Overall, the evaluation results demonstrate that TCM-BEST4SDT can objectively reflect the performance differences among various types of LLMs on TCM tasks and effectively demonstrate their potential application value in real-world clinical scenarios. By constructing a quantitative and reproducible evaluation system, this study provides a scientific basis for the clinical application of TCM domain LLMs and further promotes the standardization and industrialization of intelligent TCM research.


<div align="center" style="margin-top:8px;">
  <img src="images/qwen3_scores.png" width="60%"/>
  <br/>
  <em>Figure 2: Scores of six Qwen3 model sizes on TCM‑BEST4SDT</em>
</div>

## 🚀 Quick Start

### 1) Install Dependencies

```bash
# Create and activate env
conda create -n best4sdt python=3.10.12
conda activate best4sdt
pip install -r requirements.txt
```

### 2) Configure

Edit `config_example.json`:

>replace placeholders like reward_model_api/port/key

```json
{
  "data_path": "TCM-BEST4SDT.json",
  "local_model_gpu_id": -1,
  "reward_api_host": "reward_model_api",
  "reward_api_port": 8000,
  "reward_model_name": "fangji",
  "reward_api_key": "reward_model_key",
  "llm_judge_api_host": "judge_model_api",
  "llm_judge_api_port": 8000,
  "llm_judge_model_name": "Qwen3-32B",
  "llm_judge_api_key": "judge_model_key",
  "max_retries": 3,
  "checkpoint_interval": 10
}
```

#### 🧾 Parameter Description

| Key | Description |
|:-:|:-:|
| `data_path` | Path to evaluation dataset |
| `local_model_gpu_id` | Transformers local model GPU switch<br/>(-1 for CPU only; ≥0 to enable GPU) |
| `reward_api_host` | Reward model service host |
| `reward_api_port` | Reward model service port |
| `reward_model_name` | Reward model name |
| `reward_api_key` | Reward model API key |
| `llm_judge_api_host` | Judge model service host |
| `llm_judge_api_port` | Judge model service port |
| `llm_judge_model_name` | Judge model name |
| `llm_judge_api_key` | Judge model API key |
| `max_retries` | Max API retry times |
| `checkpoint_interval` | Checkpoint interval (num samples) |

> Note: `local_model_gpu_id = -1` forces CPU. To enable GPU, first expose devices via `CUDA_VISIBLE_DEVICES=...`, then set `local_model_gpu_id >= 0` (e.g., `0`). Actual device selection is governed by `CUDA_VISIBLE_DEVICES` plus `device_map="auto"`.

### 3) Run Evaluation

**Approach A: ☁️ Evaluation via OpenAI-Compatible Interface**

```bash
python tcm_benchmark.py \
  --model_type api \
  --api_url http://localhost:8000/v1 \
  --model_name Qwen3-8B \
  --api_key $OPENAI_API_KEY \
  --config_file ./config_example.json \
  --output_dir ./results/run-001 \
  --resume \
  --skip_think # If the model does not support CoT, this parameter can be used to skip CoT-related dimensions
```

**Approach B: 🖥️ Local Loading via Transformers**

```bash
python tcm_benchmark.py \
  --model_type local \
  --model_path /path/to/model \
  --config_file ./config_example.json \
  --output_dir ./results/run-local \
  --resume
```

#### 🧾 Parameter Description

| Argument | Purpose | Mode |
|:-:|:-:|:-:|
| `--model_type` | Model Invocation Mode: `api` or `local` | both |
| `--api_url` | OpenAI-compatible API URL | api only |
| `--model_name` | API model name | api only |
| `--api_key` | API key | api only |
| `--model_path` | Local model path | local only |
| `--config_file` | Config file path | both |
| `--output_dir` | Output directory | both |
| `--resume` | Resume if `checkpoint.json` exists | both |
| `--skip_think` | Skip CoT completeness/accuracy | both |

---

## 📚 Citation

If you use TCM‑BEST4SDT in your research, please cite our work:

```bibtex
@misc{TCM-BEST4SDT,
  author={DYJG-research},
  title = {A benchmark dataset for evaluating the comprehensive capability of Syndrome Differentiation and Treatment in traditional Chinese medicine},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/DYJG-research/TCM-BEST4SDT},
}
```

## 🙏 Acknowledgments

We thank the following open-source projects for their support:

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








