#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励模型评测器（Reward Model Evaluator）

使用 fangji 奖励模型：
- 生成被测模型的处方（组成/配伍/安全/禁忌信息）供后续解析
- 评估“方证匹配”分数（匹配分数 0-100 -> 0-1）
- 生成 4 个维度（方剂配伍规律、安全性方面、配伍禁忌、妊娠禁忌）的参考答案，由 LLMJudgeEvaluator 进行对照判分
"""

import json
import time
import logging
import re
from typing import Dict, Any, Tuple, List, Optional
from tqdm import tqdm
from openai import OpenAI

logger = logging.getLogger(__name__)

class RewardModelEvaluator:
    """奖励模型评测器"""
    
    def __init__(self, api_host: str, api_port: int, model_name: str = "fangji", api_key: Optional[str] = None):
        """
        初始化奖励模型评测器
        
        Args:
            api_host: API主机地址
            api_port: API端口
            model_name: 奖励模型名称
        """
        self.api_host = api_host
        self.api_port = api_port
        self.api_base_url = f"http://{api_host}:{api_port}/v1"
        self.model_name = model_name
        self.api_key = api_key or "dummy-key"
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
        )
        
        # 奖励模型相关维度（仅展示方证契合度，其余四项在 LLM 评分分组中展示）
        self.reward_dimensions = [
            "方证契合度", "方剂配伍规律", "药材安全性分析", "配伍禁忌", "妊娠禁忌"
        ]
        
        # 处方缓存（按案例）
        self.prescription_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"初始化奖励模型评测器: {self.api_base_url} 模型: {self.model_name}")
    
    # =================== 合并评测五个维度 ===================
    def evaluate_all(self, case: Dict[str, Any], model_interface, syndrome_response: str, pbar: tqdm, llm_judge_evaluator=None) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        奖励模型仅用于：
        - 解析被测模型生成的处方（组成/配伍/安全/禁忌）
        - 评估“方证匹配”（0-1）
        - 生成 4 维度参考答案文本，由 LLMJudgeEvaluator 对被测模型输出进行评分

        返回：(scores, responses)
        - scores 为各维度得分（方证匹配来自奖励模型；其余 4 维度来自 LLM 判分）。
        - responses 为各维度的说明性文本（均为被测模型生成的内容）。
        """
        # 生成或获取处方
        prescription_response = self._get_or_create_prescription(case, syndrome_response, model_interface, pbar)
        prescription_info = self._parse_prescription_response(prescription_response)          
        # 组织信息
        herbs_list = [f"{h[0]} {h[1]} {h[2]}".strip() for h in prescription_info.get("herbs", [])]
        compatibility_theory = prescription_info.get("compatibility_theory", "").strip()
        toxic_handling = prescription_info.get("toxic_handling", "").strip()
        incompatibility_warning = prescription_info.get("incompatibility_warning", "").strip()
        pregnancy_warning = prescription_info.get("pregnancy_warning", "").strip()

        # 参考答案将在LLM判分分支中按需生成

        # 构造“方证匹配”提示并调用奖励模型获取匹配分
        mapped_syndrome = self._map_syndrome_response(syndrome_response, case)
        herbs_text = "\n".join([f"- {x}" for x in herbs_list]) if herbs_list else ""
        match_prompt = self._build_fangzheng_match_prompt(case, mapped_syndrome, herbs_text)
        pbar.write("正在调用奖励模型进行方证匹配评估...")
        result = self._call_reward_model_json(match_prompt)

        # 依据此前规则组装得分
        scores: Dict[str, float] = {}
        responses: Dict[str, str] = {}

        # 方证契合度：使用奖励模型输出的匹配分数（0-100）归一化到0-1
        raw_match_score = result.get("匹配分数", 50)
        try:
            raw_match_score = float(raw_match_score)
        except Exception:
            raw_match_score = 50.0
        scores["方证契合度"] = max(0.0, min(1.0, raw_match_score / 100.0))
        # 展示：提取待评测模型生成的处方组成内容
        herbs_content = "\n".join([f"- {h[0]} {h[1]} {h[2]}".strip() for h in prescription_info.get("herbs", [])])
        responses["方证契合度"] = herbs_content if herbs_content else "未解析到处方组成信息"

        # 组装tested模型输出（文本）
        responses["方剂配伍规律"] = compatibility_theory or "未解析到配伍规律信息"
        responses["药材安全性分析"] = toxic_handling or "未解析到有毒药材处理信息"
        responses["配伍禁忌"] = incompatibility_warning or "未解析到配伍禁忌信息"
        responses["妊娠禁忌"] = pregnancy_warning or "未解析到妊娠禁忌信息"

        # 若提供了LLM判分器，则基于奖励模型参考答案进行判分
        if llm_judge_evaluator is not None:
            try:
                # 将处方组成转为带前缀的条目文本
                herbs_text = "\n".join([f"- {x}" for x in herbs_list]) if herbs_list else ""
                mapped_syndrome = self._map_syndrome_response(syndrome_response, case)
                # 生成四维度参考答案（奖励模型）
                ref_json = self._generate_four_dim_references(case, mapped_syndrome, herbs_text)
                references = {
                    "compatibility": ref_json.get("compatibility", ""),
                    "safety": ref_json.get("safety", ""),
                    "incompatibility": ref_json.get("incompatibility", ""),
                    "pregnancy": ref_json.get("pregnancy", ""),
                }
                model_outputs = {
                    "compatibility": responses["方剂配伍规律"],
                    "safety": responses["药材安全性分析"],
                    "incompatibility": responses["配伍禁忌"],
                    "pregnancy": responses["妊娠禁忌"],
                }
                judged_scores = llm_judge_evaluator.judge_prescription_by_reference(
                    case, mapped_syndrome, herbs_text, references, model_outputs
                )
                # 写入四维度得分
                for k, v in judged_scores.items():
                    scores[k] = v
            except Exception as e:
                logger.error(f"LLM参考判分失败，降级为0分：{e}")
                scores["方剂配伍规律"] = 0.0
                scores["安全性方面"] = 0.0
                scores["配伍禁忌"] = 0.0
                scores["妊娠禁忌"] = 0.0
        else:
            # 没有提供判分器时，保底全部置0分（避免沿用旧规则）
            scores["方剂配伍规律"] = 0.0
            scores["安全性方面"] = 0.0
            scores["配伍禁忌"] = 0.0
            scores["妊娠禁忌"] = 0.0

        return scores, responses

    def _build_fangzheng_match_prompt(self, case: Dict[str, Any], mapped_syndrome: str, herbs_text: str) -> str:
        return f"""\\no_think 你是一位中医方证匹配的专家。请基于以下病例、证型判断与处方组成，根据以下方剂与证型的匹配程度规则，判断该方剂与证型的匹配程度，并在最后对这个方剂进行打分：

病例：{case["instruction"]}

证型判断：
{mapped_syndrome}

处方药材：
{herbs_text}

1）方剂与证型的匹配程度规则如下：
   - `高度匹配`（匹配分数≥90分）：方剂与证型高度匹配。药物配伍与患者症状高度契合，该方剂可以使用，并且药物组成上不需要优化。
   - `匹配程度较高`（75分≤匹配分数≤89分）：方剂与证型匹配程度较高。药物配伍与患者症状较高契合，虽然该方剂可以使用，但是可以进行优化。
   - `匹配程度一般`（60分≤匹配分数≤74分）：方剂与证型匹配程度一般。虽然该方剂可以使用，但是部分药物与患者症状不匹配。
   - `不匹配`（匹配分数≤60分）：方剂与证型不匹配。该方剂不能使用，方剂的功能主治与患者症状完全相悖。
2）**严格且仅返回一个 JSON 对象，不要有任何多余说明或文字**。JSON 必须包含且仅包含两个字段：`匹配程度`（字符串）和`匹配分数`（整数）。
3）示例输出：
{{"匹配程度": "不匹配", "匹配分数": 36}}"""

    def _build_four_dim_reference_prompt(self, case: Dict[str, Any], mapped_syndrome: str, herbs_text: str) -> str:
        return f"""\\no_think 你是一位中药方剂分析专家，请对于以下病例、证型判断与处方组成，从方剂配伍规律、安全性方面、配伍禁忌、妊娠禁忌这四个维度进行分析。请严格遵循下列生成要求，并仅输出严格JSON（字段均为字符串，无多余解释）：

病例描述：\n{case["instruction"]}

证型判断：\n{mapped_syndrome}

处方组成：\n{herbs_text}

生成要求：
- 方剂配伍规律：配伍的理论依据和规律说明
- 有毒药材安全性分析：请分析处方中是否含有有毒药材。如果含有有毒药材，请详细说明处理方式和注意事项
- 配伍禁忌分析：请分析处方中是否违反十八反、十九畏配伍禁忌。如果存在配伍禁忌，请详细说明违反的具体配伍组合
- 妊娠安全性分析：请分析处方中是否含有妊娠禁忌药材。如果含有妊娠禁忌药材，请详细说明警示内容

仅输出如下JSON（不要包含其他文字）：
{{
  "compatibility": "<方剂配伍规律说明文本>",
  "safety": "<有毒药材及安全性处理说明文本>",
  "incompatibility": "<配伍禁忌说明文本>",
  "pregnancy": "<妊娠禁忌说明文本>"
}}"""

    def _generate_four_dim_references(self, case: Dict[str, Any], mapped_syndrome: str, herbs_text: str) -> Dict[str, str]:
        """调用奖励模型生成四维度的参考文本答案（JSON）。"""
        prompt = self._build_four_dim_reference_prompt(case, mapped_syndrome, herbs_text)
        pbar_placeholder = logging.getLogger(__name__)
        pbar_placeholder.info("调用奖励模型生成四维度参考答案...")
        return self._call_reward_model_json(prompt)

    # =================== JSON解析 ===================
    def _call_reward_model_json(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """调用奖励模型并解析为JSON。"""
        import json as _json
        last_err: Exception = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                    stream=False,
                )
                content = response.choices[0].message.content if response.choices else ""
                if not content:
                    raise RuntimeError("空响应")
                # 提取首个JSON块
                m = re.search(r"\{[\s\S]*\}", content)
                if not m:
                    raise ValueError("未找到JSON块")
                return _json.loads(m.group(0))
            except Exception as e:
                last_err = e
                logger.warning(f"奖励模型JSON调用失败（{attempt+1}/{max_retries}）：{e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise Exception(f"奖励模型JSON解析失败: {last_err}")
    
    def _parse_prescription_response(self, response: str) -> Dict[str, Any]:
        """
        解析处方响应
        
        Args:
            response: 处方响应文本
            
        Returns:
            解析后的处方信息
        """
        prescription_info = {
            "herbs": [],  # 药材列表：[(名称, 用量, 处理方法), ...]
            "toxic_handling": "",  # 有毒药材处理说明
            "pregnancy_warning": "",  # 妊娠禁忌警示
            "incompatibility_warning": "",  # 配伍禁忌说明
            "compatibility_theory": "",  # 配伍规律
            "raw_response": response
        }
        
        try:
            if "## Thinking" in response and "## Final Response" in response:
                # 提取## Final Response标签后的内容
                final_response_start = response.find("## Final Response")
                response_to_parse = response[final_response_start + len("## Final Response"):].strip()
            # 如果是传统格式（[[THINK]]标签）
            elif "<think>" in response and "</think>" in response:
                # 提取[[THINK]]标签外的内容
                parts = response.split("<think>")
                outside_think = parts[0]  # [[THINK]]之前的内容
                for part in parts[1:]:
                    if "</think>" in part:
                        outside_think += part.split("</think>", 1)[1]  # [[/THINK]]之后的内容
                    else:
                        outside_think += part  # 如果没有闭合标签，保留剩余内容
                
                # 如果[[THINK]]标签外没有内容，则处理整个响应
                response_to_parse = outside_think.strip() if outside_think.strip() else response
            else:
                response_to_parse = response
            
            # 提取处方组成
            herbs_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*处方组成\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:方剂配伍规律|有毒药材安全性分析|妊娠安全性分析|配伍禁忌分析)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            herbs_match = re.search(herbs_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if herbs_match:
                herbs_text = herbs_match.group(1)
                prescription_info["herbs"] = self._parse_herbs(herbs_text)
            
            # 提取配伍规律
            compatibility_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*方剂配伍规律\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:有毒药材安全性分析|妊娠安全性分析|配伍禁忌分析)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            compatibility_match = re.search(compatibility_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if compatibility_match:
                prescription_info["compatibility_theory"] = compatibility_match.group(1).strip()
            
            # 提取有毒药材安全性分析
            toxic_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*有毒药材安全性分析\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:妊娠安全性分析|配伍禁忌分析)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            toxic_match = re.search(toxic_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if toxic_match:
                toxic_content = toxic_match.group(1).strip()
                prescription_info["toxic_handling"] = toxic_content
            
            # 提取妊娠安全性分析
            pregnancy_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*妊娠安全性分析\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:配伍禁忌分析)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            pregnancy_match = re.search(pregnancy_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if pregnancy_match:
                pregnancy_content = pregnancy_match.group(1).strip()
                prescription_info["pregnancy_warning"] = pregnancy_content
            
            # 提取配伍禁忌分析
            incompatibility_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*配伍禁忌分析\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:妊娠安全性分析|有毒药材安全性分析|方剂配伍规律)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            incompatibility_match = re.search(incompatibility_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if incompatibility_match:
                incompatibility_content = incompatibility_match.group(1).strip()
                prescription_info["incompatibility_warning"] = incompatibility_content
            
        except Exception as e:
            logger.warning(f"解析处方响应时出错: {e}")
        
        return prescription_info
    
    def _parse_herbs(self, herbs_text: str) -> List[Tuple[str, str, str]]:
        """
        解析药材列表
        
        Args:
            herbs_text: 药材文本
            
        Returns:
            药材列表：[(名称, 用量, 处理方法), ...]
        """
        herbs = []
        
        # 按行分割
        lines = herbs_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('-') == False:
                continue
            
            # 移除开头的 "-" 或 "•"
            line = re.sub(r'^[-•]\s*', '', line)
            
            # 尝试解析：药材名称 用量 处理方法
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                dosage = parts[1]
                processing = ' '.join(parts[2:]) if len(parts) > 2 else ""
                herbs.append((name, dosage, processing))
        
        return herbs
    
    def _get_or_create_prescription(self, case: Dict[str, Any], 
                                   syndrome_response: str, 
                                   model_interface, pbar: tqdm) -> str:
        """按案例缓存处方，若无则生成一次。"""
        case_id = case.get("case_id") or case.get("id") or "unknown_case"
        if case_id in self.prescription_cache:
            return self.prescription_cache[case_id].get("raw_response", "")
        
        # 生成
        prompt = self._build_prescription_prompt(case, syndrome_response)
        pbar.write("正在生成处方...")
        response = model_interface.generate(prompt, max_tokens=4096, temperature=0.0)
        self.prescription_cache[case_id] = {"raw_response": response}
        return response

    def _build_prescription_prompt(self, case: Dict[str, Any], syndrome_response: str) -> str:
        """根据病例与证型（将字母映射为选项内容）构建处方提示词。"""
        mapped_syndrome = self._map_syndrome_response(syndrome_response, case)
        return f"""\\no_think 你是一位中医专家，请根据以下病例和证型判断，给出完整的处方。

病例描述：
{case["instruction"]}

证型判断：
{mapped_syndrome}

请按以下格式给出处方：

**处方组成**：
- 药材名称 用量g 处理方法（如果有）
- 药材名称 用量g 处理方法（如果有）
...

**方剂配伍规律**：
配伍的理论依据和规律说明

**有毒药材安全性分析**：
请分析处方中是否含有有毒药材。如果含有有毒药材，请详细说明处理方式和注意事项

**妊娠安全性分析**：
请分析处方中是否含有妊娠禁忌药材。如果含有妊娠禁忌药材，请详细说明警示内容

**配伍禁忌分析**：
请分析处方中是否违反十八反、十九畏配伍禁忌。如果存在配伍禁忌，请详细说明违反的具体配伍组合

"""

    def _map_syndrome_response(self, syndrome_response: str, case: Dict[str, Any]) -> str:
        """将模型返回的证型字母映射为选项内容，若无法映射则返回原文。支持多选如 A;C;E。"""
        try:
            options_text = case.get("output", {}).get("证型选项", "")
            if not options_text:
                return syndrome_response
            # 解析 A:内容;B:内容;...
            mapping: Dict[str, str] = {}
            for part in options_text.split(';'):
                if ':' in part:
                    label, content = part.split(':', 1)
                    mapping[label.strip().upper()] = content.strip()
            # 提取字母
            letters = [l.strip().upper() for l in re.findall(r"[A-J]", syndrome_response.upper())]
            contents = [mapping.get(l, l) for l in letters]
            return "; ".join(contents) if contents else syndrome_response
        except Exception:
            return syndrome_response
