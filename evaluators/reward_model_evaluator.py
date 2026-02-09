#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import time
import logging
import re
from typing import Dict, Any, Tuple, List, Optional
from tqdm import tqdm
from openai import OpenAI

logger = logging.getLogger(__name__)

class RewardModelEvaluator:
    """Reward-model-based evaluator."""
    
    def __init__(self, api_host: str, api_port: int, model_name: str = "Fangzheng-RM", api_key: Optional[str] = None):
        """
        Initialize the reward model evaluator.
        
        Args:
            api_host: API host
            api_port: API port
            model_name: Reward model name
        """
        self.api_host = api_host
        self.api_port = api_port
        self.api_base_url = f"http://{api_host}:{api_port}/v1"
        self.model_name = model_name
        self.api_key = api_key or "dummy-key"
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
        )
        
        # Reward-model dimensions (only the formula-match dimension is shown; the other four are grouped under LLM-judged outputs)
        self.reward_dimensions = [
            "方证契合度", "方剂配伍规律", "药材安全性分析", "配伍禁忌", "妊娠禁忌"
        ]
        
        # Per-case prescription cache
        self.prescription_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"初始化奖励模型评测器: {self.api_base_url} 模型: {self.model_name}")
    
    # =================== Combined evaluation (five dimensions) ===================
    def evaluate_all(self, case: Dict[str, Any], model_interface, syndrome_response: str, pbar: tqdm, llm_judge_evaluator=None) -> Tuple[Dict[str, float], Dict[str, str]]:
        # Generate or retrieve prescription
        prescription_response = self._get_or_create_prescription(case, syndrome_response, model_interface, pbar)
        prescription_info = self._parse_prescription_response(prescription_response)          
        # Organize parsed fields
        herbs_list = [f"{h[0]} {h[1]} {h[2]}".strip() for h in prescription_info.get("herbs", [])]
        compatibility_theory = prescription_info.get("compatibility_theory", "").strip()
        toxic_handling = prescription_info.get("toxic_handling", "").strip()
        incompatibility_warning = prescription_info.get("incompatibility_warning", "").strip()
        pregnancy_warning = prescription_info.get("pregnancy_warning", "").strip()

        mapped_syndrome = self._map_syndrome_response(syndrome_response, case)
        herbs_text = "\n".join([f"- {x}" for x in herbs_list]) if herbs_list else ""
        match_prompt = self._build_fangzheng_match_prompt(case, mapped_syndrome, herbs_text)
        pbar.write("正在调用奖励模型进行方证匹配评估...")
        result = self._call_reward_model_json(match_prompt)

        # Assemble scores following the existing rule
        scores: Dict[str, float] = {}
        responses: Dict[str, str] = {}

        # Formula-match dimension: normalize reward-model match score (0-100) to 0-1
        raw_match_score = result.get("匹配分数", 50)
        try:
            raw_match_score = float(raw_match_score)
        except Exception:
            raw_match_score = 50.0
        scores["方证契合度"] = max(0.0, min(1.0, raw_match_score / 100.0))
        # For display: extract the evaluated model's generated prescription composition
        herbs_content = "\n".join([f"- {h[0]} {h[1]} {h[2]}".strip() for h in prescription_info.get("herbs", [])])
        responses["方证契合度"] = herbs_content if herbs_content else "未解析到处方组成信息"

        # Collect evaluated model outputs (text)
        responses["方剂配伍规律"] = compatibility_theory or "未解析到配伍规律信息"
        responses["药材安全性分析"] = toxic_handling or "未解析到有毒药材处理信息"
        responses["配伍禁忌"] = incompatibility_warning or "未解析到配伍禁忌信息"
        responses["妊娠禁忌"] = pregnancy_warning or "未解析到妊娠禁忌信息"

        # If an LLM judge is provided, score against reward-model-generated references
        if llm_judge_evaluator is not None:
            try:
                # Convert prescription list to bullet items
                herbs_text = "\n".join([f"- {x}" for x in herbs_list]) if herbs_list else ""
                mapped_syndrome = self._map_syndrome_response(syndrome_response, case)
            # Generate 4-dimension reference answers (reward model)
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
                # Write back 4-dimension scores
                for k, v in judged_scores.items():
                    scores[k] = v
            except Exception as e:
                logger.error(f"LLM参考判分失败，降级为0分：{e}")
                scores["方剂配伍规律"] = 0.0
                scores["安全性方面"] = 0.0
                scores["配伍禁忌"] = 0.0
                scores["妊娠禁忌"] = 0.0
        else:
            # If no judge is provided, default to 0
            scores["方剂配伍规律"] = 0.0
            scores["安全性方面"] = 0.0
            scores["配伍禁忌"] = 0.0
            scores["妊娠禁忌"] = 0.0

        return scores, responses

    def _build_fangzheng_match_prompt(self, case: Dict[str, Any], mapped_syndrome: str, herbs_text: str) -> str:
        return f"""你是一位中医方证匹配的专家。请基于以下病例、证型判断与处方组成，根据以下方剂与证型的匹配程度规则，判断该方剂与证型的匹配程度，并在最后对这个方剂进行打分：

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
        return f"""你是一位中药方剂分析专家，请对于以下病例、证型判断与处方组成，从方剂配伍规律、安全性方面、配伍禁忌、妊娠禁忌这四个维度进行分析。请严格遵循下列生成要求，并仅输出严格JSON（字段均为字符串，无多余解释）：

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
        """Call the reward model to generate JSON reference texts for four dimensions."""
        prompt = self._build_four_dim_reference_prompt(case, mapped_syndrome, herbs_text)
        pbar_placeholder = logging.getLogger(__name__)
        pbar_placeholder.info("调用奖励模型生成四维度参考答案...")
        return self._call_reward_model_json(prompt)

    # =================== JSON parsing ===================
    def _call_reward_model_json(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Call the reward model and parse the response as JSON."""
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
                # Extract the first JSON block
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
        Parse a prescription response.
        
        Args:
            response: Raw prescription response text
            
        Returns:
            Parsed prescription info
        """
        prescription_info = {
            "herbs": [],  # list of herbs: [(name, dosage, processing), ...]
            "toxic_handling": "",  # toxic herb handling notes
            "pregnancy_warning": "",  # pregnancy contraindication warning
            "incompatibility_warning": "",  # incompatibility/contraindication notes
            "compatibility_theory": "",  # compatibility theory
            "raw_response": response
        }
        
        try:
            if "## Thinking" in response and "## Final Response" in response:
                final_response_start = response.find("## Final Response")
                response_to_parse = response[final_response_start + len("## Final Response"):].strip()
            elif "<think>" in response and "</think>" in response:
                parts = response.split("<think>")
                outside_think = parts[0]  
                for part in parts[1:]:
                    if "</think>" in part:
                        outside_think += part.split("</think>", 1)[1]  
                    else:
                        outside_think += part  
                
                response_to_parse = outside_think.strip() if outside_think.strip() else response
            else:
                response_to_parse = response
            
            # Extract prescription composition
            herbs_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*处方组成\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:方剂配伍规律|有毒药材安全性分析|妊娠安全性分析|配伍禁忌分析)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            herbs_match = re.search(herbs_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if herbs_match:
                herbs_text = herbs_match.group(1)
                prescription_info["herbs"] = self._parse_herbs(herbs_text)
            
            # Extract compatibility theory
            compatibility_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*方剂配伍规律\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:有毒药材安全性分析|妊娠安全性分析|配伍禁忌分析)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            compatibility_match = re.search(compatibility_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if compatibility_match:
                prescription_info["compatibility_theory"] = compatibility_match.group(1).strip()
            
            # Extract toxic-herb safety analysis
            toxic_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*有毒药材安全性分析\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:妊娠安全性分析|配伍禁忌分析)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            toxic_match = re.search(toxic_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if toxic_match:
                toxic_content = toxic_match.group(1).strip()
                prescription_info["toxic_handling"] = toxic_content
            
            # Extract pregnancy safety analysis
            pregnancy_pattern = r'^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*妊娠安全性分析\s*(?:\*\*|__)?\s*[：:]?\s*(.*?)(?=^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:配伍禁忌分析)\s*(?:\*\*|__)?\s*[：:]?\s*$|$\Z)'
            pregnancy_match = re.search(pregnancy_pattern, response_to_parse, re.DOTALL | re.MULTILINE)
            if pregnancy_match:
                pregnancy_content = pregnancy_match.group(1).strip()
                prescription_info["pregnancy_warning"] = pregnancy_content
            
            # Extract incompatibility analysis
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
        Parse herb list.
        
        Args:
            herbs_text: Herb section text
            
        Returns:
            List of herbs: [(name, dosage, processing), ...]
        """
        herbs = []
        
        # Split by lines
        lines = herbs_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('-') == False:
                continue
            
            # Remove leading "-" or "•"
            line = re.sub(r'^[-•]\s*', '', line)
            
            # Try parsing: name dosage processing
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
        """Cache prescription per case; generate once if missing."""
        case_id = case.get("case_id") or case.get("id") or "unknown_case"
        if case_id in self.prescription_cache:
            return self.prescription_cache[case_id].get("raw_response", "")
        
        # Generate
        prompt = self._build_prescription_prompt(case, syndrome_response)
        pbar.write("正在生成处方...")
        response = model_interface.generate(prompt, max_tokens=4096, temperature=0.0)
        self.prescription_cache[case_id] = {"raw_response": response}
        return response

    def _build_prescription_prompt(self, case: Dict[str, Any], syndrome_response: str) -> str:
        """Build the prescription prompt from the case and syndrome (map letters to option contents)."""
        mapped_syndrome = self._map_syndrome_response(syndrome_response, case)
        return f"""你是一位中医专家，请根据以下病例和证型判断，给出完整的处方。

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
        """Map syndrome letters returned by the model to option contents.

        If mapping fails, return the original text. Supports multi-select like A;C;E.
        """
        try:
            options_text = case.get("output", {}).get("证型选项", "")
            if not options_text:
                return syndrome_response
            # Parse A:content;B:content;...
            mapping: Dict[str, str] = {}
            for part in options_text.split(';'):
                if ':' in part:
                    label, content = part.split(':', 1)
                    mapping[label.strip().upper()] = content.strip()
            # Extract letters
            letters = [l.strip().upper() for l in re.findall(r"[A-J]", syndrome_response.upper())]
            contents = [mapping.get(l, l) for l in letters]
            return "; ".join(contents) if contents else syndrome_response
        except Exception:
            return syndrome_response
