#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import time
import logging
import re
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMJudgeEvaluator:
    """LLM判分评测器"""
    
    def __init__(self, api_host: str, api_port: int, model_name: str = "Qwen3-32B", api_key: Optional[str] = None):
        """
        初始化LLM判分评测器
        
        Args:
            api_host: API主机地址
            api_port: API端口
            model_name: 判分所用模型名称
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
        
        # LLM判分维度
        self.llm_judge_dimensions = [
            "煎服方法", "注意事项", "CoT内容完备性", "随症加减"
        ]
        
        logger.info(f"初始化LLM判分评测器: {self.api_base_url} 模型: {self.model_name}")
    
    def evaluate_all(self, case: Dict[str, Any], model_interface, pbar: tqdm, syndrome_choice: str = None, prescription_herbs: str = None, treatment_principles: str = None, skip_think: bool = False) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        返回：(scores, responses)
        - scores：各维度得分
        - responses：各维度的原始生成（用于留档）
        - syndrome_choice：证型的具体内容，用于注意事项和随症加减评分（若为空，则回退到 case['output']['证型']）
        - prescription_herbs：处方组成（用于煎服方法、注意事项和随症加减评分）（若为空，则回退到 case['output']['药物组成及用量']）
        - treatment_principles：治则治法内容（用于煎服方法生成）（若为空，则回退到 case['output']['治则治法']）
        - skip_think：是否跳过CoT内容完备性评测
        """
        pbar.write("正在进行LLM分维度评测...")

        # 分别生成各维度内容
        parsed_content = {}

        # 1. CoT完备性评测 - 仅需instruction（根据skip_think参数决定）
        if not skip_think:
            pbar.write("  生成CoT内容...")
            think_content = self._generate_think_content(case, model_interface)
            parsed_content["think_content"] = think_content
        else:
            pbar.write("  跳过CoT内容完备性评测")
            parsed_content["think_content"] = ""

        # 2. 煎服方法生成 
        pbar.write("  生成煎服方法 ...")
        fallback_output = case.get("output", {}) or {}
        source_prescription = prescription_herbs if prescription_herbs else fallback_output.get("药物组成及用量", "")
        source_treatment_principles = treatment_principles if treatment_principles else fallback_output.get("治则治法", "")
        cooking_method = self._generate_cooking_method(case, source_prescription, model_interface, source_treatment_principles)
        parsed_content["cooking_method"] = cooking_method

        # 3. 注意事项和随症加减合并生成 
        pbar.write("  生成注意事项和随症加减 ...")
        source_syndrome = syndrome_choice if syndrome_choice else fallback_output.get("证型", "")
        source_prescription_for_prec = source_prescription  # 与上保持一致；若为空则已回退
        if not source_prescription_for_prec:
            source_prescription_for_prec = fallback_output.get("药物组成及用量", "")
        precautions_and_modifications = self._generate_precautions_and_modifications(case, source_syndrome, source_prescription_for_prec, model_interface)
        parsed_content["precautions"] = precautions_and_modifications.get("precautions", "")
        parsed_content["syndrome_modifications"] = precautions_and_modifications.get("syndrome_modifications", "")
        
        # 调用评分模型完成维度评分
        pbar.write("  进行LLM评分...")
        scores = self._call_combined_llm_judge(parsed_content, case, skip_think=skip_think)
        
        # 构造返回的responses（根据skip_think决定格式）
        responses = {
            "煎服方法": parsed_content.get("cooking_method", ""),
            "注意事项": parsed_content.get("precautions", ""),
            "随症加减": parsed_content.get("syndrome_modifications", "")
        }
        
        # 只有当不跳过CoT评测时才添加该维度
        if not skip_think:
            responses["CoT内容完备性"] = parsed_content.get("think_content", "")
        
        return scores, responses

    def _generate_think_content(self, case: Dict[str, Any], model_interface) -> str:
        """生成CoT内容完备性评测内容"""
        prompt = f"""你是一位中医专家，请根据以下病例的症状，进行中医辨证并提供治疗药方。

病例描述：
{case["instruction"]}

"""
        
        try:
            response = model_interface.generate(prompt, max_tokens=4096, temperature=0.0, 
                                              clean_think=False)
            
            try:
                if hasattr(model_interface, "get_last_reasoning"):
                    reasoning_text = model_interface.get_last_reasoning()
                    if isinstance(reasoning_text, str) and reasoning_text.strip():
                        return reasoning_text.strip()
            except Exception:
                pass
            if "## Thinking" in response and "## Final Response" in response:
                thinking_start = response.find("## Thinking")
                final_response_start = response.find("## Final Response")
                if thinking_start != -1 and final_response_start != -1 and thinking_start < final_response_start:
                    think_content = response[thinking_start + len("## Thinking"):final_response_start].strip()
                    return think_content
                else:
                    logger.warning("标签位置不正确")
                    return ""
            else:
                if "</think>" in response and "<think>" not in response:
                    return response.split("</think>", 1)[0].strip()
                think_pattern = r'<think>(.*?)(?:</think>|$)'
                think_match = re.search(think_pattern, response, re.DOTALL)
                if think_match:
                    return think_match.group(1).strip()
                else:
                    logger.warning("未找到think标签内容")
                    return ""
                
        except Exception as e:
            logger.error(f"生成CoT内容时出错: {e}")
            return ""
    
    def _generate_cooking_method(self, case: Dict[str, Any], prescription_herbs: str, model_interface, treatment_principles: Optional[str] = None) -> str:
        """生成煎服方法"""
        herbs_info = f"\n\n处方组成：{prescription_herbs}" if prescription_herbs else ""
        tp_info = f"\n\n治则治法：{treatment_principles}" if treatment_principles else ""
        
        prompt = f"""你是一位中医专家，请根据以下病例、处方组成和治则治法，制定详细的煎服方法。

病例描述：
{case["instruction"]}{herbs_info}{tp_info}

请按以下要求制定煎服方法：

煎服方法根据处方组成部分考虑方剂配伍结构以及各药材特性，根据治则治法部分，综合治疗目的、患者病情、传统经验和临床实践等多因素结果，使用中医术语化，煎服按流程分数字小点列出。

"""
        
        try:
            response = model_interface.generate(prompt, max_tokens=4096, temperature=0.0, clean_think=True)
            if "## Final Response" in response:
                final_response_start = response.find("## Final Response")
                return response[final_response_start + len("## Final Response"):].strip()
            if "<think>" in response and "</think>" in response:
                parts = response.split("<think>")
                outside_think = parts[0]
                for part in parts[1:]:
                    if "</think>" in part:
                        outside_think += part.split("</think>", 1)[1]
                    else:
                        outside_think += part
                return outside_think.strip() if outside_think.strip() else response.strip()
            return response.strip()
        except Exception as e:
            logger.error(f"生成煎服方法时出错: {e}")
            return ""
    
    def _generate_precautions_and_modifications(self, case: Dict[str, Any], syndrome_choice: str, prescription_herbs: str, model_interface) -> Dict[str, str]:
        """合并生成注意事项和随症加减（一次模型调用）"""
        context_info = ""
        if syndrome_choice:
            context_info += f"\n\n证型诊断：{syndrome_choice}"
        if prescription_herbs:
            context_info += f"\n\n处方组成：{prescription_herbs}"

        prompt = f"""你是一位中医专家，请根据以下病例、证型诊断和处方组成，制定随症加减方案并给出注意事项。

病例描述：
{case["instruction"]}{context_info}

请按照以下格式输出：

【随症加减】
请制定全面的随症加减方案。

【注意事项】
请给出全面的注意事项

请严格按照【注意事项】和【随症加减】的标题格式输出。"""

        try:
            response = model_interface.generate(prompt, max_tokens=4096, temperature=0.0)
            return self._parse_precautions_and_modifications_response(response)
        except Exception as e:
            logger.error(f"生成注意事项和随症加减时出错: {e}")
            return {"precautions": "", "syndrome_modifications": ""}

    def _parse_precautions_and_modifications_response(self, response: str) -> Dict[str, str]:
        """解析注意事项和随症加减的合并响应"""
        result = {"precautions": "", "syndrome_modifications": ""}
        try:
            text = response.strip()
            prec_label = "【注意事项】"
            mod_label = "【随症加减】"
            len_prec = len(prec_label)
            len_mod = len(mod_label)

            idx_prec = text.find(prec_label)
            idx_mod = text.find(mod_label)

            if idx_prec == -1 and idx_mod == -1:
                lines = text.split('\n')
                half = len(lines) // 2
                result["syndrome_modifications"] = '\n'.join(lines[:half]).strip()
                result["precautions"] = '\n'.join(lines[half:]).strip()
                return result

            if idx_prec != -1 and idx_mod != -1:
                if idx_mod < idx_prec:
                    result["syndrome_modifications"] = text[idx_mod + len_mod: idx_prec].strip()
                    result["precautions"] = text[idx_prec + len_prec:].strip()
                else:
                    result["precautions"] = text[idx_prec + len_prec: idx_mod].strip()
                    result["syndrome_modifications"] = text[idx_mod + len_mod:].strip()
                return result

            if idx_prec != -1 and idx_mod == -1:
                before = text[:idx_prec].strip()
                after_prec = text[idx_prec + len_prec:].strip()
                if before:
                    result["syndrome_modifications"] = before
                result["precautions"] = after_prec
                return result

            if idx_mod != -1 and idx_prec == -1:
                before = text[:idx_mod].strip()
                after_mod = text[idx_mod + len_mod:].strip()
                if before:
                    result["precautions"] = before
                result["syndrome_modifications"] = after_mod
                return result

            return result
        except Exception as e:
            logger.error(f"解析注意事项和随症加减响应时出错: {e}")
            return {"precautions": text if 'text' in locals() else response.strip(), "syndrome_modifications": ""}

    def _call_combined_llm_judge(self, parsed_content: Dict[str, str], case: Dict[str, Any], skip_think: bool = False) -> Dict[str, float]:
        """根据skip_think参数完成相应维度的评分"""
        try:
            # 获取标准答案
            gt_cook = case["output"]["煎服方法"]
            gt_note = case["output"]["注意事项"]
            gt_modifications = case["output"]["随症加减"]
            
            # 构建基础提示词
            dimensions_count = 3 if skip_think else 4
            prompt_header = f"""你是一位资深中医评审专家，请根据病例描述，分别对以下{dimensions_count}个维度中待评估的内容按照各自的评分要点进行评分，并仅输出严格JSON。

病例：{case["instruction"]}

待评估内容（部分维度含标准答案）：

"""
            
            # 根据skip_think决定是否包含CoT内容完备性评分
            dimension_sections = []
            dimension_counter = 1
            
            if not skip_think:
                think_section = f"""{dimension_counter}) CoT内容完备性
待评估（仅评估下述思考内容）：
{parsed_content.get("think_content")}

评分要点（必须依据病例）：
判断待评估的思考过程是否完整使用病例中的关键信息要素并与辨证推理相关联，包括但不限于：性别、年龄、职业/身份、就诊或发病时间、季节/气候、诱因与生活事件、主要症状体征、舌脉所见、病程与变化等。覆盖率越高、引用越准确得分越高。

"""
                dimension_sections.append(think_section)
                dimension_counter += 1
            
            # 随症加减维度
            modifications_section = f"""{dimension_counter}) 随症加减
待评估：{parsed_content.get("syndrome_modifications")}
标准答案：{gt_modifications}

评分要点（必须依据病例并与标准答案对比）：
- 完整性与覆盖：判断与标准答案中各要点的覆盖率，覆盖率越高得分越高；对于待评估答案中不一致的要点，应合理且不与标准答案冲突。
- 机理与合理性：所选药需明确药名与剂量，要与证机、主症相符，功效说明准确，术语规范；与基础方配伍协调，不自相矛盾。

"""
            dimension_sections.append(modifications_section)
            dimension_counter += 1
            
            # 煎服方法维度
            cooking_section = f"""{dimension_counter}) 煎服方法
待评估：{parsed_content.get("cooking_method")}
标准答案：{gt_cook}

评分要点（必须依据病例并与标准答案对比）：
- 器具与禁忌：是否明确砂锅/陶瓷等合适器具，如有忌用器具是否给出。
- 药材处理：是否对药材进行正确处理。
- 步骤与参数：是否给出分次煎煮的关键步骤、加水量与火候/时长，如有两煎合并、滤清等关键节点是否说明。
- 服用方法：给出的每日剂量、分次/时机、每次服用量及配合的生活提示是否合理。
- 一致性与可执行性：与标准答案在关键原则上保持一致；允许合理等效表达与小幅参数差异，但必须完整覆盖关键要点，表达步骤化、可操作。

"""
            dimension_sections.append(cooking_section)
            dimension_counter += 1
            
            # 注意事项维度
            precautions_section = f"""{dimension_counter}) 注意事项
待评估：{parsed_content.get("precautions")}
标准答案：{gt_note}

评分要点（必须依据病例并与标准答案对比）：
- 表达与一致性：与标准答案不冲突，结构清晰、术语准确、合理即可。

"""
            dimension_sections.append(precautions_section)
            
            # 构建JSON输出格式要求
            json_keys = []
            if not skip_think:
                json_keys.append('"think_completeness": 0-100')
            json_keys.extend([
                '"syndrome_modifications": 0-100',
                '"cooking_method": 0-100',
                '"precautions": 0-100'
            ])
            
            json_format = "{\n  " + ",\n  ".join(json_keys) + "\n}"
            
            # 组装完整的提示词
            prompt = prompt_header + "".join(dimension_sections) + f"""
输出严格JSON（仅包含下列键，值为0-100的整数，不要输出任何解释）：
{json_format}"""
            
            response = self._call_qwen_api(prompt)
            result = self._parse_json_response(response)
            
            # 转换为0-1分数（根据skip_think参数决定包含的维度）
            scores = {
                "煎服方法": max(0.0, min(1.0, result.get("cooking_method", 50) / 100.0)),
                "注意事项": max(0.0, min(1.0, result.get("precautions", 50) / 100.0)),
                "随症加减": max(0.0, min(1.0, result.get("syndrome_modifications", 50) / 100.0))
            }
            
            # 只有当不跳过CoT评测时才添加该维度分数
            if not skip_think:
                scores["CoT内容完备性"] = max(0.0, min(1.0, result.get("think_completeness", 50) / 100.0))
            
            return scores
            
        except Exception as e:
            logger.error(f"合并LLM判分时出错: {e}")
            # 根据skip_think参数返回相应的默认分数
            error_scores = {
                "煎服方法": 0.0,
                "注意事项": 0.0,
                "随症加减": 0.0
            }
            if not skip_think:
                error_scores["CoT内容完备性"] = 0.0
            return error_scores

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析JSON响应"""
        import json as _json
        try:
            # 提取JSON块
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return _json.loads(json_match.group(0))
        except Exception as e:
            logger.warning(f"解析JSON响应失败: {e}")
        
        # 回退到关键词提取
        result = {
            "think_completeness": 50,
            "cooking_method": 50,
            "precautions": 50,
            "syndrome_modifications": 50,
            # 兼容病因/病机评分键
            "cause": 50,
            "mechanism": 50,
            # 兼容处方四维度评分键
            "compatibility": 50,
            "safety": 50,
            "incompatibility": 50,
            "pregnancy": 50
        }
        
        # 尝试从文本中提取分数
        for key in result.keys():
            pattern = rf'"{key}"\s*:\s*(\d+)'
            match = re.search(pattern, response)
            if match:
                try:
                    result[key] = int(match.group(1))
                except ValueError:
                    pass
        
        return result

    def _call_qwen_api(self, prompt: str, max_retries: int = 3) -> str:
        """调用Qwen API"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                    stream=False,
                )
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    if content:
                        return content.strip()
                logger.warning("Qwen API返回空内容")
            except Exception as e:
                logger.warning(f"Qwen API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise Exception(f"Qwen API调用失败，已重试 {max_retries} 次")

    def evaluate_cause_mechanism(self, case: Dict[str, Any], model_interface, pbar: tqdm) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        使用待评测模型生成病因/病机，然后由评分LLM对照标准答案进行评分。
        """
        try:
            # 1) 让被测模型生成病因/病机
            prompt = self._build_cause_mechanism_prompt(case["instruction"])
            pbar.write("正在生成病因/病机内容用于LLM评分...")
            response = model_interface.generate(prompt, max_tokens=4096, temperature=0.0)
            parsed = self._parse_cause_mechanism_response(response)

            # 2) 由评分LLM对照标准答案评分
            pbar.write("正在进行病因/病机的LLM评分...")
            scores = self._call_cause_mechanism_judge(parsed, case)

            # 3) 返回分数与被测模型生成内容
            outputs = {
                "病因": parsed.get("病因", ""),
                "病机": parsed.get("病机", ""),
            }
            return scores, outputs

        except Exception as e:
            logger.error(f"病因/病机 LLM评测失败: {e}")
            return {"病因": 0.0, "病机": 0.0}, {"病因": f"评测失败: {e}", "病机": f"评测失败: {e}"}

    def _build_cause_mechanism_prompt(self, instruction: str) -> str:
        return f"""你是一位中医专家，请根据以下中医病例，分析其病因和病机。

病例描述：
{instruction}

病因分析要包括导致疾病发生的内因、外因等各种因素；病机分析要阐述疾病发生发展的病理机制和变化规律。
请严格按以下格式输出（不要输出其他内容）：

病因：在此撰写病因
病机：在此撰写病机
"""

    def _parse_cause_mechanism_response(self, response: str) -> Dict[str, str]:
        result = {"病因": "", "病机": ""}
        try:
            cause_start = response.find("病因：")
            mechanism_start = response.find("病机：")
            if cause_start != -1:
                if mechanism_start != -1 and mechanism_start > cause_start:
                    result["病因"] = response[cause_start + 3:mechanism_start].strip()
                else:
                    result["病因"] = response[cause_start + 3:].strip()
            if mechanism_start != -1:
                if cause_start == -1 or mechanism_start < cause_start:
                    result["病机"] = response[mechanism_start + 3:].strip()
                else:
                    tail = response[mechanism_start + 3:].strip()
                    result["病机"] = tail
            if not result["病因"] and not result["病机"]:
                lines = response.strip().split('\n')
                half = len(lines) // 2
                result["病因"] = '\n'.join(lines[:half]).strip()
                result["病机"] = '\n'.join(lines[half:]).strip()
        except Exception as e:
            logger.warning(f"解析病因/病机失败: {e}")
            result["病因"] = response.strip()
            result["病机"] = ""
        return result

    def _call_cause_mechanism_judge(self, parsed: Dict[str, str], case: Dict[str, Any]) -> Dict[str, float]:
        gt_cause = case["output"].get("病因", "")
        gt_mechanism = case["output"].get("病机", "")
        prompt = f"""你是一位资深中医评审专家，请根据病例描述，结合各自的标准答案，分别对以下两个维度中的待评估内容按照各自的评分要点进行评分，并仅输出严格JSON：

病例：{case["instruction"]}

1）病因
待评估：{parsed.get("病因")}
标准答案：{gt_cause}

评分要点（必须与标准答案对比）：
- 准确性：基于标准答案，判断待评估答案给出的病因是否合理、是否全面。
- 专业性：判断待评估答案是否符合中医理论的专业术语和规范表达。

2)病机
待评估：{parsed.get("病机")}
标准答案：{gt_mechanism}

评分要点（必须与标准答案对比）：
- 准确性：基于标准答案，判断待评估答案给出的病机推理是否合理、是否全面。
- 专业性：判断待评估答案是否符合中医理论的专业术语和规范表达。

输出JSON（不要包含其他文字）：
{{
  "cause": 0-100,
  "mechanism": 0-100
}}"""
        try:
            resp = self._call_qwen_api(prompt)
            result = self._parse_json_response(resp)
            cause_raw = result.get("cause", 50)
            mech_raw = result.get("mechanism", 50)
            scores = {
                "病因": max(0.0, min(1.0, float(cause_raw) / 100.0)),
                "病机": max(0.0, min(1.0, float(mech_raw) / 100.0)),
            }
            return scores
        except Exception as e:
            logger.error(f"病因/病机 判分失败: {e}")
            return {"病因": 0.0, "病机": 0.0}

    def judge_prescription_by_reference(
        self,
        case: Dict[str, Any],
        syndrome_choice: str,
        herbs_list_text: str,
        references: Dict[str, str],
        model_outputs: Dict[str, str],
    ) -> Dict[str, float]:
        prompt = f"""你是一位资深中医评审专家，请根据病例描述、证型判断以及处方组成，结合各自的标准答案，分别对以下四个维度中的待评估内容按照各自的评分要点进行评分，并仅输出严格JSON：

病例：{case.get("instruction", "")}
证型判断：{syndrome_choice}
处方组成：\n{herbs_list_text}

1) 方剂配伍规律
待评估内容：{model_outputs.get('compatibility')}
标准答案：{references.get('compatibility')}

评分要点（必须依据病例、证型判断、处方组成并与标准答案对比）：
- 准确性：基于标准答案，判断待评估答案给出的方剂配伍规律是否准确、是否全面。
- 专业性：判断待评估答案是否符合中医理论的专业术语和规范表达。

2) 安全性方面（有毒药材处理与安全性）
待评估内容：{model_outputs.get('safety')}
标准答案：{references.get('safety')}

评分要点（必须依据病例、证型判断、处方组成并与标准答案对比）：
- 准确性：基于标准答案，判断待评估答案给出的有毒药材的判别是否准确、是否全面，对于有毒药材的处理方式是否准确。
- 专业性：判断待评估答案是否符合中医理论的专业术语和规范表达。

3) 配伍禁忌
待评估内容：{model_outputs.get('incompatibility')}
标准答案：{references.get('incompatibility')}

评分要点（必须依据病例、证型判断、处方组成并与标准答案对比）：
- 准确性：基于标准答案，判断待评估答案给出的配伍禁忌是否准确、是否全面。
- 专业性：判断待评估答案是否符合中医理论的专业术语和规范表达。

4) 妊娠禁忌
待评估内容：{model_outputs.get('pregnancy')}
标准答案：{references.get('pregnancy')}

评分要点（必须依据病例、证型判断、处方组成并与标准答案对比）：
- 准确性：基于标准答案，判断待评估答案给出的妊娠禁忌是否准确、是否全面，对于给出的妊娠警示是否合理。
- 专业性：判断待评估答案是否符合中医理论的专业术语和规范表达。

输出JSON（不要包含其他文字）：
{{
  "compatibility": 0-100,
  "safety": 0-100,
  "incompatibility": 0-100,
  "pregnancy": 0-100
}}"""
        try:
            resp = self._call_qwen_api(prompt)
            result = self._parse_json_response(resp)
            scores = {
                "方剂配伍规律": max(0.0, min(1.0, float(result.get("compatibility", 50)) / 100.0)),
                "药材安全性分析": max(0.0, min(1.0, float(result.get("safety", 50)) / 100.0)),
                "配伍禁忌": max(0.0, min(1.0, float(result.get("incompatibility", 50)) / 100.0)),
                "妊娠禁忌": max(0.0, min(1.0, float(result.get("pregnancy", 50)) / 100.0)),
            }
            return scores
        except Exception as e:
            logger.error(f"奖励模型参考答案的LLM判分失败: {e}")
            return {"方剂配伍规律": 0.0, "药材安全性分析": 0.0, "配伍禁忌": 0.0, "妊娠禁忌": 0.0}

    def evaluate_hallucination(self, case: Dict[str, Any], think_content: str, 
                              pbar: tqdm) -> Tuple[float, Dict[str, Any]]:
        """
        评测CoT中的幻觉（CoT准确性）
        
        Args:
            case: 案例数据
            think_content: Think内容（已在Think完备性评测时生成，直接使用）
            pbar: 进度条
            
        Returns:
            (CoT准确性分数, 幻觉详细信息)
            - CoT准确性分数：0-1，用于 dimension_scores["CoT准确性"]
            - 幻觉详细信息：完整的检测结果，用于 hallucination_details
        """
        pbar.write("  评测CoT准确性（幻觉检测）...")
        
        try:
            instruction = case["instruction"]
            
            prompt = f"""你是一位严谨的中医临床专家和信息审核专家。
请仔细对比以下【病例描述】和【模型思考过程（CoT）】，识别CoT中所有提及的**事实性信息点**，
并判断每个信息点是否存在幻觉（即与病例描述不符或病例中未提及）。

【病例描述（instruction）】：
{instruction}

【模型思考过程（CoT）】：
{think_content}

【任务要求】：

1. **仅提取CoT中的事实性信息点（关于患者已有的信息）**：
   
   **应该提取的（事实性陈述）**：
   a) 患者基本信息：姓名、性别、年龄、职业/身份
   b) 时间信息：就诊时间、发病时间、病程、季节/气候
   c) 症状体征：主诉、现病史、刻下症状、舌象、脉象
   d) 病史信息：既往史、个人史、家族史、辅助检查结果
   e) 诱因与生活事件：发病诱因、已有的生活习惯、已有的情志因素
   
   **不应该提取的（非事实性内容，直接跳过）**：
   - ❌ 诊疗建议（如"建议避免辛辣"、"宜清淡饮食"）
   - ❌ 治疗方案（如"可用..."、"宜..."）
   - ❌ 医嘱指导（如"保持..."、"注意..."）
   - ❌ 中医理论推断（如"根据舌红判断为阴虚"）
   - ❌ 证型诊断（如"肝肾阴虚证"）
   - ❌ 病机分析（如"气滞血瘀"）
   - ❌ 对检查结果的医学判断（如"属于正常范围"）

2. **判断每个事实性信息点的准确性**：
   - ✅ **正确**：信息点在病例描述中明确提及，且描述一致
   - ❌ **幻觉-篡改**：信息点在病例中提及，但CoT的描述与病例不符
   - ❌ **幻觉-捏造**：信息点在病例描述中完全未提及
   
   **重要：以下情况不算幻觉**：
   - 合理的中医术语转换（如"舌红"→"舌质红"）
   - 合理的同义表达（如"头痛"→"头部疼痛"）
   - 对数值的合理归纳（如"175 nmol/L"→"正常"）
   - 对症状的专业归纳（如"口干、口苦"→"口干口苦"）

【输出格式】（必须严格JSON，不要有任何额外文字）：
{{
  "information_points": [
    {{
      "category": "患者基本信息/时间信息/症状体征/病史信息/诱因与生活事件",
      "point_description": "信息点的简要描述",
      "cot_content": "CoT中的原文表述",
      "instruction_content": "病例中的对应表述或'未提及'",
      "is_hallucination": true/false,
      "hallucination_type": "correct/modification/fabrication",
      "explanation": "判断理由（1句话）"
    }}
  ]
}}
"""
            
            # 调用API（流式输出）
            response = self._call_qwen_api_stream(prompt)
            
            # 解析并自动统计
            result = self._parse_hallucination_response(response)
            
            return result["hallucination_score"], result
            
        except Exception as e:
            logger.error(f"CoT准确性评测失败: {e}")
            return 0.0, {
                "total_info_points": 0,
                "hallucination_count": 0,
                "hallucination_rate": 0.0,
                "information_points": [],
                "overall_assessment": f"CoT准确性评测失败: {str(e)}"
            }
    
    def _call_qwen_api_stream(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8192,
            stream=True
        )
        
        full_content = []
        print("\n" + "="*60)
        print("🤖 CoT准确性评测:")
        print("="*60)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                full_content.append(content_piece)
                print(content_piece, end='', flush=True)
        
        print("\n" + "="*60 + "\n")
        
        return ''.join(full_content).strip()
    
    def _parse_hallucination_response(self, response: str) -> Dict[str, Any]:
        import json as _json
        
        try:
            # 提取JSON块
            json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response)
            if not json_match:
                json_match = re.search(r'\{[\s\S]*\}', response)
            
            if json_match:
                json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
                data = _json.loads(json_str)
                
                # 校验必需字段
                if "information_points" not in data:
                    raise ValueError("缺少必需字段: information_points")
                
                info_points = data["information_points"]
                if not isinstance(info_points, list):
                    raise ValueError("information_points 必须是列表")
                
                # 程序自动统计
                total_count = len(info_points)
                hallucination_count = sum(
                    1 for point in info_points 
                    if point.get("is_hallucination", False) == True
                )
                
                hallucination_rate = hallucination_count / total_count if total_count > 0 else 0.0
                
                # 统计幻觉类型
                modification_count = sum(
                    1 for point in info_points 
                    if point.get("hallucination_type") == "modification"
                )
                fabrication_count = sum(
                    1 for point in info_points 
                    if point.get("hallucination_type") == "fabrication"
                )
                
                # 生成 overall_assessment
                if "overall_assessment" not in data or not data["overall_assessment"]:
                    overall_assessment = (
                        f"CoT共提及{total_count}个信息点，"
                        f"其中{hallucination_count}个存在幻觉"
                        f"（{modification_count}个篡改，{fabrication_count}个捏造），"
                        f"幻觉率为{hallucination_rate:.2%}"
                    )
                else:
                    overall_assessment = data["overall_assessment"]
                
                logger.info(f"CoT准确性统计: {total_count}个信息点, {hallucination_count}个幻觉, 幻觉率={hallucination_rate:.2%}")
                
                return {
                    "total_info_points": total_count,
                    "hallucination_count": hallucination_count,
                    "hallucination_rate": hallucination_rate,
                    "hallucination_score": 1.0 - hallucination_rate,
                    "information_points": info_points,
                    "overall_assessment": overall_assessment
                }
            else:
                raise ValueError("未找到JSON块")
                
        except Exception as e:
            logger.error(f"解析幻觉检测响应失败: {e}")
            return {
                "total_info_points": 0,
                "hallucination_count": 0,
                "hallucination_rate": 0.0,
                "hallucination_score": 0.0,
                "information_points": [],
                "overall_assessment": f"解析失败: {str(e)}"
            }
