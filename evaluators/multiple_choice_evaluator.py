#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import re
import logging
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MultipleChoiceEvaluator:
    """选择题评测器（合并评测）"""
    
    def __init__(self, random_seed: int = 42):
        """初始化选择题评测器"""
        # 选择题配置
        self.choice_configs = {
            "证型": {"num_options": 10, "multiple": True},
            "病位": {"num_options": 10, "multiple": True}, 
            "治则治法": {"num_options": 10, "multiple": True},
            "病性": {"num_options": 4, "multiple": False}
        }
        # 设置随机种子
        self.random_seed = random_seed
        random.seed(random_seed)
    
    # =============== 基础解析与随机化 ===============
    def _parse_options(self, options_text: str) -> List[Tuple[str, str]]:
        """
        解析选项文本，格式如 "A:选项1;B:选项2;..."
        返回：[(标签, 内容), ...]
        """
        options: List[Tuple[str, str]] = []
        for part in options_text.split(';'):
            part = part.strip()
            if ':' in part:
                label, content = part.split(':', 1)
                options.append((label.strip(), content.strip()))
        return options
    
    def _randomize_options(self, options: List[Tuple[str, str]], round_index: int = 1) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
        """
        随机化选项顺序，返回（随机后选项，新->旧标签映射）
        使用案例特定的随机种子确保可重现性
        
        Args:
            options: 原始选项列表
            round_index: 轮次索引（1, 2, 3...），用于生成不同的随机化
        """
        # 使用选项内容 + 轮次索引生成案例特定的种子
        import hashlib
        options_str = str(options) + f"_round_{round_index}"  # 添加轮次标识
        case_seed = int(hashlib.md5(options_str.encode()).hexdigest()[:8], 16) % (2**31)
        case_seed = (case_seed + self.random_seed) % (2**31)
        
        # 创建独立的随机状态
        rng = random.Random(case_seed)
        
        labels = [chr(ord('A') + i) for i in range(len(options))]
        contents = [content for _, content in options]
        rng.shuffle(contents)  # 使用独立的随机状态
        randomized_options = [(labels[i], contents[i]) for i in range(len(options))]
        answer_mapping: Dict[str, str] = {}
        for new_label, content in randomized_options:
            for orig_label, orig_content in options:
                if content == orig_content:
                    answer_mapping[new_label] = orig_label
                    break
        return randomized_options, answer_mapping

    # =============== 格式化方法 ===============
    def _format_parsed_answers(self, parsed_answers: Dict[str, List[str]]) -> Dict[str, str]:
        """
        格式化解析的答案，将列表转换为分号分隔的字符串
        
        Args:
            parsed_answers: 解析的答案字典
            
        Returns:
            格式化后的答案字典
        """
        formatted = {}
        for dim, answers in parsed_answers.items():
            formatted[dim] = ";".join(answers) if answers else ""
        return formatted
    
    def _format_options_mapping(self, options_by_dim: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Dict[str, str]]:
        """
        格式化选项映射，将选项列表转换为字符串形式的字典
        
        Args:
            options_by_dim: 各维度的选项列表
            
        Returns:
            格式化后的选项映射
        """
        formatted = {}
        for dim, options in options_by_dim.items():
            # 将选项列表转换为字典形式
            letter_to_content = self._letter_to_content_map(options)
            # 格式化为单行字符串显示
            formatted_content = "; ".join([f"{k}:{v}" for k, v in letter_to_content.items()])
            formatted[dim] = {
                "letter_to_content": formatted_content
            }
        return formatted

    # =============== 合并评测所需辅助方法 ===============
    def _letter_to_content_map(self, options: List[Tuple[str, str]]) -> Dict[str, str]:
        return {label: content for label, content in options}

    def _letters_to_contents(self, letters: List[str], letter2content: Dict[str, str]) -> List[str]:
        contents: List[str] = []
        for l in letters:
            if l in letter2content:
                contents.append(letter2content[l])
        # 去重并保持顺序
        seen = set()
        unique: List[str] = []
        for c in contents:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def _build_combined_prompt(self, instruction: str, options_by_dim: Dict[str, List[Tuple[str, str]]]) -> str:
        lines: List[str] = [
            "你是一位中医专家，请根据以下病例，同时完成证型、病性、病位、治则治法这4个维度的选择题判断。",
            "",
            "病例描述：",
            instruction,
            "",
        ]
        order = ["证型", "病性", "病位", "治则治法"]
        for dim in order:
            opts = options_by_dim[dim]
            opts_text = "\n".join([f"{label}. {content}" for label, content in opts])
            if self.choice_configs[dim]["multiple"]:
                # 对于证型/病位/治则治法，提示为不定项（可选一个或多个）
                if dim in ["证型", "病位", "治则治法"]:
                    tip = "（不定项，可选一个或多个）"
                else:
                    tip = "（可多选）"
            else:
                tip = "（单选）"
            lines.extend([f"{dim}选项{tip}：", opts_text, ""]) 
        lines.extend([
            "请只输出以下四行答案（不要输出其他内容）：",
            "证型答案：<字母>或<字母;字母;...>",
            "病性答案：<字母>",
            "病位答案：<字母>或<字母;字母;...>",
            "治则治法答案：<字母>或<字母;字母;...>",
        ])
        return "\n".join(lines)

    def _parse_combined_response(self, response: str) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        patterns = {
            "证型": r"证型答案[：:]\s*<?([A-J;；，,\s]+)>?",
            "病性": r"病性答案[：:]\s*<?([A-D;；，,\s]+)>?",
            "病位": r"病位答案[：:]\s*<?([A-J;；，,\s]+)>?",
            "治则治法": r"治则治法答案[：:]\s*<?([A-J;；，,\s]+)>?",
        }
        
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
            
        for dim, pat in patterns.items():
            m = re.search(pat, response_to_parse, re.IGNORECASE)
            letters = []
            if m:
                text = m.group(1)
                letters = re.findall(r'[A-J]', text.upper())
            else:
                lines = response_to_parse.strip().split('\n')
                for line in lines:
                    m = re.search(pat, line, re.IGNORECASE)
                    if m:
                        text = m.group(1)
                        letters = re.findall(r'[A-J]', text.upper())
                        break
            
            seen = set()
            uniq: List[str] = []
            for l in letters:
                if l not in seen:
                    seen.add(l)
                    uniq.append(l)
            result[dim] = uniq
            
        return result

    # =============== 合并评测主流程 ===============
    def evaluate_combined(self, case: Dict[str, Any], model_interface, pbar: tqdm) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, Any]]:
        dims = ["证型", "病性", "病位", "治则治法"]
        gt = case["output"]
        # 原始选项解析
        orig_options_map: Dict[str, List[Tuple[str, str]]] = {
            d: self._parse_options(gt[f"{d}选项"]) for d in dims
        }
        # 原始映射：字母->内容
        orig_letter2content = {d: self._letter_to_content_map(orig_options_map[d]) for d in dims}
        
        runs_options: List[Dict[str, List[Tuple[str, str]]]] = []
        # run0: 原序
        runs_options.append({d: orig_options_map[d] for d in dims})
        # run1, run2: 不同的随机化
        for round_idx in range(1, 3):  # round_idx = 1, 2
            rnd: Dict[str, List[Tuple[str, str]]] = {}
            for d in dims:
                randomized, _ = self._randomize_options(orig_options_map[d], round_idx)
                rnd[d] = randomized
            runs_options.append(rnd)
        
        # 执行三次
        all_runs_letters: List[Dict[str, List[str]]] = []
        first_run_answers: Dict[str, str] = {}
        
        # 详细结果记录
        detailed_results = {
            "runs": [],
            "final_scores": {}
        }
        
        for run_idx, opts_by_dim in enumerate(runs_options):
            prompt = self._build_combined_prompt(case["instruction"], opts_by_dim)
            pbar.write(f"正在评测选择题合并（第{run_idx+1}次）...")
            response = model_interface.generate(prompt, max_tokens=4096, temperature=0.0)
            parsed = self._parse_combined_response(response)
            all_runs_letters.append(parsed)
            
            # 记录本轮详细结果
            run_result = {
                "run_index": run_idx + 1,
                "run_type": "原序" if run_idx == 0 else f"随机{run_idx}",
                "parsed_answers": self._format_parsed_answers(parsed),  # 格式化答案显示
                "response": response,  # 添加完整响应以便调试
                "options_mapping": self._format_options_mapping(opts_by_dim)  # 格式化选项映射显示
            }
            
            detailed_results["runs"].append(run_result)
            
            if run_idx == 0:
                # 保存第一次（原序）答案字母串
                for d in dims:
                    first_run_answers[d] = ";".join(parsed.get(d, []))
        
        # 计算得分
        scores: Dict[str, float] = {d: 0.0 for d in dims}
        
        # 病性：三次全对才满分
        correct_bingxing_label = gt["病性答案"].strip()
        correct_bingxing_content = orig_letter2content["病性"].get(correct_bingxing_label)
        all_correct = True
        bingxing_round_results = []
        
        for run_idx, parsed in enumerate(all_runs_letters):
            letters = parsed.get("病性", [])
            if len(letters) != 1:
                all_correct = False
                bingxing_round_results.append({
                    "run": run_idx + 1,
                    "selected_letters": ";".join(letters),
                    "correct": False,
                    "reason": "选择数量不正确"
                })
                break
            if run_idx == 0:
                pred_content = orig_letter2content["病性"].get(letters[0])
            else:
                letter2content = self._letter_to_content_map(runs_options[run_idx]["病性"])
                pred_content = letter2content.get(letters[0])
            
            is_correct = pred_content == correct_bingxing_content
            bingxing_round_results.append({
                "run": run_idx + 1,
                "selected_letters": ";".join(letters),
                "selected_content": pred_content,
                "correct_content": correct_bingxing_content,
                "correct": is_correct
            })
            
            if not is_correct:
                all_correct = False
        
        scores["病性"] = 1.0 if all_correct else 0.0
        detailed_results["final_scores"]["病性"] = {
            "score": scores["病性"],
            "round_results": bingxing_round_results,
            "all_correct": all_correct
        }
        
        # 三个维度（证型/病位/治则治法）：采用评分公式
        # Sp = |A∩B| / (|A| + |Ā∩B|)
        # 其中 A 为标准答案集合，B 为模型选择集合；Ā 为 A 的补集
        # 三轮分别计算 Sp，最终得分为三轮得分的平均值
        for d in ["证型", "病位", "治则治法"]:
            round_results = []
            run_scores: List[float] = []

            # 正确集合（按内容）基于原始选项映射
            correct_labels = [x.strip() for x in gt[f"{d}答案"].split(';') if x.strip()]
            correct_contents = set(self._letters_to_contents(correct_labels, orig_letter2content[d]))
            gt_size = len(correct_contents)

            for run_idx, parsed in enumerate(all_runs_letters):
                letters = parsed.get(d, [])
                # 每一轮根据对应的选项映射将字母转内容
                if run_idx == 0:
                    letter2content = orig_letter2content[d]
                else:
                    letter2content = self._letter_to_content_map(runs_options[run_idx][d])
                chosen_contents = set(self._letters_to_contents(letters, letter2content))

                # 计算 TP / FP
                tp = len(chosen_contents & correct_contents)  # 选对的个数
                fp = len(chosen_contents - correct_contents)  # 错选的个数
                denom = gt_size + fp
                sp = (tp / denom) if denom > 0 else 0.0
                # 统一四位小数
                sp = float(f"{sp:.4f}")
                run_scores.append(sp)

                round_results.append({
                    "run": run_idx + 1,
                    "selected_letters": ";".join(letters),
                    "selected_contents": ";".join(list(chosen_contents)),
                    "tp": tp,
                    "fp": fp,
                    "gt_size": gt_size,
                    "run_score": sp
                })

            final_score = sum(run_scores) / len(run_scores) if run_scores else 0.0
            final_score = float(f"{final_score:.4f}")
            scores[d] = final_score

            detailed_results["final_scores"][d] = {
                "score": final_score,
                "round_results": round_results,
                "correct_contents": ";".join(list(correct_contents)),
                "run_scores": run_scores,
                "formula": "Sp = |A∩B| / (|A| + |Ā∩B|)"
            }
        
        # 添加证型的字母到内容映射信息，供后续使用
        detailed_results["syndrome_mapping"] = {
            "letter_to_content": orig_letter2content["证型"],
            "first_run_letters": all_runs_letters[0].get("证型", []) if all_runs_letters else []
        }

        # 添加治则治法的字母到内容映射信息（第一次原序的选择），供后续使用
        detailed_results["treatment_principles_mapping"] = {
            "letter_to_content": orig_letter2content["治则治法"],
            "first_run_letters": all_runs_letters[0].get("治则治法", []) if all_runs_letters else []
        }

        return scores, first_run_answers, detailed_results

    def evaluate_new_class(self, case: Dict[str, Any], model_interface, pbar: tqdm) -> Tuple[Dict[str, float], str, Dict[str, Any]]:
        question = case["question"]
        options = case["option"]
        correct_answer = case["answer"]
        question_type = case["question_type"]
        
        # 转换选项为列表格式
        option_list = [(k, v) for k, v in options.items()]
        
        # 准备三次运行的选项
        runs_options: List[List[Tuple[str, str]]] = []
        # run0: 原序
        runs_options.append(option_list)
        # run1, run2: 不同的随机化
        for round_idx in range(1, 3):
            randomized, _ = self._randomize_options(option_list, round_idx)
            runs_options.append(randomized)
        
        # 执行三次
        all_runs_letters: List[str] = []
        first_run_answer = ""
        
        # 详细结果记录
        detailed_results = {
            "runs": [],
            "final_score": 0.0
        }
        
        for run_idx, opts in enumerate(runs_options):
            # 构建提示
            prompt = self._build_new_class_prompt(question, opts, question_type)
            pbar.write(f"正在评测{case.get('class', '未知类别')}（第{run_idx+1}次）...")
            response = model_interface.generate(prompt, max_tokens=4096, temperature=0.0)
            parsed = self._parse_new_class_response(response)
            all_runs_letters.append(parsed)
            
            # 记录本轮详细结果
            letter2content = self._letter_to_content_map(opts)
            
            # 获取选择的内容
            if parsed:
                selected_letters = parsed.split(",")
                selected_contents = [letter2content.get(letter, "") for letter in selected_letters]
                selected_content_str = ";".join(selected_contents)
            else:
                selected_content_str = ""
            
            run_result = {
                "run_index": run_idx + 1,
                "run_type": "原序" if run_idx == 0 else f"随机{run_idx}",
                "parsed_answer": parsed,
                "selected_content": selected_content_str,
                "response": response
            }
            
            detailed_results["runs"].append(run_result)
            
            if run_idx == 0:
                first_run_answer = parsed
        
        # 计算得分与详细结果
        # 获取正确答案内容（基于原始选项顺序）
        orig_letter2content = self._letter_to_content_map(option_list)

        if question_type == "单项选择题":
            # 单选题评分逻辑：一致性 + 完全匹配
            score = 0.0
            round_results = []

            correct_content = orig_letter2content.get(correct_answer, "")
            correct_letters = [correct_answer]
            correct_contents = [correct_content]

            # 将每次选择的字母转换为内容进行比较
            all_runs_contents = []
            for run_idx, letter_str in enumerate(all_runs_letters):
                letter2content = self._letter_to_content_map(runs_options[run_idx])
                if letter_str:
                    letters = letter_str.split(",")
                    contents = [letter2content.get(letter, "") for letter in letters]
                    contents.sort()
                    content_set = set(contents)
                else:
                    content_set = set()
                all_runs_contents.append(content_set)

            is_consistent = all(content_set == all_runs_contents[0] for content_set in all_runs_contents)

            if not is_consistent:
                score = 0.0
                score_reason = "三次选择不一致"
            else:
                chosen_content_set = all_runs_contents[0]
                correct_content_set = set(correct_contents)
                if len(chosen_content_set) == 1 and chosen_content_set == correct_content_set:
                    score = 1.0
                    score_reason = "完全正确"
                else:
                    score = 0.0
                    score_reason = "选择错误"

            # 记录每轮结果
            for run_idx, letter_str in enumerate(all_runs_letters):
                letter2content = self._letter_to_content_map(runs_options[run_idx])
                if letter_str:
                    letters = letter_str.split(",")
                    contents = [letter2content.get(letter, "") for letter in letters]
                    selected_content = ";".join(contents)
                    content_set = set(contents)
                else:
                    selected_content = ""
                    content_set = set()

                correct_content_set = set(correct_contents)
                is_correct = content_set == correct_content_set if len(content_set) == 1 else False

                round_results.append({
                    "run": run_idx + 1,
                    "selected_letters": letter_str,
                    "selected_contents": selected_content,
                    "correct_letters": correct_answer,
                    "correct_contents": correct_content,
                    "correct": is_correct
                })

            detailed_results["final_score"] = score
            detailed_results["round_results"] = round_results
            detailed_results["consistency"] = is_consistent
            detailed_results["score_reason"] = score_reason
            detailed_results["correct_answer"] = correct_answer
            detailed_results["correct_content"] = correct_content

            return {"accuracy": score}, first_run_answer, detailed_results
        else:
            # 多选题：使用 Sp = |A∩B| / (|A| + |Ā∩B|)，三轮平均
            # 构建正确答案内容集合（基于原始选项映射）
            if ";" in correct_answer:
                correct_letters = [x for x in correct_answer.split(";") if x]
            else:
                correct_letters = list(correct_answer)
            correct_contents_set = set([orig_letter2content.get(letter, "") for letter in correct_letters if letter])
            correct_contents_set.discard("")
            gt_size = len(correct_contents_set)

            run_scores: List[float] = []
            round_results = []

            for run_idx, letter_str in enumerate(all_runs_letters):
                letter2content = self._letter_to_content_map(runs_options[run_idx])
                if letter_str:
                    letters = [x for x in letter_str.split(",") if x]
                    chosen_contents = set([letter2content.get(letter, "") for letter in letters])
                else:
                    chosen_contents = set()
                chosen_contents.discard("")

                tp = len(chosen_contents & correct_contents_set)
                fp = len(chosen_contents - correct_contents_set)
                denom = gt_size + fp
                sp = (tp / denom) if denom > 0 else 0.0
                sp = float(f"{sp:.4f}")
                run_scores.append(sp)

                round_results.append({
                    "run": run_idx + 1,
                    "selected_letters": letter_str or "",
                    "selected_contents": ";".join(sorted(list(chosen_contents))),
                    "tp": tp,
                    "fp": fp,
                    "gt_size": gt_size,
                    "run_score": sp
                })

            final_score = sum(run_scores) / len(run_scores) if run_scores else 0.0
            final_score = float(f"{final_score:.4f}")

            detailed_results["final_score"] = final_score
            detailed_results["round_results"] = round_results
            detailed_results["correct_contents"] = ";".join(sorted(list(correct_contents_set)))
            detailed_results["formula"] = "Sp = |A∩B| / (|A| + |Ā∩B|)"

            return {"accuracy": final_score}, first_run_answer, detailed_results

    def _build_new_class_prompt(self, question: str, options: List[Tuple[str, str]], question_type: str) -> str:
        lines: List[str] = [
            "请根据以下题目，选择正确答案。",
            "",
            "题目：",
            question,
            "",
            "选项："
        ]
        
        for label, content in options:
            lines.append(f"{label}. {content}")
        
        # 根据题目类型提供不同的指导
        if question_type == "单项选择题":
            answer_instruction = "请从以上选项中选择一个正确答案，只输出答案字母，不要输出其他内容。"
        else:  # 多选题
            answer_instruction = "请从以上选项中选择所有正确答案，多个答案用分号分隔，只输出答案字母，不要输出其他内容。"
        
        lines.extend([
            "",
            answer_instruction,
            "答案："
        ])
        
        return "\n".join(lines)

    def _parse_new_class_response(self, response: str) -> str:
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

        response_to_parse = response_to_parse.replace("；", ";").replace("，", ",").replace(" ", "")
        
        lines = response_to_parse.strip().split('\n')
        
        for line in lines:
            m = re.search(r"[答答案][：:]\s*([A-Z;,\s]+)", line, re.IGNORECASE)
            if m:
                content = m.group(1)
                letters = re.findall(r"[A-Z]", content)
                if letters:
                    return ",".join(letters)
        
        for line in lines:
            m = re.search(r"[选选择题][：:]\s*([A-Z;,\s]+)", line, re.IGNORECASE)
            if m:
                content = m.group(1)
                letters = re.findall(r"[A-Z]", content)
                if letters:
                    return ",".join(letters)
        
        m = re.search(r"[A-Z]{2,}", response_to_parse, re.IGNORECASE)
        if m:
            letters = list(m.group(0))
            return ",".join(letters)
        
        letters = re.findall(r"[A-Z]", response_to_parse, re.IGNORECASE)
        if letters:
            unique_letters = []
            seen = set()
            for letter in letters:
                upper_letter = letter.upper()
                if upper_letter not in seen:
                    seen.add(upper_letter)
                    unique_letters.append(upper_letter)
            return ",".join(unique_letters)

        return ""
