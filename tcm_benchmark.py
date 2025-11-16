#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中医辨证论治Benchmark评测框架

主要功能：
1. 对大语言模型进行中医辨证论治能力评测
2. 支持14个维度的综合评估
3. 包含选择题、奖励模型评分、LLM判分等多种评测方式

Usage:
    python tcm_benchmark.py --model_type api --api_url http://localhost:8000/v1 --model_name gpt-4
    python tcm_benchmark.py --model_type local --model_path /path/to/model
"""

import json
import random
import time
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 导入各个评测模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import TCMDataLoader
from evaluators import (
    MultipleChoiceEvaluator,
    RewardModelEvaluator,
    LLMJudgeEvaluator,
)
from model_interface import ModelInterface
from report_generator import ReportGenerator
from utils import setup_logging, save_checkpoint, load_checkpoint

# 配置日志
setup_logging()
logger = logging.getLogger(__name__)

# 显示名映射（仅用于输出/报告，不改变内部评测键）
DISPLAY_NAME_MAP = {
    "安全问题": "大模型内容安全",
}

@dataclass
class EvaluationConfig:
    """评测配置"""
    # 数据路径
    data_path: str = "benchmark.json"
    
    # 模型配置
    local_model_gpu_id: int = -1
    reward_api_host: str = "36.103.196.56"
    reward_api_port: int = 8001
    llm_judge_api_host: str = "36.103.196.56"
    llm_judge_api_port: int = 8002
    # 可配置的模型名称
    reward_model_name: str = "fangji"
    llm_judge_model_name: str = "qwen"
    # API Keys（用于OpenAI兼容服务）
    reward_api_key: str = ""
    llm_judge_api_key: str = ""
    
    # 评测配置
    random_seed: Optional[int] = None  # None表示自动生成随机种子
    max_retries: int = 3
    checkpoint_interval: int = 10

    stop_on_model_error: bool = True
    fatal_error_keywords: List[str] = field(default_factory=lambda: [
        "AccountOverdueError", "insufficient_quota", "insufficient quota", "quota exceeded",
        "Forbidden", "Error code: 403", "403", "rate limit", "too many requests",
        "unauthorized", "invalid_api_key", "notfound", "not found", "the model", "does not exist"
    ])
    
    def __post_init__(self):
        # 如果没有指定随机种子，则自动生成
        if self.random_seed is None:
            # 使用当前时间戳 + 进程ID + 随机数生成真正随机的种子
            import os
            timestamp = int(datetime.now().timestamp() * 1000000)  # 微秒级时间戳
            pid = os.getpid()  # 进程ID
            rand_component = random.randint(0, 999999)  # 额外随机分量
            self.random_seed = (timestamp + pid + rand_component) % (2**31 - 1)
            logger.info(f"自动生成随机种子: {self.random_seed}")

class TCMBenchmark:
    """中医辨证论治Benchmark主类"""
    
    def __init__(self, config: EvaluationConfig, skip_think: bool = False):
        """
        初始化评测器
        
        Args:
            config: 评测配置
            skip_think: 是否跳过Think内容完备性评测
        """
        self.config = config
        self.skip_think = skip_think
        
        # 设置随机种子（确保可重现性）
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # 初始化各个组件
        self.data_loader = TCMDataLoader(config.data_path)
        self.multiple_choice_evaluator = MultipleChoiceEvaluator(config.random_seed)
        # 不再使用BGE文本相似度评测器，改为由LLM进行病因/病机判分
        self.text_similarity_evaluator = None
        self.reward_model_evaluator = RewardModelEvaluator(
            config.reward_api_host,
            config.reward_api_port,
            config.reward_model_name,
            api_key=getattr(config, 'reward_api_key', None)
        )
        self.llm_judge_evaluator = LLMJudgeEvaluator(
            config.llm_judge_api_host,
            config.llm_judge_api_port,
            config.llm_judge_model_name,
            api_key=getattr(config, 'llm_judge_api_key', None)
        )
        self.report_generator = ReportGenerator()
        
        # 评测维度（固定顺序）
        base_dimensions = [
            "证型", "病性", "病位", "治则治法",  # 客观题评测
            "病因", "病机",  # LLM评分
            "方证契合度", "方剂配伍规律", "配伍禁忌", "药材安全性分析", "妊娠禁忌",  # 处方相关（方证契合度为奖励模型评分，其余归属 LLM 评分显示）
            "煎服方法", "注意事项", "随症加减"  # LLM评分
        ]
        
        # 根据skip_think参数决定是否包含CoT内容完备性和CoT准确性
        if not self.skip_think:
            # 在LLM判分维度的开头插入CoT内容完备性和CoT准确性
            self.evaluation_dimensions = base_dimensions[:11] + ["CoT内容完备性", "CoT准确性"] + base_dimensions[11:]
        else:
            self.evaluation_dimensions = base_dimensions
            
        logger.info(f"评测维度数量: {len(self.evaluation_dimensions)}")
        if self.skip_think:
            logger.info("已跳过CoT内容完备性和CoT准确性评测")
        
        # 存储详细结果
        self.detailed_results: List[Dict[str, Any]] = []
        # 存储通用评测任务（原新增类别）结果
        self.general_assessment_task_results: Dict[str, List[Dict[str, Any]]] = {
            "中医基础知识": [],
            "医学伦理": [],
            "安全问题": []
        }

    def run_evaluation(self, model_interface: ModelInterface, 
                      output_dir: str = "benchmark/results",
                      resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        运行完整评测
        
        Args:
            model_interface: 待评测模型接口
            output_dir: 输出目录
            resume_from_checkpoint: 是否从断点恢复
            
        Returns:
            评测结果字典
        """
        logger.info("开始中医辨证论治Benchmark评测")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # 加载数据
        cases = self.data_loader.load_cases()
        logger.info(f"加载了 {len(cases)} 个案例")
        
        # 按类别分类数据
        tcm_cases = [case for case in cases if case.get("class") == "中医辨证论治"]
        basic_cases = [case for case in cases if case.get("class") == "中医基础知识"]
        ethics_cases = [case for case in cases if case.get("class") == "医学伦理"]
        safety_cases = [case for case in cases if case.get("class") == "安全问题"]
        
        logger.info(f"中医辨证论治: {len(tcm_cases)} 个案例")
        logger.info(f"中医基础知识: {len(basic_cases)} 个案例")
        logger.info(f"医学伦理: {len(ethics_cases)} 个案例")
        logger.info(f"安全问题: {len(safety_cases)} 个案例")
        
        # 尝试从断点恢复
        start_idx = 0
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            checkpoint_data = load_checkpoint(checkpoint_path)
            if checkpoint_data:
                start_idx = checkpoint_data.get("completed_cases", 0)
                self.detailed_results = checkpoint_data.get("detailed_results", [])
                # 恢复通用评测任务结果（不兼容旧键）
                self.general_assessment_task_results = checkpoint_data.get("general_assessment_task_results", {
                    "中医基础知识": [],
                    "医学伦理": [],
                    "安全问题": []
                })
                # 恢复随机种子
                if "random_seed" in checkpoint_data:
                    self.config.random_seed = checkpoint_data["random_seed"]
                logger.info(f"从断点恢复，已完成 {start_idx} 个案例")
        
        # 使用随机种子进行案例顺序随机化
        random.seed(self.config.random_seed)
        random.shuffle(tcm_cases)
        random.shuffle(basic_cases)
        random.shuffle(ethics_cases)
        random.shuffle(safety_cases)
        logger.info(f"使用随机种子 {self.config.random_seed} 进行案例顺序随机化")
        
        # 评测中医辨证论治类别（保持原有逻辑）
        if len(tcm_cases) > 0:
            logger.info("开始评测中医辨证论治类别")
            self._evaluate_tcm_cases(tcm_cases, model_interface, output_dir, resume_from_checkpoint)
        
        # 评测通用评测任务（原新增类别）
        if len(basic_cases) > 0:
            logger.info("开始评测中医基础知识类别")
            self._evaluate_new_class_cases("中医基础知识", basic_cases, model_interface, output_dir)
            
        if len(ethics_cases) > 0:
            logger.info("开始评测医学伦理类别")
            self._evaluate_new_class_cases("医学伦理", ethics_cases, model_interface, output_dir)
            
        if len(safety_cases) > 0:
            logger.info("开始评测安全问题类别")
            self._evaluate_new_class_cases("安全问题", safety_cases, model_interface, output_dir)
        
        # 计算最终结果
        final_results = self._calculate_final_scores()
        
        # 生成报告
        report_path = os.path.join(output_dir, "evaluation_report.html")
        self.report_generator.generate_report(
            final_results, 
            self.detailed_results, 
            report_path,
            general_assessment_task_results=self.general_assessment_task_results
        )
        
        # 保存详细结果
        results_path = os.path.join(output_dir, "detailed_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            # 在写入JSON前统一将所有分数四舍五入至小数点后4位
            def r4(x: float) -> float:
                try:
                    return float(f"{float(x):.4f}")
                except Exception:
                    return x

            rounded_dimension_scores = {k: r4(v) for k, v in final_results.get("dimension_scores", {}).items()}
            rounded_general_tasks = {k: r4(v) for k, v in final_results.get("general_assessment_tasks", {}).items()}

            # 逐案例维度分数保留4位小数
            rounded_detailed_results = []
            for cr in self.detailed_results:
                cr_copy = cr.copy()
                ds = cr_copy.get("dimension_scores", {}) or {}
                cr_copy["dimension_scores"] = {k: r4(v) for k, v in ds.items()}
                rounded_detailed_results.append(cr_copy)

            # 通用评测任务的详细结果同样处理
            rounded_general_task_detailed_results = {}
            for class_name, items in self.general_assessment_task_results.items():
                new_list = []
                for cr in items:
                    cr_copy = cr.copy()
                    ds = cr_copy.get("dimension_scores", {}) or {}
                    cr_copy["dimension_scores"] = {k: r4(v) for k, v in ds.items()}
                    # 输出中替换显示的类别名
                    if "class" in cr_copy:
                        cr_copy["class"] = DISPLAY_NAME_MAP.get(cr_copy["class"], cr_copy["class"])
                    new_list.append(cr_copy)
                if new_list:  # 只包含有结果的类别
                    out_key = DISPLAY_NAME_MAP.get(class_name, class_name)
                    rounded_general_task_detailed_results[out_key] = new_list

            # 重构输出结构，避免数据冗余
            # 在写出配置前遮蔽敏感字段与内部字段
            sanitized_config = {
                "data_path": self.config.data_path,
                "local_model_gpu_id": self.config.local_model_gpu_id,
                "reward_api_host": self.config.reward_api_host,
                "reward_api_port": self.config.reward_api_port,
                "llm_judge_api_host": self.config.llm_judge_api_host,
                "llm_judge_api_port": self.config.llm_judge_api_port,
                "reward_model_name": self.config.reward_model_name,
                "llm_judge_model_name": self.config.llm_judge_model_name,
                "max_retries": self.config.max_retries,
                "checkpoint_interval": self.config.checkpoint_interval,
                # 不输出：API Keys、stop_on_model_error、fatal_error_keywords 等内部控制项
            }

            # 计算 SDT_task.score（等价于对 dimension_scores 的均值）
            try:
                sdt_mean = float(f"{np.mean(list(rounded_dimension_scores.values())) if rounded_dimension_scores else 0.0:.4f}")
            except Exception:
                sdt_mean = 0.0

            output_data = {
                "final_scores": {
                    "total_score": r4(final_results.get("total_score", 0.0)),
                    # SDT_task 区块紧随 total_score，包含维度得分与均值
                    "SDT_task": {
                        "dimension_scores": rounded_dimension_scores,
                        "score": sdt_mean
                    },
                    # 实际参与计分的类别
                    "participating_classes": [DISPLAY_NAME_MAP.get(x, x) for x in final_results.get("participating_classes", [])],
                    # 将 num_cases 放在 participating_classes 下方
                    "num_cases": final_results.get("num_cases", 0),
                    # 通用评测任务（原 new_class_scores）
                    "general_assessment_tasks": {DISPLAY_NAME_MAP.get(k, k): v for k, v in rounded_general_tasks.items()}
                },
                "config": sanitized_config,
                "random_seed": self.config.random_seed,
                "detailed_results": rounded_detailed_results,
                # 通用评测任务的详细结果
                "general_assessment_task_detailed_results": rounded_general_task_detailed_results
            }
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评测完成！总分：{final_results['total_score']:.4f}")
        logger.info(f"报告已保存至：{report_path}")
        
        return final_results

    def _is_fatal_error(self, err: Any) -> bool:
        """检查异常或错误消息是否属于致命错误（需要立即停止评测）。"""
        try:
            msg = str(err or "")
            msg_lower = msg.lower()
            for kw in self.config.fatal_error_keywords:
                if kw and kw.lower() in msg_lower:
                    return True
        except Exception:
            return False
        return False

    def _evaluate_tcm_cases(self, cases: List[Dict], model_interface: ModelInterface, 
                           output_dir: str, resume_from_checkpoint: bool):
        """
        评测中医辨证论治类别（保持原有逻辑）
        """
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # 获取已评测的案例数
        completed_cases = len(self.detailed_results)
        
        # 创建进度条，初始位置设置为已评测的案例数
        pbar = tqdm(
            total=len(cases), 
            desc="评测进度 - 中医辨证论治",
            position=0,
            leave=True,
            initial=completed_cases  # 设置初始已完成的数量
        )
        
        try:
            # 逐案例评测
            for case_idx in range(completed_cases, len(cases)):
                case = cases[case_idx]
                
                pbar.set_description(f"评测案例 {case_idx + 1}/{len(cases)} - 中医辨证论治")
                
                # 评测单个案例
                case_result = self._evaluate_single_case(case, model_interface, pbar)
                self.detailed_results.append(case_result)
                
                # 更新进度条
                pbar.update(1)
                
                # 定期保存断点
                if (case_idx + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_data = {
                        "completed_cases": case_idx + 1,
                        "detailed_results": self.detailed_results,
                        "general_assessment_task_results": self.general_assessment_task_results,
                        "random_seed": self.config.random_seed,
                    }
                    save_checkpoint(checkpoint_data, checkpoint_path)
                    pbar.write(f"已保存断点：完成 {case_idx + 1} 个中医辨证论治案例")
        
        except KeyboardInterrupt:
            logger.info("评测被用户中断")
            # 保存当前进度
            checkpoint_data = {
                "completed_cases": len(self.detailed_results),
                "detailed_results": self.detailed_results,
                "general_assessment_task_results": self.general_assessment_task_results,
                "random_seed": self.config.random_seed,
            }
            save_checkpoint(checkpoint_data, checkpoint_path)
            logger.info(f"已保存断点：完成 {len(self.detailed_results)} 个中医辨证论治案例")
            raise

        except Exception as e:
            # 发生致命错误或其他未处理异常时，保存断点并停止
            logger.error(f"评测过程中发生错误，正在保存断点并停止：{e}")
            checkpoint_data = {
                "completed_cases": len(self.detailed_results),
                "detailed_results": self.detailed_results,
                "general_assessment_task_results": self.general_assessment_task_results,
                "random_seed": self.config.random_seed,
            }
            save_checkpoint(checkpoint_data, checkpoint_path)
            raise       
        finally:
            pbar.close()

    def _evaluate_new_class_cases(self, class_name: str, cases: List[Dict], 
                                 model_interface: ModelInterface, output_dir: str):
        """
        评测新增类别（中医基础知识、医学伦理、安全问题）
        """
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # 获取该类别已评测的案例数
        completed_cases = len(self.general_assessment_task_results[class_name])
        
        # 创建进度条，初始位置设置为已评测的案例数
        pbar = tqdm(
            total=len(cases), 
            desc=f"评测进度 - {class_name}",
            position=0,
            leave=True,
            initial=completed_cases
        )
        
        try:
            # 逐案例评测
            for case_idx in range(completed_cases, len(cases)):
                case = cases[case_idx]
                
                pbar.set_description(f"评测案例 {case_idx + 1}/{len(cases)} - {class_name}")
                
                # 评测单个案例
                case_result = self._evaluate_new_class_single_case(case, model_interface, pbar)
                self.general_assessment_task_results[class_name].append(case_result)
                
                # 更新进度条
                pbar.update(1)
                
                # 定期保存断点
                if (case_idx + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_data = {
                        "completed_cases": len(self.detailed_results),  # 保持与原有逻辑一致
                        "detailed_results": self.detailed_results,
                        "general_assessment_task_results": self.general_assessment_task_results,
                        "random_seed": self.config.random_seed,
                    }
                    save_checkpoint(checkpoint_data, checkpoint_path)
                    pbar.write(f"已保存断点：完成 {case_idx + 1} 个{class_name}案例")
        
        except KeyboardInterrupt:
            logger.info(f"{class_name}评测被用户中断")
            # 保存当前进度
            checkpoint_data = {
                "completed_cases": len(self.detailed_results),
                "detailed_results": self.detailed_results,
                "general_assessment_task_results": self.general_assessment_task_results,
                "random_seed": self.config.random_seed,
            }
            save_checkpoint(checkpoint_data, checkpoint_path)
            logger.info(f"已保存断点：完成 {len(self.general_assessment_task_results[class_name])} 个{class_name}案例")
            raise

        except Exception as e:
            logger.error(f"{class_name}评测过程中发生错误，正在保存断点并停止：{e}")
            checkpoint_data = {
                "completed_cases": len(self.detailed_results),
                "detailed_results": self.detailed_results,
                "general_assessment_task_results": self.general_assessment_task_results,
                "random_seed": self.config.random_seed,
            }
            save_checkpoint(checkpoint_data, checkpoint_path)
            raise      
        finally:
            pbar.close()

    def _evaluate_new_class_single_case(self, case: Dict[str, Any], 
                                       model_interface: ModelInterface,
                                       pbar: tqdm) -> Dict[str, Any]:
        """
        评测新增类别单个案例（中医基础知识、医学伦理、安全问题）
        """
        case_id = case.get("id") or "unknown_case"
        
        case_result = {
            "case_id": case_id,
            "question": case["question"],
            "ground_truth": case["answer"],
            "question_type": case["question_type"],
            "options": case["option"],
            "model_responses": {},
            "dimension_scores": {},
            "detailed_evaluation_results": {},
            "class": case.get("class", "未知类别")  # 添加类别信息
        }
        
        # 使用选择题评测器进行三轮评测
        try:
            scores, first_answers, detailed_results = self.multiple_choice_evaluator.evaluate_new_class(case, model_interface, pbar)
            case_result["dimension_scores"]["accuracy"] = scores.get("accuracy", 0.0)
            case_result["model_responses"]["answer"] = first_answers
            case_result["detailed_evaluation_results"] = detailed_results
            
        except Exception as e:
            logger.error(f"新增类别选择题评测失败: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            case_result["dimension_scores"]["accuracy"] = 0.0
            case_result["model_responses"]["answer"] = f"评测失败: {str(e)}"
            case_result["detailed_evaluation_results"] = {"error": str(e)}
        
        return case_result

    def _evaluate_single_case(self, case: Dict[str, Any], 
                             model_interface: ModelInterface,
                             pbar: tqdm) -> Dict[str, Any]:
        """
        评测单个案例（固定顺序 + 合并选择题评测）
        """
        # 统一case_id获取逻辑，确保与其他评测器一致
        case_id = case.get("case_id") or case.get("id") or "unknown_case"

        case_result = {
            "case_id": case_id,
            "instruction": case["instruction"],
            "ground_truth": case["output"],
            "diagnosis": case["中医疾病诊断"],
            "dimension_scores": {},
            "model_responses": {},
            "detailed_evaluation_results": {}  # 新增：详细评测结果
        }
        
        # 1) 选择题合并评测（证型-病性-病位-治则治法）
        try:
            scores_mc, first_answers, detailed_mc_results = self.multiple_choice_evaluator.evaluate_combined(case, model_interface, pbar)
            for dim in ["证型", "病性", "病位", "治则治法"]:
                case_result["dimension_scores"][dim] = scores_mc.get(dim, 0.0)
                # 为后续依赖（证型 -> 奖励模型），仅记录第一次（原序）答案字母串
                case_result["model_responses"][dim] = f"{dim}答案：" + first_answers.get(dim, "")
            
            # 保存选择题的三轮评测详细结果
            case_result["detailed_evaluation_results"]["multiple_choice"] = detailed_mc_results
            
        except Exception as e:
            logger.error(f"选择题合并评测失败: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            for dim in ["证型", "病性", "病位", "治则治法"]:
                case_result["dimension_scores"][dim] = 0.0
                case_result["model_responses"][dim] = f"评测失败: {str(e)}"
            case_result["detailed_evaluation_results"]["multiple_choice"] = {"error": str(e)}
        
        # 2) 病因/病机改为LLM判分
        try:
            scores_cm, responses_cm = self.llm_judge_evaluator.evaluate_cause_mechanism(case, model_interface, pbar)
            for dim in ["病因", "病机"]:
                case_result["dimension_scores"][dim] = scores_cm.get(dim, 0.0)
                case_result["model_responses"][dim] = responses_cm.get(dim, "")
        except Exception as e:
            logger.error(f"病因/病机 LLM评测失败: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            for dim in ["病因", "病机"]:
                case_result["dimension_scores"][dim] = 0.0
                case_result["model_responses"][dim] = f"评测失败: {str(e)}"
        
        # 3) 处方相关维度（方证契合度-方剂配伍规律-配伍禁忌-药材安全性分析-妊娠禁忌）
        try:
            scores_sp, responses_sp = self.reward_model_evaluator.evaluate_all(
                case, model_interface, first_answers.get("证型", ""), pbar, llm_judge_evaluator=self.llm_judge_evaluator
            )
            for dim in ["方证契合度", "方剂配伍规律", "配伍禁忌", "药材安全性分析", "妊娠禁忌"]:
                case_result["dimension_scores"][dim] = scores_sp.get(dim, 0.0)
                if dim == "方证契合度":
                    # 将模型输出改写为“药物组成及用量”键
                    case_result["model_responses"]["药物组成及用量"] = responses_sp.get(dim, "")
                else:
                    case_result["model_responses"][dim] = responses_sp.get(dim, "")
        except Exception as e:
            logger.error(f"奖励模型评测失败: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            for dim in ["方证契合度", "方剂配伍规律", "配伍禁忌", "药材安全性分析", "妊娠禁忌"]:
                case_result["dimension_scores"][dim] = 0.0
                if dim == "方证契合度":
                    case_result["model_responses"]["药物组成及用量"] = f"评测失败: {str(e)}"
                else:
                    case_result["model_responses"][dim] = f"评测失败: {str(e)}"
 
        # 4) LLM判分（CoT内容完备性-煎服方法-注意事项-随症加减）
        try:
            # 使用数据集中的标准答案作为生成时提供的关键信息，避免被测模型结果污染
            gt_prescription_herbs = case["output"].get("药物组成及用量", "")
            gt_syndrome = case["output"].get("证型", "")
            gt_treatment_principles = case["output"].get("治则治法", "")

            scores_llm, responses_llm = self.llm_judge_evaluator.evaluate_all(
                case, model_interface, pbar,
                syndrome_choice=gt_syndrome,                      # 注意事项/随症加减：提供标准证型
                prescription_herbs=gt_prescription_herbs,         # 三项生成使用标准处方组成
                treatment_principles=gt_treatment_principles,     # 煎服方法：提供标准治则治法
                skip_think=self.skip_think                        # 传递skip_think参数
            )
            
            # 根据skip_think参数决定要处理的维度
            llm_dimensions = ["煎服方法", "注意事项", "随症加减"]
            if not self.skip_think:
                llm_dimensions = ["CoT内容完备性"] + llm_dimensions
            
            for dim in llm_dimensions:
                case_result["dimension_scores"][dim] = scores_llm.get(dim, 0.0)
                case_result["model_responses"][dim] = responses_llm.get(dim, "")
            
            # 🆕 新增：CoT准确性（幻觉）评测
            if not self.skip_think and "CoT内容完备性" in responses_llm:
                think_content = responses_llm["CoT内容完备性"]
                
                # 调用幻觉评测
                cot_accuracy_score, hallucination_details = \
                    self.llm_judge_evaluator.evaluate_hallucination(case, think_content, pbar)
                
                # 保存分数到 dimension_scores
                case_result["dimension_scores"]["CoT准确性"] = cot_accuracy_score
                
                # 保存完整信息到 hallucination_details
                case_result["hallucination_details"] = hallucination_details
                
        except Exception as e:
            logger.error(f"LLM合并评测失败: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            # 根据skip_think参数决定要处理的维度
            llm_dimensions = ["煎服方法", "注意事项", "随症加减"]
            if not self.skip_think:
                llm_dimensions = ["CoT内容完备性", "CoT准确性"] + llm_dimensions
            
            for dim in llm_dimensions:
                case_result["dimension_scores"][dim] = 0.0
                # CoT准确性不保存在model_responses中
                if dim != "CoT准确性":
                    case_result["model_responses"][dim] = f"评测失败: {str(e)}"
        
        return case_result

    def _get_prescription_herbs_from_cache(self, case: Dict[str, Any]) -> str:
        """
        从奖励模型评测器的处方缓存中获取处方组成信息

        Args:
            case: 案例数据

        Returns:
            处方组成字符串
        """
        try:
            case_id = case.get("case_id") or case.get("id") or "unknown_case"
            if case_id in self.reward_model_evaluator.prescription_cache:
                prescription_response = self.reward_model_evaluator.prescription_cache[case_id].get("raw_response", "")
                # 解析处方信息
                prescription_info = self.reward_model_evaluator._parse_prescription_response(prescription_response)
                herbs_list = [f"{h[0]} {h[1]} {h[2]}".strip() for h in prescription_info.get("herbs", [])]
                return "\n".join(['- '+x for x in herbs_list]) if herbs_list else ""
            else:
                logger.warning(f"未找到案例 {case_id} 的处方缓存")
                return ""
        except Exception as e:
            logger.error(f"获取处方组成失败: {e}")
            return ""

    def _get_syndrome_contents_from_detailed_results(self, detailed_mc_results: Dict[str, Any]) -> str:
        """
        从选择题详细结果中获取证型的具体内容

        Args:
            detailed_mc_results: 选择题详细评测结果

        Returns:
            证型内容字符串，多个证型用分号分隔
        """
        try:
            syndrome_mapping = detailed_mc_results.get("syndrome_mapping", {})
            letter_to_content = syndrome_mapping.get("letter_to_content", {})
            first_run_letters = syndrome_mapping.get("first_run_letters", [])

            # 将字母转换为具体内容
            syndrome_contents = []
            for letter in first_run_letters:
                content = letter_to_content.get(letter, "")
                if content:
                    syndrome_contents.append(content)

            result = ";".join(syndrome_contents)
            logger.info(f"证型内容: {result}")
            return result

        except Exception as e:
            logger.error(f"获取证型内容失败: {e}")
            return ""

    def _get_treatment_principles_from_detailed_results(self, detailed_mc_results: Dict[str, Any]) -> str:
        """
        从选择题详细结果中获取治则治法的具体内容（第一次原序选择）

        Args:
            detailed_mc_results: 选择题详细评测结果

        Returns:
            治则治法内容字符串，多个用分号分隔
        """
        try:
            tp_mapping = detailed_mc_results.get("treatment_principles_mapping", {})
            letter_to_content = tp_mapping.get("letter_to_content", {})
            first_run_letters = tp_mapping.get("first_run_letters", [])

            contents = []
            for letter in first_run_letters:
                content = letter_to_content.get(letter, "")
                if content:
                    contents.append(content)

            result = ";".join(contents)
            logger.info(f"治则治法内容: {result}")
            return result
        except Exception as e:
            logger.error(f"获取治则治法内容失败: {e}")
            return ""

    def _calculate_final_scores(self) -> Dict[str, Any]:
        """
        计算最终得分
        
        Returns:
            最终得分字典
        """
        if not self.detailed_results and not any(self.general_assessment_task_results.values()):
            return {"total_score": 0.0, "dimension_scores": {}}
        
        result = {
            "total_score": 0.0,
            "dimension_scores": {},
            "general_assessment_tasks": {},
            "num_cases": len(self.detailed_results)  # 原有类别案例数
        }
        
        # 计算各类别的案例数
        total_cases = len(self.detailed_results)
        for class_results in self.general_assessment_task_results.values():
            total_cases += len(class_results)
        
        result["num_cases"] = total_cases
        
        # 计算中医辨证论治类别得分（保持现有逻辑但权重相同）
        if self.detailed_results:
            # 计算各维度平均分
            dimension_avg_scores = {}
            for dimension in self.evaluation_dimensions:
                scores = [
                    result["dimension_scores"].get(dimension, 0.0) 
                    for result in self.detailed_results
                ]
                dimension_avg_scores[dimension] = np.mean(scores) if scores else 0.0
            
            # 计算中医辨证论治类别总分（实际参与维度平均分的平均值，根据skip_think参数可能是14或15个维度）
            actual_dimension_count = len(dimension_avg_scores)
            tcm_score = np.mean(list(dimension_avg_scores.values())) if dimension_avg_scores else 0.0
            result["dimension_scores"] = dimension_avg_scores
            result["tcm_score"] = tcm_score
            
            # 记录实际参与计算的维度数量
            logger.info(f"中医辨证论治类别总分计算: 使用 {actual_dimension_count} 个维度, 平均分: {tcm_score:.4f}")
            if self.skip_think:
                logger.info("已跳过Think内容完备性维度，使用14维度计算")
        else:
            tcm_score = 0.0
            result["tcm_score"] = tcm_score
        
        # 计算新增类别的得分（只对有样本的类别计入最终平均）
        general_task_scores = {}
        participating_classes = []
        participating_scores = []

        # 主类参与计分
        if self.detailed_results:
            participating_classes.append("中医辨证论治")
            participating_scores.append(tcm_score)

        # 新增类别：记录每类平均分，但仅将“有样本”的类别计入参与平均
        for class_name, results in self.general_assessment_task_results.items():
            if results:
                scores = [result["dimension_scores"].get("accuracy", 0.0) for result in results]
                # 准确率已在[0,1]范围
                avg_score = np.mean(scores) if scores else 0.0
                out_name = DISPLAY_NAME_MAP.get(class_name, class_name)
                general_task_scores[out_name] = avg_score
                participating_classes.append(out_name)
                participating_scores.append(avg_score)

        result["general_assessment_tasks"] = general_task_scores
        result["participating_classes"] = participating_classes

        # 仅对“实际参与评测的类别”取平均
        total_score = np.mean(participating_scores) if participating_scores else 0.0
        result["total_score"] = total_score
        
        return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BEST4SDT评测")
    parser.add_argument("--model_type", choices=["api", "local"], required=True, help="模型类型")
    parser.add_argument("--api_url", help="API地址（API模式）")
    parser.add_argument("--model_name", help="模型名称（API模式）")
    parser.add_argument("--api_key", help="API Key（API模式）")
    parser.add_argument("--model_path", help="本地模型路径（本地模式）")
    parser.add_argument("--config_file", required=True, help="配置文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--resume", action="store_true", help="从断点恢复")
    parser.add_argument("--skip_think", action="store_true", help="跳过CoT内容完备性评测（适用于不支持CoT的模型）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = EvaluationConfig()
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # 创建模型接口
    if args.model_type == "api":
        if not args.api_url or not args.model_name or not args.api_key:
            raise ValueError("API模式需要提供api_url、model_name和api_key")
        from model_interface import APIModelInterface
        model_interface = APIModelInterface(args.api_url, args.model_name, args.api_key)
    else:  # local
        if not args.model_path:
            raise ValueError("本地模式需要提供model_path")
        from model_interface import LocalModelInterface
        model_interface = LocalModelInterface(args.model_path, config.local_model_gpu_id)
    
    # 运行评测
    benchmark = TCMBenchmark(config, skip_think=args.skip_think)
    results = benchmark.run_evaluation(model_interface, args.output_dir, args.resume)
    
    print(f"评测完成！总分：{results['total_score']:.4f}")

if __name__ == "__main__":
    main()