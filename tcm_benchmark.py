#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Import evaluation modules
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

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Display-name mapping (output/report only; does not change internal evaluation keys)
DISPLAY_NAME_MAP = {
    "安全问题": "大模型内容安全",
}

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Data path
    data_path: str = "TCM-BEST4SDT.json"
    
    # Model configuration
    local_model_gpu_id: int = -1
    reward_api_host: str = "127.0.0.1"
    reward_api_port: int = 8001
    llm_judge_api_host: str = "127.0.0.1"
    llm_judge_api_port: int = 8002
    # Configurable model names
    reward_model_name: str = "Fangzheng-RM"
    llm_judge_model_name: str = "Qwen3-32B"
    # API keys (for OpenAI-compatible services)
    reward_api_key: str = ""
    llm_judge_api_key: str = ""
    
    # Evaluation configuration
    random_seed: Optional[int] = None  # None means auto-generate
    max_retries: int = 3
    checkpoint_interval: int = 10

    stop_on_model_error: bool = True
    fatal_error_keywords: List[str] = field(default_factory=lambda: [
        "AccountOverdueError", "insufficient_quota", "insufficient quota", "quota exceeded",
        "Forbidden", "Error code: 403", "403", "rate limit", "too many requests",
        "unauthorized", "invalid_api_key", "notfound", "not found", "the model", "does not exist"
    ])
    
    def __post_init__(self):
        # Auto-generate a random seed if not provided
        if self.random_seed is None:
            # Use timestamp + PID + random component to generate a seed
            import os
            timestamp = int(datetime.now().timestamp() * 1000000)  # microsecond timestamp
            pid = os.getpid()  # process ID
            rand_component = random.randint(0, 999999)  # extra randomness
            self.random_seed = (timestamp + pid + rand_component) % (2**31 - 1)
            logger.info(f"自动生成随机种子: {self.random_seed}")

class TCMBenchmark:   
    def __init__(self, config: EvaluationConfig, skip_think: bool = False):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Evaluation configuration
            skip_think: Whether to skip judging CoT completeness/accuracy
        """
        self.config = config
        self.skip_think = skip_think
        
        # Set random seeds (reproducibility)
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize components
        self.data_loader = TCMDataLoader(config.data_path)
        self.multiple_choice_evaluator = MultipleChoiceEvaluator(config.random_seed)
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
        
        # Evaluation dimensions (fixed order)
        base_dimensions = [
            "证型", "病性", "病位", "治则治法",  
            "病因", "病机",  
            "方证契合度", "方剂配伍规律", "配伍禁忌", "药材安全性分析", "妊娠禁忌",  
            "煎服方法", "注意事项", "随症加减"  
        ]
        
        # Include/exclude CoT completeness & CoT accuracy based on skip_think
        if not self.skip_think:
            # Insert CoT completeness & CoT accuracy at the start of the LLM-judged section
            self.evaluation_dimensions = base_dimensions[:11] + ["CoT内容完备性", "CoT准确性"] + base_dimensions[11:]
        else:
            self.evaluation_dimensions = base_dimensions
            
        logger.info(f"评测维度数量: {len(self.evaluation_dimensions)}")
        if self.skip_think:
            logger.info("已跳过CoT内容完备性和CoT准确性评测")
        
        # Store detailed results
        self.detailed_results: List[Dict[str, Any]] = []
        # Store general-assessment task results
        self.general_assessment_task_results: Dict[str, List[Dict[str, Any]]] = {
            "中医基础知识": [],
            "医学伦理": [],
            "安全问题": []
        }

    def run_evaluation(self, model_interface: ModelInterface, 
                      output_dir: str = "benchmark/results",
                      resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Run the full evaluation.
        
        Args:
            model_interface: Evaluated model interface
            output_dir: Output directory
            resume_from_checkpoint: Whether to resume from checkpoint
            
        Returns:
            Result dictionary
        """
        logger.info("开始中医辨证论治Benchmark评测")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # Load data
        cases = self.data_loader.load_cases()
        logger.info(f"加载了 {len(cases)} 个案例")
        
        # Split cases by class
        tcm_cases = [case for case in cases if case.get("class") == "中医辨证论治"]
        basic_cases = [case for case in cases if case.get("class") == "中医基础知识"]
        ethics_cases = [case for case in cases if case.get("class") == "医学伦理"]
        safety_cases = [case for case in cases if case.get("class") == "安全问题"]
        
        logger.info(f"中医辨证论治: {len(tcm_cases)} 个案例")
        logger.info(f"中医基础知识: {len(basic_cases)} 个案例")
        logger.info(f"医学伦理: {len(ethics_cases)} 个案例")
        logger.info(f"安全问题: {len(safety_cases)} 个案例")
        
        # Try to resume from checkpoint
        start_idx = 0
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            checkpoint_data = load_checkpoint(checkpoint_path)
            if checkpoint_data:
                start_idx = checkpoint_data.get("completed_cases", 0)
                self.detailed_results = checkpoint_data.get("detailed_results", [])
                # Restore general-assessment task results
                self.general_assessment_task_results = checkpoint_data.get("general_assessment_task_results", {
                    "中医基础知识": [],
                    "医学伦理": [],
                    "安全问题": []
                })
                # Restore random seed
                if "random_seed" in checkpoint_data:
                    self.config.random_seed = checkpoint_data["random_seed"]
                logger.info(f"从断点恢复，已完成 {start_idx} 个案例")
        
        # Shuffle case order using the configured random seed
        random.seed(self.config.random_seed)
        random.shuffle(tcm_cases)
        random.shuffle(basic_cases)
        random.shuffle(ethics_cases)
        random.shuffle(safety_cases)
        logger.info(f"使用随机种子 {self.config.random_seed} 进行案例顺序随机化")
        
        # Evaluate the SDT (syndrome differentiation and treatment) category
        if len(tcm_cases) > 0:
            logger.info("开始评测中医辨证论治类别")
            self._evaluate_tcm_cases(tcm_cases, model_interface, output_dir, resume_from_checkpoint)
        
        # Evaluate general-assessment tasks
        if len(basic_cases) > 0:
            logger.info("开始评测中医基础知识类别")
            self._evaluate_new_class_cases("中医基础知识", basic_cases, model_interface, output_dir)
            
        if len(ethics_cases) > 0:
            logger.info("开始评测医学伦理类别")
            self._evaluate_new_class_cases("医学伦理", ethics_cases, model_interface, output_dir)
            
        if len(safety_cases) > 0:
            logger.info("开始评测安全问题类别")
            self._evaluate_new_class_cases("安全问题", safety_cases, model_interface, output_dir)
        
        # Compute final results
        final_results = self._calculate_final_scores()
        
        # Generate report
        report_path = os.path.join(output_dir, "evaluation_report.html")
        self.report_generator.generate_report(
            final_results, 
            self.detailed_results, 
            report_path,
            general_assessment_task_results=self.general_assessment_task_results
        )
        
        # Save detailed results
        results_path = os.path.join(output_dir, "detailed_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            # Round all scores to 4 decimals before writing JSON
            def r4(x: float) -> float:
                try:
                    return float(f"{float(x):.4f}")
                except Exception:
                    return x

            rounded_dimension_scores = {k: r4(v) for k, v in final_results.get("dimension_scores", {}).items()}
            rounded_general_tasks = {k: r4(v) for k, v in final_results.get("general_assessment_tasks", {}).items()}

            # Per-case dimension scores: keep 4 decimals
            rounded_detailed_results = []
            for cr in self.detailed_results:
                cr_copy = cr.copy()
                ds = cr_copy.get("dimension_scores", {}) or {}
                cr_copy["dimension_scores"] = {k: r4(v) for k, v in ds.items()}
                rounded_detailed_results.append(cr_copy)

            # Do the same for general-assessment task detailed results
            rounded_general_task_detailed_results = {}
            for class_name, items in self.general_assessment_task_results.items():
                new_list = []
                for cr in items:
                    cr_copy = cr.copy()
                    ds = cr_copy.get("dimension_scores", {}) or {}
                    cr_copy["dimension_scores"] = {k: r4(v) for k, v in ds.items()}
                    # Replace class name for display in output
                    if "class" in cr_copy:
                        cr_copy["class"] = DISPLAY_NAME_MAP.get(cr_copy["class"], cr_copy["class"])
                    new_list.append(cr_copy)
                if new_list:  # Only include classes with results
                    out_key = DISPLAY_NAME_MAP.get(class_name, class_name)
                    rounded_general_task_detailed_results[out_key] = new_list

            # Restructure output to reduce redundancy
            # Mask sensitive/internal fields in config before writing
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
            }

            # Compute SDT_task.score (equivalent to the mean of dimension_scores)
            try:
                sdt_mean = float(f"{np.mean(list(rounded_dimension_scores.values())) if rounded_dimension_scores else 0.0:.4f}")
            except Exception:
                sdt_mean = 0.0

            output_data = {
                "final_scores": {
                    "total_score": r4(final_results.get("total_score", 0.0)),
                    "SDT_task": {
                        "dimension_scores": rounded_dimension_scores,
                        "score": sdt_mean
                    },
                        # Classes that actually participate in scoring
                    "participating_classes": [DISPLAY_NAME_MAP.get(x, x) for x in final_results.get("participating_classes", [])],
                    "num_cases": final_results.get("num_cases", 0),
                        # General-assessment tasks
                    "general_assessment_tasks": {DISPLAY_NAME_MAP.get(k, k): v for k, v in rounded_general_tasks.items()}
                },
                "config": sanitized_config,
                "random_seed": self.config.random_seed,
                "detailed_results": rounded_detailed_results,
                # Detailed results for general-assessment tasks
                "general_assessment_task_detailed_results": rounded_general_task_detailed_results
            }
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评测完成！总分：{final_results['total_score']:.4f}")
        logger.info(f"报告已保存至：{report_path}")
        
        return final_results

    def _is_fatal_error(self, err: Any) -> bool:
        """Return True if the exception/message indicates a fatal error (stop evaluation immediately)."""
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
        Evaluate SDT (中医辨证论治) cases.
        """
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # Number of completed cases
        completed_cases = len(self.detailed_results)
        
        # Create a progress bar; set the initial position to the number of already evaluated cases.
        pbar = tqdm(
            total=len(cases), 
            desc="评测进度 - 中医辨证论治",
            position=0,
            leave=True,
            initial=completed_cases  # Set the initial completed count.
        )
        
        try:
            # Evaluate case by case.
            for case_idx in range(completed_cases, len(cases)):
                case = cases[case_idx]
                
                pbar.set_description(f"评测案例 {case_idx + 1}/{len(cases)} - 中医辨证论治")
                
                # Evaluate a single case.
                case_result = self._evaluate_single_case(case, model_interface, pbar)
                self.detailed_results.append(case_result)
                
                # Update the progress bar.
                pbar.update(1)
                
                # Save checkpoints periodically.
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
            # Save current progress.
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
            # On fatal errors or other unhandled exceptions, save a checkpoint and stop.
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
        Evaluate a new class (Basic TCM knowledge / Medical ethics / Content safety).
        """
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # Get the number of already evaluated cases for this class.
        completed_cases = len(self.general_assessment_task_results[class_name])
        
        # Create a progress bar; set the initial position to the number of already evaluated cases.
        pbar = tqdm(
            total=len(cases), 
            desc=f"评测进度 - {class_name}",
            position=0,
            leave=True,
            initial=completed_cases
        )
        
        try:
            # Evaluate case by case.
            for case_idx in range(completed_cases, len(cases)):
                case = cases[case_idx]
                
                pbar.set_description(f"评测案例 {case_idx + 1}/{len(cases)} - {class_name}")
                
                # Evaluate a single case.
                case_result = self._evaluate_new_class_single_case(case, model_interface, pbar)
                self.general_assessment_task_results[class_name].append(case_result)
                
                # Update the progress bar.
                pbar.update(1)
                
                # Save checkpoints periodically.
                if (case_idx + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_data = {
                        "completed_cases": len(self.detailed_results),  
                        "detailed_results": self.detailed_results,
                        "general_assessment_task_results": self.general_assessment_task_results,
                        "random_seed": self.config.random_seed,
                    }
                    save_checkpoint(checkpoint_data, checkpoint_path)
                    pbar.write(f"已保存断点：完成 {case_idx + 1} 个{class_name}案例")
        
        except KeyboardInterrupt:
            logger.info(f"{class_name}评测被用户中断")
            # Save current progress.
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
        Evaluate a single case for the new classes (Basic TCM knowledge / Medical ethics / Content safety).
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
            "class": case.get("class", "未知类别")  # Attach class information.
        }
        
        # Run 3 rounds of multiple-choice evaluation.
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
        Evaluate a single SDT case (fixed order + combined multiple-choice evaluation).
        """
        # Unify case_id retrieval logic to stay consistent across evaluators.
        case_id = case.get("case_id") or case.get("id") or "unknown_case"

        case_result = {
            "case_id": case_id,
            "instruction": case["instruction"],
            "ground_truth": case["output"],
            "diagnosis": case["中医疾病诊断"],
            "dimension_scores": {},
            "model_responses": {},
            "detailed_evaluation_results": {}  # Detailed evaluation results.
        }
        
        # 1) Combined multiple-choice evaluation (syndrome / property / location / principles).
        try:
            scores_mc, first_answers, detailed_mc_results = self.multiple_choice_evaluator.evaluate_combined(case, model_interface, pbar)
            for dim in ["证型", "病性", "病位", "治则治法"]:
                case_result["dimension_scores"][dim] = scores_mc.get(dim, 0.0)
                # For downstream dependency (syndrome -> reward model), record only the first-run (original order) letter string.
                case_result["model_responses"][dim] = f"{dim}答案：" + first_answers.get(dim, "")
            
            # Save detailed results for the 3-round multiple-choice evaluation.
            case_result["detailed_evaluation_results"]["multiple_choice"] = detailed_mc_results
            
        except Exception as e:
            logger.error(f"选择题合并评测失败: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            for dim in ["证型", "病性", "病位", "治则治法"]:
                case_result["dimension_scores"][dim] = 0.0
                case_result["model_responses"][dim] = f"评测失败: {str(e)}"
            case_result["detailed_evaluation_results"]["multiple_choice"] = {"error": str(e)}
        
        # 2) LLM-judged scoring for cause/mechanism.
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
        
        # 3) Prescription-related dimensions (formula-match / compatibility / contraindications / herb safety / pregnancy contraindications).
        try:
            scores_sp, responses_sp = self.reward_model_evaluator.evaluate_all(
                case, model_interface, first_answers.get("证型", ""), pbar, llm_judge_evaluator=self.llm_judge_evaluator
            )
            for dim in ["方证契合度", "方剂配伍规律", "配伍禁忌", "药材安全性分析", "妊娠禁忌"]:
                case_result["dimension_scores"][dim] = scores_sp.get(dim, 0.0)
                if dim == "方证契合度":
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
 
        # 4) LLM-judged scoring (CoT completeness / decoction method / precautions / modifications).
        try:
            gt_prescription_herbs = case["output"].get("药物组成及用量", "")
            gt_syndrome = case["output"].get("证型", "")
            gt_treatment_principles = case["output"].get("治则治法", "")

            scores_llm, responses_llm = self.llm_judge_evaluator.evaluate_all(
                case, model_interface, pbar,
                syndrome_choice=gt_syndrome,                      # Precautions/modifications: provide the reference syndrome.
                prescription_herbs=gt_prescription_herbs,         # Use the reference prescription herbs for the 3 generations.
                treatment_principles=gt_treatment_principles,     # Decoction method: provide reference treatment principles.
                skip_think=self.skip_think                        # Pass through skip_think.
            )
            
            # Decide which dimensions to process based on skip_think.
            llm_dimensions = ["煎服方法", "注意事项", "随症加减"]
            if not self.skip_think:
                llm_dimensions = ["CoT内容完备性"] + llm_dimensions
            
            for dim in llm_dimensions:
                case_result["dimension_scores"][dim] = scores_llm.get(dim, 0.0)
                case_result["model_responses"][dim] = responses_llm.get(dim, "")
            
            # CoT accuracy (hallucination) evaluation.
            if not self.skip_think and "CoT内容完备性" in responses_llm:
                think_content = responses_llm["CoT内容完备性"]
                
                # Run hallucination evaluation.
                cot_accuracy_score, hallucination_details = \
                    self.llm_judge_evaluator.evaluate_hallucination(case, think_content, pbar)
                
                # Save the score into dimension_scores.
                case_result["dimension_scores"]["CoT准确性"] = cot_accuracy_score
                
                # Save full details.
                case_result["hallucination_details"] = hallucination_details
                
        except Exception as e:
            logger.error(f"LLM合并评测失败: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            # Decide which dimensions to process based on skip_think.
            llm_dimensions = ["煎服方法", "注意事项", "随症加减"]
            if not self.skip_think:
                llm_dimensions = ["CoT内容完备性", "CoT准确性"] + llm_dimensions
            
            for dim in llm_dimensions:
                case_result["dimension_scores"][dim] = 0.0
                # CoT accuracy is not stored in model_responses.
                if dim != "CoT准确性":
                    case_result["model_responses"][dim] = f"评测失败: {str(e)}"
        
        return case_result

    def _get_prescription_herbs_from_cache(self, case: Dict[str, Any]) -> str:
        """
        Get prescription composition from the reward-model evaluator's cache.

        Args:
            case: 案例数据

        Returns:
            Prescription composition string.
        """
        try:
            case_id = case.get("case_id") or case.get("id") or "unknown_case"
            if case_id in self.reward_model_evaluator.prescription_cache:
                prescription_response = self.reward_model_evaluator.prescription_cache[case_id].get("raw_response", "")
                # Parse prescription information.
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
        Get the syndrome contents from detailed multiple-choice results.

        Args:
            detailed_mc_results: 选择题详细评测结果

        Returns:
            Syndrome contents string; multiple syndromes are separated by semicolons.
        """
        try:
            syndrome_mapping = detailed_mc_results.get("syndrome_mapping", {})
            letter_to_content = syndrome_mapping.get("letter_to_content", {})
            first_run_letters = syndrome_mapping.get("first_run_letters", [])

            # Convert letters to concrete contents.
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
        Get treatment principles from detailed multiple-choice results (first-run original-order selection).

        Args:
            detailed_mc_results: 选择题详细评测结果

        Returns:
            Treatment-principles contents string; multiple items are separated by semicolons.
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
        Compute final scores.

        Returns:
            Final score dict.
        """
        if not self.detailed_results and not any(self.general_assessment_task_results.values()):
            return {"total_score": 0.0, "dimension_scores": {}}
        
        result = {
            "total_score": 0.0,
            "dimension_scores": {},
            "general_assessment_tasks": {},
            "num_cases": len(self.detailed_results)  
        }
        
        # Count total evaluated cases across all classes.
        total_cases = len(self.detailed_results)
        for class_results in self.general_assessment_task_results.values():
            total_cases += len(class_results)
        
        result["num_cases"] = total_cases
        
        # SDT task (TCM SDT) score.
        if self.detailed_results:
            # Average score for each evaluation dimension.
            dimension_avg_scores = {}
            for dimension in self.evaluation_dimensions:
                scores = [
                    result["dimension_scores"].get(dimension, 0.0) 
                    for result in self.detailed_results
                ]
                dimension_avg_scores[dimension] = np.mean(scores) if scores else 0.0
            
            # SDT task score = mean over the dimension means.
            actual_dimension_count = len(dimension_avg_scores)
            tcm_score = np.mean(list(dimension_avg_scores.values())) if dimension_avg_scores else 0.0
            result["dimension_scores"] = dimension_avg_scores
            result["tcm_score"] = tcm_score
            
            # Log the dimension count used for SDT task scoring.
            logger.info(f"中医辨证论治类别总分计算: 使用 {actual_dimension_count} 个维度, 平均分: {tcm_score:.4f}")
            if self.skip_think:
                logger.info("已跳过Think内容完备性维度，使用14维度计算")
        else:
            tcm_score = 0.0
            result["tcm_score"] = tcm_score
        
        # General assessment tasks (multiple-choice accuracy). Only classes with samples are included.
        general_task_scores = {}
        participating_classes: List[str] = []

        # Weighted total score across the four tasks.
        # Requested weights: SDT 40%, Basic 30%, Ethics 15%, Safety 15%.
        task_weights = {
            "中医辨证论治": 0.40,
            "中医基础知识": 0.30,
            "医学伦理": 0.15,
            # Display name used in outputs/reports.
            "大模型内容安全": 0.15,
            # Backward-compatible key (if any downstream still uses the original name).
            "安全问题": 0.15,
        }
        weighted_numerator = 0.0
        weighted_denom = 0.0

        # SDT task participates if there are SDT cases.
        if self.detailed_results:
            participating_classes.append("中医辨证论治")
            weighted_numerator += tcm_score * task_weights["中医辨证论治"]
            weighted_denom += task_weights["中医辨证论治"]

        # Compute per-class means; include only classes with samples.
        for class_name, results in self.general_assessment_task_results.items():
            if results:
                scores = [result["dimension_scores"].get("accuracy", 0.0) for result in results]
                # Accuracy is already in [0, 1].
                avg_score = np.mean(scores) if scores else 0.0
                out_name = DISPLAY_NAME_MAP.get(class_name, class_name)
                general_task_scores[out_name] = avg_score
                participating_classes.append(out_name)

                w = task_weights.get(out_name) or task_weights.get(class_name)
                if w is not None:
                    weighted_numerator += avg_score * float(w)
                    weighted_denom += float(w)

        result["general_assessment_tasks"] = general_task_scores
        result["participating_classes"] = participating_classes

        # Weighted average over tasks that actually have samples.
        total_score = (weighted_numerator / weighted_denom) if weighted_denom > 0 else 0.0
        result["total_score"] = total_score
        
        return result

def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="TCM-BEST4SDT评测")
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
    
    # Load config.
    config = EvaluationConfig()
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create model interface.
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
    
    # Run evaluation.
    benchmark = TCMBenchmark(config, skip_think=args.skip_think)
    results = benchmark.run_evaluation(model_interface, args.output_dir, args.resume)
    
    print(f"评测完成！总分：{results['total_score']:.4f}")

if __name__ == "__main__":
    main()