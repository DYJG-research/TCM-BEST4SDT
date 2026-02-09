#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader module.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class TCMDataLoader:
    """TCM benchmark data loader."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the dataset file.
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    def load_cases(self) -> List[Dict[str, Any]]:
        """
        Load all cases.

        Returns:
            A list of cases.
        """
        logger.info(f"正在加载数据文件: {self.data_path}")
        
        cases = []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                # Load a standard JSON file directly.
                json_data = json.load(f)
            
            # Ensure the root JSON object is a list.
            if not isinstance(json_data, list):
                raise ValueError("JSON文件应该包含一个数组")
            
            for idx, data in enumerate(json_data):
                case = self._process_case(data, idx)
                if case:
                    cases.append(case)
            
            logger.info(f"成功加载 {len(cases)} 个案例")
            return cases
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON文件格式错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载数据时出错: {e}")
            raise
    
    def _process_case(self, data: Dict[str, Any], case_idx: int) -> Dict[str, Any]:
        """
        Process a single case.

        Args:
            data: Raw case record.
            case_idx: Case index.

        Returns:
            Processed case dict.
        """
        try:
            # Determine the case class.
            case_class = data.get("class", "")
            
            if case_class in ["中医基础知识", "医学伦理", "安全问题"]:
                required_fields = ["id", "question", "answer", "option", "question_type"]
                for field in required_fields:
                    if field not in data:
                        logger.warning(f"案例 {case_idx} 缺少必需字段: {field}")
                        return None
                
                case = {
                    "id": str(data["id"]),
                    "class": data["class"],
                    "question": data["question"],
                    "answer": data["answer"],
                    "option": data["option"],
                    "question_type": data["question_type"]
                }
                
                optional_fields = ["exam_type", "exam_class", "exam_subject"]
                for field in optional_fields:
                    if field in data:
                        case[field] = data[field]
                
                return case
            
            elif case_class == "中医辨证论治":
                required_fields = ["id", "instruction", "output", "中医疾病诊断"]
                for field in required_fields:
                    if field not in data:
                        logger.warning(f"案例 {case_idx} 缺少必需字段: {field}")
                        return None
                
                output_dict = self._parse_output_list(data["output"])
                if not output_dict:
                    logger.warning(f"案例 {case_idx} 的output字段解析失败")
                    return None
                
                raw_id = data["id"]
                case = {
                    "case_id": str(raw_id),
                    "instruction": data["instruction"],
                    "output": output_dict,
                    "中医疾病诊断": data["中医疾病诊断"],
                    "class": data.get("class", "中医辨证论治")
                }
                
                return case
            
            else:
                required_fields = ["id", "instruction", "output", "中医疾病诊断"]
                for field in required_fields:
                    if field not in data:
                        logger.warning(f"案例 {case_idx} 缺少必需字段: {field}")
                        return None
                
                output_dict = self._parse_output_list(data["output"])
                if not output_dict:
                    logger.warning(f"案例 {case_idx} 的output字段解析失败")
                    return None
                
                raw_id = data["id"]
                case = {
                    "case_id": str(raw_id),
                    "instruction": data["instruction"],
                    "output": output_dict,
                    "中医疾病诊断": data["中医疾病诊断"],
                    "class": data.get("class", "中医辨证论治")
                }
                
                return case
            
        except Exception as e:
            logger.error(f"处理案例 {case_idx} 时出错: {e}")
            return None
    
    def _parse_output_list(self, output_list: List[str]) -> Dict[str, Any]:
        """
        Parse an output list into a dict.

        Args:
            output_list: List of output strings.

        Returns:
            Parsed dict.
        """
        output_dict = {}
        
        for item in output_list:
            if isinstance(item, str) and '：' in item:
                key, value = item.split('：', 1)
                output_dict[key.strip()] = value.strip()
            else:
                logger.warning(f"无法解析output项目: {item}")
        
        # Validate required fields.
        required_output_fields = [
            "证型", "证型选项", "证型答案",
            "病因", "病机", 
            "病性", "病性选项", "病性答案",
            "病位", "病位选项", "病位答案",
            "治则治法", "治则治法选项", "治则治法答案",
            "药物组成及用量", "方剂配伍规律", "随症加减", "煎服方法", "注意事项"
        ]
        
        missing_fields = [field for field in required_output_fields if field not in output_dict]
        if missing_fields:
            logger.warning(f"output缺少字段: {missing_fields}")
            return None
        
        return output_dict
    
    def get_case_by_id(self, case_id: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a case by ID.

        Args:
            case_id: Case ID.
            cases: Case list.

        Returns:
            Case dict; returns None if not found.
        """
        for case in cases:
            if case.get("case_id", case.get("id", "")) == case_id:
                return case
        return None
    
    def get_statistics(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Args:
            cases: Case list.

        Returns:
            Statistics dict.
        """
        if not cases:
            return {}
        
        # Disease/class distribution.
        disease_counts = {}
        for case in cases:
            disease = case.get("中医疾病诊断", case.get("class", "未知类别"))
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        # Instruction/question length statistics.
        instruction_lengths = []
        for case in cases:
            if "instruction" in case:
                instruction_lengths.append(len(case["instruction"]))
            elif "question" in case:
                instruction_lengths.append(len(case["question"]))
        
        statistics = {
            "total_cases": len(cases),
            "disease_distribution": disease_counts,
            "instruction_length_stats": {
                "min": min(instruction_lengths) if instruction_lengths else 0,
                "max": max(instruction_lengths) if instruction_lengths else 0,
                "mean": sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
                "median": sorted(instruction_lengths)[len(instruction_lengths) // 2] if instruction_lengths else 0
            },
            "unique_diseases": len(disease_counts)
        }
        
        return statistics