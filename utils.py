#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块

提供日志配置、断点保存/加载等工具函数
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
    """
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器（如果指定了日志文件）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

def save_checkpoint(data: Dict[str, Any], checkpoint_path: str):
    """
    保存断点数据
    
    Args:
        data: 要保存的数据
        checkpoint_path: 断点文件路径
    """
    try:
        # 添加时间戳
        data["timestamp"] = datetime.now().isoformat()
        
        # 保存到临时文件，然后重命名（原子操作）
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 重命名为最终文件
        os.rename(temp_path, checkpoint_path)
        
    except Exception as e:
        logging.error(f"保存断点失败: {e}")
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    加载断点数据
    
    Args:
        checkpoint_path: 断点文件路径
        
    Returns:
        断点数据，如果加载失败返回None
    """
    try:
        if not os.path.exists(checkpoint_path):
            return None
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"成功加载断点: {checkpoint_path}")
        return data
        
    except Exception as e:
        logging.error(f"加载断点失败: {e}")
        return None

def ensure_dir(path: str):
    """
    确保目录存在
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)

def format_score(score: float, precision: int = 2) -> str:
    """
    格式化分数显示
    
    Args:
        score: 分数
        precision: 小数位数
        
    Returns:
        格式化后的分数字符串
    """
    return f"{score:.{precision}f}"

def format_percentage(score: float, precision: int = 1) -> str:
    """
    格式化百分比显示
    
    Args:
        score: 分数 (0.0-1.0)
        precision: 小数位数
        
    Returns:
        格式化后的百分比字符串
    """
    return f"{score * 100:.{precision}f}%"

def calculate_statistics(scores: list) -> Dict[str, float]:
    """
    计算分数统计信息
    
    Args:
        scores: 分数列表
        
    Returns:
        统计信息字典
    """
    if not scores:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }
    
    import numpy as np
    
    scores_array = np.array(scores)
    
    return {
        "mean": float(np.mean(scores_array)),
        "median": float(np.median(scores_array)),
        "std": float(np.std(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "count": len(scores)
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        是否有效
    """
    required_fields = [
        "data_path",
        "reward_api_host",
        "reward_api_port",
        "llm_judge_api_host",
        "llm_judge_api_port",
        "reward_api_key",
        "llm_judge_api_key",
    ]
    
    for field in required_fields:
        if field not in config:
            logging.error(f"配置缺少必需字段: {field}")
            return False
    
    # 验证文件路径
    if not os.path.exists(config["data_path"]):
        logging.error(f"数据文件不存在: {config['data_path']}")
        return False
    
    return True

def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        默认配置字典
    """
    return {
        "data_path": "TCM-BEST4SDT.jsonl",
        "local_model_gpu_id": -1,
        "reward_api_host": "127.0.0.1",
        "reward_api_port": 8001,
        "reward_model_name": "Fangzheng-RM",
        "reward_api_key": "YOUR_REWARD_API_KEY",
        "llm_judge_api_host": "127.0.0.1", 
        "llm_judge_api_port": 8002,
        "llm_judge_model_name": "Qwen3-32B",
        "llm_judge_api_key": "YOUR_JUDGE_API_KEY",
        "random_seed": None,  # None表示自动生成随机种子
        "max_retries": 3,
        "checkpoint_interval": 10
    }

def merge_configs(default_config: Dict[str, Any], 
                 user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置
    
    Args:
        default_config: 默认配置
        user_config: 用户配置
        
    Returns:
        合并后的配置
    """
    merged_config = default_config.copy()
    
    for key, value in user_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            # 递归合并字典
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config
