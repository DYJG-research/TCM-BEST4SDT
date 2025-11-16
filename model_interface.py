#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型接口模块

提供统一的模型调用接口，支持API和本地模型
"""

import logging
import re
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

logger = logging.getLogger(__name__)

class ModelInterface(ABC):
    """模型接口抽象基类"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 4096, 
                temperature: float = 0.0, clean_think: bool = True) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            clean_think: 是否清理<think>标签内容（除Think完备性外应为True）
            
        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        pass
class APIModelInterface(ModelInterface):
    """API模型接口（OpenAI兼容）"""
    
    def __init__(self, api_url: str, model_name: str, api_key: str):
        """
        初始化API模型接口
        
        Args:
            api_url: API地址
            model_name: 模型名称
            api_key: API Key
        """
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )
        self._last_reasoning_text = ""

        logger.info(f"初始化API模型接口: {api_url} 模型: {model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 4096, 
                temperature: float = 0.0, clean_think: bool = True) -> str:
        """
        通过API生成文本（流式输出至终端）
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            clean_think: 是否清理<think>标签内容
            
        Returns:
            生成的文本（根据clean_think清理或保留think内容）
        """
        # 流式输出
        full_text_parts = []
        reasoning_parts = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            # rea = ''
            for chunk in stream:
                try:
                    # if hasattr(chunk.choices[0].delta,'reasoning_content'):
                    #     rea += chunk.choices[0].delta.reasoning_content
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                try:
                    reasoning_delta = chunk.choices[0].delta.reasoning_content
                except Exception:
                    reasoning_delta = None
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                if delta:
                    print(delta, end="", flush=True)
                    full_text_parts.append(delta)
        except Exception as e:
            logger.error(f"API流式调用失败: {e}")
            raise
        finally:
            # 流式结束后换行，避免挤在同一行
            print()
        
        self._last_reasoning_text = "".join(reasoning_parts).strip()

        content = "".join(full_text_parts)
        if clean_think:
            return self._clean_think_tags(content)
        return content.strip()
    
    def _clean_think_tags(self, content: str) -> str:
        """清理<think>标签内容"""
        # 同时存在成对标签：移除<think>...</think>
        if "<think>" in content and "</think>" in content:
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return cleaned.strip()
        # 仅存在闭合标签：保留</think>之后的最终答案
        if "</think>" in content and "<think>" not in content:
            idx = content.find("</think>")
            return content[idx + len("</think>"):].strip()
        # 无或其他情况：原样返回
        return content.strip()

    def get_last_reasoning(self) -> str:
        """
        返回最近一次 generate 调用期间通过流式接口收集到的 reasoning_content。
        若后端不支持 reasoning_content，则返回空字符串。
        """
        return self._last_reasoning_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": "api",
            "api_url": self.api_url,
            "model_name": self.model_name
        }

class LocalModelInterface(ModelInterface):
    """本地模型接口"""
    
    def __init__(self, model_path: str, gpu_id: int = -1):
        """
        初始化本地模型接口（完全参考tuili.py的加载方式）
        
        Args:
            model_path: 模型路径
            gpu_id: GPU ID（实际由CUDA_VISIBLE_DEVICES环境变量控制），
                   -1表示使用CPU，>=0表示使用GPU模式
        
        Note:
            使用GPU时，请在运行脚本前设置 CUDA_VISIBLE_DEVICES 环境变量
            例如: CUDA_VISIBLE_DEVICES=5 python tcm_benchmark.py ...
        """
        self.model_path = model_path
        self.gpu_id = gpu_id
        
         # 加载模型和分词器
        try:
            # 加载分词器（完全参考tuili.py）
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=True
            )
            
            # 加载模型（完全参考tuili.py的方式）
            if gpu_id >= 0 and torch.cuda.is_available():
                # GPU模式：使用device_map="auto"自动分配
                # 注意：实际使用的GPU由环境变量CUDA_VISIBLE_DEVICES控制
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.device = "cuda:0"  # device_map="auto"时使用cuda:0
                
                # 显示当前可见的GPU
                visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
                logger.info(f"本地模型加载成功: {model_path}")
                logger.info(f"  - CUDA_VISIBLE_DEVICES: {visible_devices}")
                logger.info(f"  - 使用设备: {self.device}")
            else:
                # CPU模式
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.device = "cpu"
                self.model = self.model.to(self.device)
                logger.info(f"本地模型加载成功: {model_path} 设备: cpu")
            
        except Exception as e:
            logger.error(f"本地模型加载失败: {e}")
            raise
    def generate(self, prompt: str, max_tokens: int = 4096, 
                temperature: float = 0.0, clean_think: bool = True) -> str:
        """
        生成文本（带流式输出到终端）
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            clean_think: 是否清理<think>标签内容
            
        Returns:
            生成的文本
        """
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 根据设备类型移动inputs
            if self.device == "auto":
                inputs = inputs.to(self.model.device)
            else:
                inputs = inputs.to(self.device)
            
            # 创建流式输出器
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 生成参数
            generation_kwargs = {
                **inputs,
                'max_new_tokens': max_tokens,
                'temperature': temperature if temperature > 0 else None,
                'do_sample': temperature > 0,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'streamer': streamer,
            }
            
            # 在后台线程中生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 收集完整文本并实时显示
            generated_parts = []
            for new_text in streamer:
                print(new_text, end='', flush=True)
                generated_parts.append(new_text)
            
            # 等待生成完成
            thread.join()
            
            # 换行
            print()
            
            # 完整生成内容
            content = ''.join(generated_parts).strip()
            
            # 根据clean_think参数决定是否清理<think>标签
            if clean_think:
                content = self._clean_think_tags(content)
            
            return content
            
        except Exception as e:
            logger.error(f"本地模型生成失败: {e}")
            raise
    
    def _clean_think_tags(self, content: str) -> str:
        """清理<think>标签内容"""
        # 移除<think>...</think>标签及其内容
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return cleaned.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": "local",
            "model_path": self.model_path,
            "device": self.device
        }
