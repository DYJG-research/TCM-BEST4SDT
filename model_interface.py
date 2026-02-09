#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model interface module.

Provides a unified interface for model invocation, supporting both API models and local models.
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
    """Abstract base class for model interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 4096, 
                temperature: float = 0.0, clean_think: bool = True,
                system_prompt: Optional[str] = None) -> str:
        """
        Generate text.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            clean_think: Whether to remove <think>...</think> content (should be True except for CoT completeness).
            system_prompt: Optional system prompt.

        Returns:
            Generated text.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Model info dict.
        """
        pass
class APIModelInterface(ModelInterface):
    """API model interface (OpenAI-compatible)."""
    
    def __init__(self, api_url: str, model_name: str, api_key: str):
        """
        Initialize the API model interface.

        Args:
            api_url: Base URL for the API.
            model_name: Model name.
            api_key: API key.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize the OpenAI client.
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )
        self._last_reasoning_text = ""

        logger.info(f"初始化API模型接口: {api_url} 模型: {model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 4096, 
                temperature: float = 0.0, clean_think: bool = True,
                system_prompt: Optional[str] = None) -> str:
        """
        Generate text via API (stream output to the terminal).

        Args:
            prompt: Input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            clean_think: Whether to remove <think>...</think> content.
            system_prompt: Optional system prompt.

        Returns:
            Generated text (optionally cleaned based on clean_think).
        """
        # Build message list.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Streaming output.
        full_text_parts = []
        reasoning_parts = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                try:
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
            print()
        
        self._last_reasoning_text = "".join(reasoning_parts).strip()

        content = "".join(full_text_parts)
        if clean_think:
            return self._clean_think_tags(content)
        return content.strip()
    
    def _clean_think_tags(self, content: str) -> str:
        if "<think>" in content and "</think>" in content:
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return cleaned.strip()
        if "</think>" in content and "<think>" not in content:
            idx = content.find("</think>")
            return content[idx + len("</think>"):].strip()
        return content.strip()

    def get_last_reasoning(self) -> str:
        """
        返回最近一次 generate 调用期间通过流式接口收集到的 reasoning_content。
        若后端不支持 reasoning_content，则返回空字符串。
        """
        return self._last_reasoning_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_type": "api",
            "api_url": self.api_url,
            "model_name": self.model_name
        }

class LocalModelInterface(ModelInterface):
    """Local model interface."""
    
    def __init__(self, model_path: str, gpu_id: int = -1):
        """
        Initialize the local model interface.

        Args:
            model_path: Local model path.
            gpu_id: GPU ID (controlled by CUDA_VISIBLE_DEVICES env var).
                   -1 means CPU mode; >=0 means GPU mode.

        Note:
            When using GPU, set CUDA_VISIBLE_DEVICES before running, e.g.
            CUDA_VISIBLE_DEVICES=5 python tcm_benchmark.py ...
        """
        self.model_path = model_path
        self.gpu_id = gpu_id
        
        # Load model and tokenizer.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=True
            )
            
            if gpu_id >= 0 and torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.device = "cuda:0"  
                
                visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
                logger.info(f"本地模型加载成功: {model_path}")
                logger.info(f"  - CUDA_VISIBLE_DEVICES: {visible_devices}")
                logger.info(f"  - 使用设备: {self.device}")
            else:
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
                temperature: float = 0.0, clean_think: bool = True,
                system_prompt: Optional[str] = None) -> str:
        """
        Generate text (stream output to the terminal).

        Args:
            prompt: Input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            clean_think: Whether to remove <think>...</think> content.
            system_prompt: Optional system prompt.

        Returns:
            Generated text.
        """
        try:
            # If a system prompt is provided, prepend it to the prompt.
            final_prompt = prompt
            if system_prompt:
                final_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Tokenize inputs.
            inputs = self.tokenizer(final_prompt, return_tensors="pt")
            
            # Move tensors to the target device.
            if self.device == "auto":
                inputs = inputs.to(self.model.device)
            else:
                inputs = inputs.to(self.device)
            
            # Create a streamer for incremental decoding.
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generation parameters.
            generation_kwargs = {
                **inputs,
                'max_new_tokens': max_tokens,
                'temperature': temperature if temperature > 0 else None,
                'do_sample': temperature > 0,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'streamer': streamer,
            }
            
            # Generate in a background thread.
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Collect full text while streaming to stdout.
            generated_parts = []
            for new_text in streamer:
                print(new_text, end='', flush=True)
                generated_parts.append(new_text)
            
            # Wait for completion.
            thread.join()
            
            # Newline after streaming.
            print()
            
            # Full generated content.
            content = ''.join(generated_parts).strip()
            
            # Optionally strip <think> tags.
            if clean_think:
                content = self._clean_think_tags(content)
            
            return content
            
        except Exception as e:
            logger.error(f"本地模型生成失败: {e}")
            raise
    
    def _clean_think_tags(self, content: str) -> str:
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return cleaned.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_type": "local",
            "model_path": self.model_path,
            "device": self.device
        }
