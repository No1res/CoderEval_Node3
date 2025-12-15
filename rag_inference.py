#!/usr/bin/env python3
"""
CoderEval RAG 上下文推理脚本
============================

读取 RAG 检索生成的上下文数据集，调用模型进行推理。
支持三种推理后端：vLLM（推荐）、Transformers、API

使用方法:

1. vLLM 推理（推荐，高效）:
    python rag_inference.py \
        --input ./rag_contexts \
        --output ./rag_inference_results \
        --method bm25 \
        --backend vllm \
        --model-path /path/to/Qwen3-4B-Instruct-2507 \
        --tensor-parallel-size 1 \
        --all-lengths

2. Transformers 推理:
    python rag_inference.py \
        --input ./rag_contexts \
        --method bm25 \
        --backend transformers \
        --model-path /path/to/Qwen3-4B-Instruct-2507 \
        --all-lengths

3. API 推理（vLLM 服务器或 OpenAI 兼容 API）:
    python rag_inference.py \
        --input ./rag_contexts \
        --method bm25 \
        --backend api \
        --api-url http://localhost:8000/v1 \
        --model-name Qwen3-4B-Instruct-2507 \
        --all-lengths
"""

import json
import os
import sys
import argparse
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Complete the following function based on the given context. "
    "Only output the function implementation, without any explanation."
)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGInferenceConfig:
    """RAG 推理配置"""
    input_dir: str = "./rag_contexts"
    output_dir: str = "./rag_inference_results"
    
    # 检索方法
    method: str = "bm25"  # bm25 或 jaccard
    
    # 要处理的上下文长度
    context_length: Optional[int] = None
    process_all_lengths: bool = False
    
    # 推理后端: vllm, transformers, api
    backend: str = "vllm"
    
    # 模型配置
    model_name: str = "Qwen3-4B-Instruct-2507"
    model_path: str = ""
    api_base_url: str = ""
    api_key: str = ""
    
    # 推理配置
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    num_samples: int = 10
    
    # vLLM 特定配置
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None  # 模型最大上下文长度
    max_num_seqs: int = 256
    
    # 注意力输出配置
    save_attention: bool = True
    attention_output_dir: str = "./attention_data"
    
    # 其他
    batch_size: int = 1
    timeout: int = 60
    seed: int = 42
    resume: bool = True


class RAGModelInference:
    """
    RAG 模型推理接口
    支持三种后端：vLLM（推荐）、Transformers、API
    """
    
    def __init__(self, config: RAGInferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.llm = None  # vLLM LLM 对象
        self.sampling_params = None  # vLLM SamplingParams
        
    def load_model(self):
        """加载模型"""
        backend = self.config.backend.lower()
        
        if backend == "vllm":
            self._load_vllm_model()
        elif backend == "transformers":
            self._load_transformers_model()
        elif backend == "api":
            if not self.config.api_base_url:
                raise ValueError("API 模式需要配置 --api-url")
            logger.info(f"使用 API 模式: {self.config.api_base_url}")
        else:
            raise ValueError(f"未知的后端类型: {backend}，支持: vllm, transformers, api")
    
    def _load_vllm_model(self):
        """加载 vLLM 模型"""
        try:
            from vllm import LLM, SamplingParams
            
            if not self.config.model_path:
                raise ValueError("vLLM 模式需要配置 --model-path")
            
            logger.info(f"使用 vLLM 加载模型: {self.config.model_path}")
            logger.info(f"  tensor_parallel_size: {self.config.tensor_parallel_size}")
            logger.info(f"  gpu_memory_utilization: {self.config.gpu_memory_utilization}")
            
            # 构建 vLLM 参数
            llm_kwargs = {
                "model": self.config.model_path,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "trust_remote_code": True,
                "seed": self.config.seed,
                "max_num_seqs": self.config.max_num_seqs,
            }
            
            # 如果指定了最大模型长度
            if self.config.max_model_len:
                llm_kwargs["max_model_len"] = self.config.max_model_len
            
            self.llm = LLM(**llm_kwargs)
            
            # 创建 SamplingParams
            self.sampling_params = SamplingParams(
                n=self.config.num_samples,  # 一次生成多个样本
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                seed=self.config.seed,
            )
            
            # 获取 tokenizer 用于 chat template
            self.tokenizer = self.llm.get_tokenizer()
            
            logger.info("vLLM 模型加载完成")
            
        except ImportError:
            logger.error("请安装 vLLM: pip install vllm")
            raise
    
    def _load_transformers_model(self):
        """加载 Transformers 模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            if not self.config.model_path:
                raise ValueError("Transformers 模式需要配置 --model-path")
            
            logger.info(f"使用 Transformers 加载模型: {self.config.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            # 构建模型加载参数
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
                "trust_remote_code": True,
            }

            # 如果需要保存注意力，必须使用 eager attention 实现
            # SDPA (Scaled Dot Product Attention) 不支持 output_attentions
            if self.config.save_attention:
                model_kwargs["attn_implementation"] = "eager"
                logger.info("  注意力输出已启用 (attn_implementation='eager')")
                logger.warning("  注意: eager 模式比 SDPA/Flash Attention 慢，仅在需要注意力分析时使用")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            logger.info("Transformers 模型加载完成")
            
        except ImportError:
            logger.error("请安装 transformers: pip install transformers torch")
            raise
    
    def _build_prompt(self, user_prompt: str) -> str:


        """构建完整的 prompt（包括 chat template）"""

        messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return text
            except Exception as e:
                logger.warning(f"apply_chat_template 失败，回退到手写模板: {e}")

            # fallback：保持与 messages 一致（包含 system + user）
        return (
            "<|im_start|>system\n"
            f"{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
    )
    
    def generate(self, prompt: str, num_samples: int = 1, 
                 return_attention: bool = False) -> Dict:
        """
        生成代码
        
        Returns:
            {
                'generated_codes': List[str],
                'attention_data': Optional[...]
            }
        """
        backend = self.config.backend.lower()
        
        if backend == "vllm":
            return self._generate_vllm(prompt, num_samples)
        elif backend == "transformers":
            return self._generate_transformers(prompt, num_samples, return_attention)
        else:
            return self._generate_api(prompt, num_samples)
    
    def _generate_vllm(self, prompt: str, num_samples: int) -> Dict:
        """使用 vLLM 生成"""
        full_prompt = self._build_prompt(prompt)
        
        # 更新 SamplingParams 的 n 值
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=num_samples,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=self.config.seed,
        )
        
        # 生成
        outputs = self.llm.generate([full_prompt], sampling_params)
        
        generated_codes = []
        for output in outputs:
            for completion in output.outputs:
                code = self._extract_code(completion.text)
                generated_codes.append(code)
        
        return {
            'generated_codes': generated_codes,
            'attention_data': None  # vLLM 目前不直接支持注意力输出
        }
    
    def generate_batch(self, prompts: List[str], num_samples: int = 1) -> List[Dict]:
        """
        批量生成（仅 vLLM 支持高效批处理）
        
        Args:
            prompts: prompt 列表
            num_samples: 每个 prompt 生成的样本数
            
        Returns:
            List[Dict]: 每个 prompt 的生成结果
        """
        if self.config.backend.lower() != "vllm":
            # 非 vLLM 后端退化为串行处理
            return [self.generate(p, num_samples) for p in prompts]
        
        from vllm import SamplingParams
        
        # 构建所有完整 prompt
        full_prompts = [self._build_prompt(p) for p in prompts]
        
        sampling_params = SamplingParams(
            n=num_samples,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=self.config.seed,
        )
        
        # 批量生成
        outputs = self.llm.generate(full_prompts, sampling_params)
        
        results = []
        for output in outputs:
            generated_codes = []
            for completion in output.outputs:
                code = self._extract_code(completion.text)
                generated_codes.append(code)
            
            results.append({
                'generated_codes': generated_codes,
                'attention_data': None
            })
        
        return results
    
    def _generate_transformers(self, prompt: str, num_samples: int,
                            return_attention: bool = False) -> Dict:
        """Transformers 生成：只保存可回放材料（ids/raw），注意力统一后处理回放做。"""

        full_prompt = self._build_prompt(prompt)

        enc = self.tokenizer(full_prompt,return_tensors="pt",add_special_tokens=False)

        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        prompt_input_ids = enc["input_ids"][0].tolist()
        prompt_len_tokens = len(prompt_input_ids)

        generated_codes = []
        generated_texts_raw = []
        generated_token_ids = []   # 仅生成部分
        sequence_token_ids = []    # prompt+生成

        sample_seeds = []

        for i in range(num_samples):
            sample_seed = int(self.config.seed + i)
            sample_seeds.append(sample_seed)

            # 让每个 sample 可复现
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)

            with torch.no_grad():
                outputs = self.model.generate(
                    **enc,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,   # 关键：拿 sequences
                    output_scores=False,
                )

            seq = outputs.sequences[0]
            seq_ids = seq.tolist()
            gen_ids = seq[prompt_len_tokens:].tolist()

            sequence_token_ids.append(seq_ids)
            generated_token_ids.append(gen_ids)

            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            generated_texts_raw.append(raw_text)
            generated_codes.append(self._extract_code(raw_text))

        return {
            "generated_codes": generated_codes,
            "generated_texts_raw": generated_texts_raw,
            "generated_token_ids": generated_token_ids,
            "sequence_token_ids": sequence_token_ids,

            "full_prompt": full_prompt,
            "prompt_len_tokens": prompt_len_tokens,
            "prompt_input_ids": prompt_input_ids,
            "sample_seeds": sample_seeds,

            # 彻底不再在 generate 阶段输出注意力
            "attention_data": None,
        }
    
    def _generate_api(self, prompt: str, num_samples: int) -> Dict:
        """使用 API 生成（兼容 vLLM 服务器和 OpenAI API）"""
        import requests
        
        generated_codes = []
        
        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Complete the following function based on the given context. Only output the function implementation."},
            {"role": "user", "content": prompt}
        ]
        
        for i in range(num_samples):
            try:
                headers = {"Content-Type": "application/json"}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                
                data = {
                    "model": self.config.model_name,
                    "messages": messages,
                    "max_tokens": self.config.max_new_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p
                }
                
                response = requests.post(
                    f"{self.config.api_base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    generated_codes.append(self._extract_code(content))
                else:
                    logger.warning(f"API 返回错误: {response.status_code} - {response.text}")
                    generated_codes.append("")
                    
            except Exception as e:
                logger.warning(f"API 调用异常: {e}")
                generated_codes.append("")
        
        return {
            'generated_codes': generated_codes,
            'attention_data': None
        }
    
    def _extract_code(self, text: str) -> str:
        """从生成的文本中提取代码"""
        # 移除 markdown 代码块标记
        text = re.sub(r'^```python\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'```$', '', text)
        
        # 移除思考过程（某些模型会输出 <think>...</think>）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检测代码开始
            if stripped.startswith('def ') or stripped.startswith('async def ') or \
               stripped.startswith('class ') or \
               stripped.startswith('import ') or stripped.startswith('from ') or \
               stripped.startswith('@'):
                in_code = True
            
            if in_code:
                code_lines.append(line)
            elif line.startswith('    ') or line.startswith('\t'):
                # 缩进的内容也算代码
                code_lines.append(line)
        
        result = '\n'.join(code_lines).strip()
        return result if result else text.strip()


class RAGInferenceRunner:
    """
    RAG 推理运行器
    """
    
    def __init__(self, config: RAGInferenceConfig):
        self.config = config
        self.model_inference = None
        self.available_lengths = []
        
    def setup(self):
        """初始化"""
        if not os.path.exists(self.config.input_dir):
            raise FileNotFoundError(f"输入目录不存在: {self.config.input_dir}")
        
        # 检测可用的上下文长度
        self.available_lengths = self._detect_context_lengths()
        logger.info(f"可用的上下文长度: {self.available_lengths}")
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if self.config.save_attention:
            os.makedirs(self.config.attention_output_dir, exist_ok=True)
        
        # 加载模型
        self.model_inference = RAGModelInference(self.config)
        self.model_inference.load_model()
        
        logger.info("RAG 推理器初始化完成")
    
    def _detect_context_lengths(self) -> List[int]:
        """检测可用的上下文长度"""
        lengths = []
        method = self.config.method
        
        for f in os.listdir(self.config.input_dir):
            if f.startswith(f'rag_{method}_') and f.endswith('tokens.jsonl'):
                try:
                    length = int(f.replace(f'rag_{method}_', '').replace('tokens.jsonl', ''))
                    lengths.append(length)
                except ValueError:
                    pass
        
        return sorted(lengths)
    
    def load_rag_contexts(self, context_length: int) -> List[Dict]:
        """加载 RAG 上下文数据"""
        input_file = os.path.join(
            self.config.input_dir,
            f"rag_{self.config.method}_{context_length}tokens.jsonl"
        )
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到数据文件: {input_file}")
        
        tasks = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        
        logger.info(f"加载了 {len(tasks)} 个任务 (method={self.config.method}, length={context_length})")
        return tasks
    
    def get_completed_tasks(self, context_length: int) -> set:
        """获取已完成的任务 ID"""
        output_file = os.path.join(
            self.config.output_dir,
            f"results_{self.config.method}_{context_length}tokens.jsonl"
        )
        
        completed = set()
        if os.path.exists(output_file) and self.config.resume:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        completed.add(result.get('_id', ''))
        
        return completed
    
    def run_inference(self, context_length: int):
        """运行推理"""
        tasks = self.load_rag_contexts(context_length)
        completed = self.get_completed_tasks(context_length)
        
        # 过滤已完成的任务
        pending_tasks = [t for t in tasks if t.get('_id', '') not in completed]
        
        output_file = os.path.join(
            self.config.output_dir,
            f"results_{self.config.method}_{context_length}tokens.jsonl"
        )
        
        total = len(tasks)
        skipped = len(completed)
        
        logger.info(f"开始推理: method={self.config.method}, length={context_length}")
        logger.info(f"  总任务数: {total}, 已完成: {skipped}, 待处理: {len(pending_tasks)}")
        
        # 根据后端选择处理方式
        if self.config.backend == "vllm" and self.config.batch_size > 1:
            self._run_inference_batch(pending_tasks, context_length, output_file)
        else:
            self._run_inference_sequential(pending_tasks, context_length, output_file)
        
        logger.info(f"  完成! 上下文长度 {context_length} 推理完毕")
    
    def _run_inference_batch(self, tasks: List[Dict], context_length: int, output_file: str):
        """批量推理 (vLLM)"""
        batch_size = self.config.batch_size
        total = len(tasks)
        processed = 0
        
        attention_data_list = []
        
        with open(output_file, 'a', encoding='utf-8') as fw:
            # 按批次处理
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch_tasks = tasks[batch_start:batch_end]
                
                # 提取 prompts
                prompts = [t.get('prompt', '') for t in batch_tasks]
                prompts = [p for p in prompts if p]  # 过滤空 prompt
                
                if not prompts:
                    continue
                
                start_time = time.time()
                
                try:
                    # 批量生成
                    outputs = self.model_inference.generate_batch(prompts, self.config.num_samples)
                    elapsed = time.time() - start_time
                    
                    # 保存结果
                    for task, output in zip(batch_tasks, outputs):
                        task_id = task.get('_id', '')
                        
                        result = {
                            '_id': task_id,
                            'name': task.get('name', ''),
                            'project': task.get('project', ''),
                            'level': task.get('level', ''),
                            'retrieval_method': self.config.method,
                            'context_length': context_length,
                            'actual_tokens': task.get('actual_tokens', 0),
                            'num_retrieved': task.get('num_retrieved', 0),
                            'inference_time': elapsed / len(batch_tasks),
                            'generate_results': output['generated_codes']
                        }
                        
                        fw.write(json.dumps(result, ensure_ascii=False) + '\n')
                        processed += 1
                    
                    fw.flush()
                    
                except Exception as e:
                    logger.error(f"批次 {batch_start}-{batch_end} 推理失败: {e}")
                    # 回退到逐个处理
                    for task in batch_tasks:
                        task_id = task.get('_id', '')
                        result = {
                            '_id': task_id,
                            'retrieval_method': self.config.method,
                            'context_length': context_length,
                            'error': str(e),
                            'generate_results': []
                        }
                        fw.write(json.dumps(result, ensure_ascii=False) + '\n')
                        processed += 1
                    fw.flush()
                
                if processed % (batch_size * 5) == 0:
                    logger.info(f"  进度: {processed}/{total}")
        
        # 保存注意力数据
        if attention_data_list:
            self._save_attention_data(attention_data_list, context_length)
    
    def _run_inference_sequential(self, tasks: List[Dict], context_length: int, output_file: str):
        """串行推理"""
        total = len(tasks)
        processed = 0
        
        attention_data_list = []
        
        with open(output_file, 'a', encoding='utf-8') as fw:
            for idx, task in enumerate(tasks):
                task_id = task.get('_id', '')
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"  进度: {processed}/{total}")
                
                try:
                    prompt = task.get('prompt', '')
                    
                    if not prompt:
                        continue
                    
                    start_time = time.time()
                    
                    output = self.model_inference.generate(
                        prompt,
                        self.config.num_samples,
                        return_attention=self.config.save_attention
                    )
                    
                    elapsed = time.time() - start_time
                    
                    result = {
                        '_id': task_id,
                        'name': task.get('name', ''),
                        'project': task.get('project', ''),
                        'level': task.get('level', ''),
                        'retrieval_method': self.config.method,
                        'context_length': context_length,
                        'actual_tokens': task.get('actual_tokens', 0),
                        'num_retrieved': task.get('num_retrieved', 0),
                        'inference_time': elapsed,
                        'generate_results': output['generated_codes'],

                        # ---- 新增：回放式 forward 必需材料 ----
                        "full_prompt": output.get("full_prompt"),
                        "prompt_len_tokens": output.get("prompt_len_tokens"),
                        "prompt_input_ids": output.get("prompt_input_ids"),

                        "generated_texts_raw": output.get("generated_texts_raw"),
                        "generated_token_ids": output.get("generated_token_ids"),
                        "sequence_token_ids": output.get("sequence_token_ids"),

                        "sample_seeds": output.get("sample_seeds"),

                        "generation_config": {
                            "backend": self.config.backend,
                            "model_path": self.config.model_path,
                            "model_name": self.config.model_name,
                            "max_new_tokens": self.config.max_new_tokens,
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "num_samples": self.config.num_samples,
                            "seed_base": self.config.seed,
                        },
                    }
                    
                    fw.write(json.dumps(result, ensure_ascii=False) + '\n')
                    fw.flush()
                    
                    if self.config.save_attention and output.get('attention_data'):
                        attention_data_list.append({
                            'task_id': task_id,
                            'context_length': context_length,
                            'retrieved_snippets': task.get('retrieved_snippets', []),
                            'attention': output['attention_data']
                        })
                    
                except Exception as e:
                    logger.error(f"任务 {task_id} 推理失败: {e}")
                    result = {
                        '_id': task_id,
                        'retrieval_method': self.config.method,
                        'context_length': context_length,
                        'error': str(e),
                        'generate_results': []
                    }
                    fw.write(json.dumps(result, ensure_ascii=False) + '\n')
                    fw.flush()
        
        if attention_data_list:
            self._save_attention_data(attention_data_list, context_length)
    
    def _save_attention_data(self, attention_data_list: List[Dict], context_length: int):
        """保存注意力数据"""
        attn_file = os.path.join(
            self.config.attention_output_dir,
            f"attention_{self.config.method}_{context_length}tokens.json"
        )
        with open(attn_file, 'w', encoding='utf-8') as f:
            json.dump(attention_data_list, f, ensure_ascii=False)
        logger.info(f"注意力数据已保存: {attn_file}")
    
    def run(self):
        """运行所有推理"""
        self.setup()
        
        if self.config.process_all_lengths:
            lengths_to_process = self.available_lengths
        elif self.config.context_length:
            lengths_to_process = [self.config.context_length]
        else:
            raise ValueError("请指定 --context-length 或 --all-lengths")
        
        logger.info(f"将处理: {lengths_to_process}")
        
        for length in lengths_to_process:
            if length not in self.available_lengths:
                logger.warning(f"上下文长度 {length} 不可用，跳过")
                continue
            
            self.run_inference(length)
        
        # 保存汇总
        summary = {
            'completed_at': datetime.now().isoformat(),
            'model_name': self.config.model_name,
            'retrieval_method': self.config.method,
            'num_samples': self.config.num_samples,
            'processed_lengths': lengths_to_process
        }
        
        summary_file = os.path.join(
            self.config.output_dir,
            f'inference_summary_{self.config.method}.json'
        )
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"推理汇总已保存: {summary_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='CoderEval RAG 推理脚本（支持 vLLM/Transformers/API）'
    )
    
    # 基本配置
    parser.add_argument('--input', type=str, default='./rag_contexts',
                        help='RAG 上下文目录')
    parser.add_argument('--output', type=str, default='./rag_inference_results',
                        help='输出目录')
    parser.add_argument('--method', type=str, choices=['bm25', 'jaccard'],
                        default='bm25', help='检索方法')
    
    parser.add_argument('--context-length', type=int, default=None,
                        help='要处理的上下文长度')
    parser.add_argument('--all-lengths', action='store_true',
                        help='处理所有可用的上下文长度')
    
    # 后端选择
    parser.add_argument('--backend', type=str, 
                        choices=['vllm', 'transformers', 'api'],
                        default='vllm',
                        help='推理后端: vllm (推荐), transformers, api')
    
    # 模型配置
    parser.add_argument('--model-path', type=str, default='',
                        help='本地模型路径 (vllm/transformers 模式必需)')
    parser.add_argument('--api-url', type=str, default='',
                        help='API 地址 (api 模式必需，如 http://localhost:8000/v1)')
    parser.add_argument('--api-key', type=str, default='',
                        help='API 密钥')
    parser.add_argument('--model-name', type=str,
                        default='Qwen3-4B-Instruct-2507',
                        help='模型名称')
    
    # vLLM 特定配置
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='vLLM tensor 并行数（GPU 数量）')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='vLLM GPU 内存使用率 (0.0-1.0)')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='vLLM 模型最大上下文长度')
    parser.add_argument('--max-num-seqs', type=int, default=10,
                        help='vLLM 最大序列数')
    # 生成配置
    parser.add_argument('--num-samples', type=int, default=10,
                        help='每个任务生成的代码数量 (n)')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='生成温度')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p 采样参数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 批处理配置
    parser.add_argument('--batch-size', type=int, default=8,
                        help='vLLM 批处理大小')
    
    # 注意力配置
    parser.add_argument('--save-attention', action='store_true',
                        help='保存注意力权重 (仅 transformers 后端)')
    parser.add_argument('--attention-dir', type=str, default='./attention_data',
                        help='注意力数据保存目录')
    
    # 其他
    parser.add_argument('--no-resume', action='store_true',
                        help='禁用断点续传')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = RAGInferenceConfig(
        input_dir=args.input,
        output_dir=args.output,
        method=args.method,
        context_length=args.context_length,
        process_all_lengths=args.all_lengths,
        backend=args.backend,
        model_name=args.model_name,
        model_path=args.model_path,
        api_base_url=args.api_url,
        api_key=args.api_key,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        batch_size=args.batch_size,
        seed=args.seed,
        save_attention=args.save_attention,
        attention_output_dir=args.attention_dir,
        resume=not args.no_resume
    )
    
    # 打印配置信息
    logger.info("="*60)
    logger.info("RAG 推理配置")
    logger.info("="*60)
    logger.info(f"  后端: {config.backend}")
    logger.info(f"  检索方法: {config.method}")
    logger.info(f"  模型: {config.model_name or config.model_path}")
    logger.info(f"  每任务生成数: {config.num_samples}")
    logger.info(f"  温度: {config.temperature}")
    if config.backend == "vllm":
        logger.info(f"  tensor_parallel_size: {config.tensor_parallel_size}")
        logger.info(f"  gpu_memory_utilization: {config.gpu_memory_utilization}")
        logger.info(f"  batch_size: {config.batch_size}")
    logger.info("="*60)
    
    runner = RAGInferenceRunner(config)
    runner.run()


if __name__ == '__main__':
    main()

