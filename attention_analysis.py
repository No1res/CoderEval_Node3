#!/usr/bin/env python3
"""
CoderEval 注意力分析脚本
========================

分析模型对不同上下文的注意力分布，评估模型是否正确关注了有效信息。

功能：
1. 加载注意力数据
2. 分析注意力在不同上下文区域的分布
3. 与原始数据集的有效上下文对比
4. 生成可视化图表

使用方法:
    python attention_analysis.py \
        --attention-dir ./attention_data \
        --rag-dir ./rag_contexts \
        --dataset home/travis/builds/CoderEval4Python.json \
        --output ./attention_analysis_output \
        --method bm25
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AttentionAnalysisConfig:
    """注意力分析配置"""
    attention_dir: str = "./attention_data"
    rag_dir: str = "./rag_contexts"
    dataset_path: str = "CoderEval4Python.json"
    output_dir: str = "./attention_analysis_output"
    
    method: str = "bm25"  # bm25 或 jaccard
    
    context_lengths: List[int] = field(default_factory=lambda: [
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608
    ])


class AttentionAnalyzer:
    """
    注意力分析器
    """
    
    def __init__(self, config: AttentionAnalysisConfig):
        self.config = config
        self.oracle_contexts = {}  # task_id -> oracle context info
        self.attention_data = {}  # context_length -> List[attention_record]
        
    def load_oracle_contexts(self):
        """加载原始数据集中的有效上下文（oracle）"""
        logger.info(f"加载原始数据集: {self.config.dataset_path}")
        
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data.get('RECORDS', []):
            task_id = record.get('_id', '')
            
            # 解析 all_context 和 oracle_context
            all_context = record.get('all_context', '{}')
            oracle_context = record.get('oracle_context', '{}')
            
            try:
                all_ctx = json.loads(all_context) if isinstance(all_context, str) else all_context
                oracle_ctx = json.loads(oracle_context) if isinstance(oracle_context, str) else oracle_context
            except:
                all_ctx = {}
                oracle_ctx = {}
            
            self.oracle_contexts[task_id] = {
                'imports': all_ctx.get('import', '').split(),
                'file_context': all_ctx.get('file', ''),
                'class_context': all_ctx.get('class', ''),
                'oracle_apis': oracle_ctx.get('apis', '[]'),
                'oracle_classes': oracle_ctx.get('classes', '[]'),
                'oracle_vars': oracle_ctx.get('vars', '[]'),
                'ground_truth': record.get('code', '')
            }
        
        logger.info(f"加载了 {len(self.oracle_contexts)} 个任务的 oracle 上下文")
    
    def load_attention_data(self, context_length: int) -> List[Dict]:
        """加载指定上下文长度的注意力数据"""
        attn_file = os.path.join(
            self.config.attention_dir,
            f"attention_{self.config.method}_{context_length}tokens.json"
        )
        
        if not os.path.exists(attn_file):
            logger.warning(f"注意力文件不存在: {attn_file}")
            return []
        
        with open(attn_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_rag_context(self, context_length: int) -> Dict[str, Dict]:
        """加载 RAG 上下文数据"""
        rag_file = os.path.join(
            self.config.rag_dir,
            f"rag_{self.config.method}_{context_length}tokens.jsonl"
        )
        
        if not os.path.exists(rag_file):
            logger.warning(f"RAG 上下文文件不存在: {rag_file}")
            return {}
        
        contexts = {}
        with open(rag_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    contexts[record['_id']] = record
        
        return contexts
    
    def analyze_attention_distribution(self, attention_record: Dict, 
                                       rag_context: Dict) -> Dict:
        """
        分析单个任务的注意力分布
        
        评估注意力是否集中在有效区域：
        1. 检索到的代码片段区域
        2. 函数签名和 docstring 区域
        3. 与 oracle 上下文相关的区域
        """
        task_id = attention_record.get('task_id', '')
        attn_data = attention_record.get('attention', {})
        retrieved_snippets = attention_record.get('retrieved_snippets', [])
        
        if not attn_data:
            return {}
        
        input_length = attn_data.get('input_length', 0)
        avg_attention = attn_data.get('avg_attention', [])
        
        if not avg_attention or not input_length:
            return {}
        
        # 转换为 numpy 数组
        try:
            attn_weights = np.array(avg_attention)
            if len(attn_weights.shape) > 1:
                # 如果是多维的，取最后一个 token 对所有输入的注意力
                attn_weights = attn_weights[-1, :]
        except:
            return {}
        
        # 分析注意力分布
        total_attn = np.sum(attn_weights)
        if total_attn == 0:
            return {}
        
        # 归一化
        attn_weights = attn_weights / total_attn
        
        # 估算不同区域的位置
        # 假设 context 结构：[检索到的片段...] [目标函数签名]
        num_snippets = len(retrieved_snippets)
        
        # 简单估算：每个检索片段平均占用的 token 数
        if num_snippets > 0:
            avg_snippet_tokens = (input_length * 0.8) // num_snippets
        else:
            avg_snippet_tokens = 0
        
        # 计算各区域的注意力
        analysis = {
            'task_id': task_id,
            'input_length': input_length,
            'num_retrieved_snippets': num_snippets,
            'attention_entropy': float(-np.sum(attn_weights * np.log(attn_weights + 1e-10))),
            'max_attention': float(np.max(attn_weights)),
            'attention_concentration': {}
        }
        
        # 分区分析
        if input_length > 100:
            # 前 1/4：可能是检索到的重要代码
            front_quarter = attn_weights[:input_length // 4]
            # 中间 1/2
            middle = attn_weights[input_length // 4: 3 * input_length // 4]
            # 后 1/4：接近目标函数的部分
            back_quarter = attn_weights[3 * input_length // 4:]
            
            analysis['attention_concentration'] = {
                'front_quarter': float(np.sum(front_quarter)),
                'middle_half': float(np.sum(middle)),
                'back_quarter': float(np.sum(back_quarter))
            }
        
        # 与 oracle 对比
        oracle = self.oracle_contexts.get(task_id, {})
        if oracle:
            # 检查检索到的内容是否包含 oracle 中提到的元素
            oracle_apis = str(oracle.get('oracle_apis', ''))
            oracle_classes = str(oracle.get('oracle_classes', ''))
            
            # 统计检索到的片段中包含 oracle 元素的比例
            relevant_snippets = 0
            for snippet in retrieved_snippets:
                snippet_name = snippet.get('name', '')
                if any(api in snippet_name for api in oracle_apis.split("'") if len(api) > 2):
                    relevant_snippets += 1
                if any(cls in snippet_name for cls in oracle_classes.split("'") if len(cls) > 2):
                    relevant_snippets += 1
            
            analysis['oracle_relevance'] = {
                'relevant_snippets': relevant_snippets,
                'total_snippets': num_snippets,
                'relevance_ratio': relevant_snippets / num_snippets if num_snippets > 0 else 0
            }
        
        return analysis
    
    def analyze_context_length(self, context_length: int) -> Dict:
        """分析指定上下文长度的注意力分布"""
        logger.info(f"分析 context_length={context_length}")
        
        attention_data = self.load_attention_data(context_length)
        rag_contexts = self.load_rag_context(context_length)
        
        if not attention_data:
            return {'context_length': context_length, 'error': 'No attention data'}
        
        analyses = []
        for attn_record in attention_data:
            task_id = attn_record.get('task_id', '')
            rag_ctx = rag_contexts.get(task_id, {})
            
            analysis = self.analyze_attention_distribution(attn_record, rag_ctx)
            if analysis:
                analyses.append(analysis)
        
        if not analyses:
            return {'context_length': context_length, 'error': 'No valid analyses'}
        
        # 汇总统计
        summary = {
            'context_length': context_length,
            'num_tasks': len(analyses),
            'avg_entropy': np.mean([a['attention_entropy'] for a in analyses]),
            'avg_max_attention': np.mean([a['max_attention'] for a in analyses]),
            'attention_distribution': {
                'avg_front_quarter': np.mean([
                    a['attention_concentration'].get('front_quarter', 0) 
                    for a in analyses if a.get('attention_concentration')
                ]),
                'avg_middle_half': np.mean([
                    a['attention_concentration'].get('middle_half', 0) 
                    for a in analyses if a.get('attention_concentration')
                ]),
                'avg_back_quarter': np.mean([
                    a['attention_concentration'].get('back_quarter', 0) 
                    for a in analyses if a.get('attention_concentration')
                ])
            },
            'oracle_relevance': {
                'avg_relevance_ratio': np.mean([
                    a.get('oracle_relevance', {}).get('relevance_ratio', 0)
                    for a in analyses
                ])
            },
            'individual_analyses': analyses
        }
        
        return summary
    
    def run_analysis(self) -> Dict[int, Dict]:
        """运行完整分析"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 加载 oracle 上下文
        self.load_oracle_contexts()
        
        all_results = {}
        
        for ctx_len in self.config.context_lengths:
            result = self.analyze_context_length(ctx_len)
            all_results[ctx_len] = result
            
            # 保存单个长度的分析结果
            output_file = os.path.join(
                self.config.output_dir,
                f"attention_analysis_{self.config.method}_{ctx_len}tokens.json"
            )
            
            # 创建一个不包含 individual_analyses 的版本用于保存
            result_to_save = {k: v for k, v in result.items() if k != 'individual_analyses'}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_to_save, f, indent=2, ensure_ascii=False)
        
        # 保存汇总
        summary = {
            'analyzed_at': datetime.now().isoformat(),
            'method': self.config.method,
            'results': {
                str(k): {key: v[key] for key in v if key != 'individual_analyses'}
                for k, v in all_results.items()
            }
        }
        
        summary_file = os.path.join(
            self.config.output_dir,
            f'attention_summary_{self.config.method}.json'
        )
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分析汇总已保存: {summary_file}")
        
        return all_results


def visualize_attention_analysis(results: Dict[int, Dict], output_dir: str, 
                                 method: str, dpi: int = 400):
    """
    可视化注意力分析结果
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 过滤有效结果
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        logger.warning("没有有效的分析结果可供可视化")
        return
    
    context_lengths = sorted(valid_results.keys())
    x_labels = [f"{l//1024}k" if l >= 1024 else str(l) for l in context_lengths]
    x_pos = np.arange(len(context_lengths))
    
    # ==================== 图1: 注意力熵随上下文长度变化 ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    entropies = [valid_results[l]['avg_entropy'] for l in context_lengths]
    
    ax.plot(x_pos, entropies, 'o-', color='#e74c3c', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2)
    
    for i, v in enumerate(entropies):
        ax.annotate(f'{v:.2f}', (x_pos[i], v), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    
    ax.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Entropy', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.title(f'Attention Entropy vs Context Length ({method.upper()})',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'attention_entropy_{method}.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # ==================== 图2: 注意力分布区域对比 ====================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    front_attn = [valid_results[l]['attention_distribution']['avg_front_quarter'] 
                  for l in context_lengths]
    middle_attn = [valid_results[l]['attention_distribution']['avg_middle_half'] 
                   for l in context_lengths]
    back_attn = [valid_results[l]['attention_distribution']['avg_back_quarter'] 
                 for l in context_lengths]
    
    width = 0.25
    ax.bar(x_pos - width, front_attn, width, label='Front Quarter (Retrieved)', color='#3498db', alpha=0.8)
    ax.bar(x_pos, middle_attn, width, label='Middle Half', color='#2ecc71', alpha=0.8)
    ax.bar(x_pos + width, back_attn, width, label='Back Quarter (Target)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Weight Proportion', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.title(f'Attention Distribution by Context Region ({method.upper()})',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'attention_distribution_{method}.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # ==================== 图3: Oracle 相关性 ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    relevance = [valid_results[l]['oracle_relevance']['avg_relevance_ratio'] * 100
                 for l in context_lengths]
    
    colors = plt.cm.RdYlGn(np.array(relevance) / max(relevance) if max(relevance) > 0 else np.zeros(len(relevance)))
    bars = ax.bar(x_pos, relevance, color=colors, width=0.6, edgecolor='black', linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Oracle Relevance Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(relevance) * 1.3 if max(relevance) > 0 else 10)
    
    plt.title(f'Retrieved Context Oracle Relevance ({method.upper()})',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'oracle_relevance_{method}.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # ==================== 图4: 综合对比图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 子图1: 注意力熵
    ax1 = axes[0, 0]
    ax1.plot(x_pos, entropies, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('Entropy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=8)
    ax1.set_title('Attention Entropy', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 子图2: 最大注意力
    ax2 = axes[0, 1]
    max_attn = [valid_results[l]['avg_max_attention'] for l in context_lengths]
    ax2.plot(x_pos, max_attn, 's-', color='#9b59b6', linewidth=2, markersize=8)
    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Max Attention')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.set_title('Max Attention Weight', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 子图3: 区域分布堆叠面积图
    ax3 = axes[1, 0]
    ax3.stackplot(x_pos, front_attn, middle_attn, back_attn,
                  labels=['Front', 'Middle', 'Back'],
                  colors=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    ax3.set_xlabel('Context Length')
    ax3.set_ylabel('Attention Proportion')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, fontsize=8)
    ax3.set_title('Attention Region Distribution', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    
    # 子图4: Oracle 相关性
    ax4 = axes[1, 1]
    ax4.bar(x_pos, relevance, color='#27ae60', width=0.6, alpha=0.8)
    ax4.set_xlabel('Context Length')
    ax4.set_ylabel('Relevance (%)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels, fontsize=8)
    ax4.set_title('Oracle Relevance', fontweight='bold')
    
    plt.suptitle(f'Attention Analysis Summary ({method.upper()})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'attention_summary_{method}.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"注意力分析图表已保存到: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='CoderEval 注意力分析'
    )
    
    parser.add_argument('--attention-dir', type=str, default='./attention_data',
                        help='注意力数据目录')
    parser.add_argument('--rag-dir', type=str, default='./rag_contexts',
                        help='RAG 上下文目录')
    parser.add_argument('--dataset', type=str,
                        default='CoderEval4Python.json',
                        help='原始数据集路径')
    parser.add_argument('--output', type=str, default='./attention_analysis_output',
                        help='输出目录')
    parser.add_argument('--method', type=str, choices=['bm25', 'jaccard'],
                        default='bm25', help='检索方法')
    parser.add_argument('--context-lengths', type=int, nargs='*',
                        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608],
                        help='要分析的上下文长度')
    parser.add_argument('--dpi', type=int, default=400,
                        help='图片分辨率')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='跳过分析，直接使用已有结果生成图表')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = AttentionAnalysisConfig(
        attention_dir=args.attention_dir,
        rag_dir=args.rag_dir,
        dataset_path=args.dataset,
        output_dir=args.output,
        method=args.method,
        context_lengths=args.context_lengths
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    if args.skip_analysis:
        # 从已有结果加载
        summary_file = os.path.join(config.output_dir, f'attention_summary_{config.method}.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            results = {int(k): v for k, v in summary['results'].items()}
        else:
            logger.error(f"找不到分析结果: {summary_file}")
            return
    else:
        analyzer = AttentionAnalyzer(config)
        results = analyzer.run_analysis()
    
    # 生成可视化
    visualize_attention_analysis(results, config.output_dir, config.method, args.dpi)


if __name__ == '__main__':
    main()

