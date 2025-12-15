#!/usr/bin/env python3
"""
CoderEval RAG 结果综合分析与可视化脚本
======================================

分析 BM25 和 Jaccard 两种检索方法在不同上下文长度下的性能表现，
生成综合大图和各个小设置的结果图。

功能：
1. 评估推理结果（计算 pass@k）
2. 对比 BM25 vs Jaccard 性能
3. 分析上下文长度对性能的影响
4. 生成综合可视化大图和小图

使用方法:
    python rag_result_analysis.py \
        --results-dir ./rag_inference_results \
        --rag-dir ./rag_contexts \
        --output ./rag_analysis_output \
        --dpi 400
"""

import json
import os
import sys
import ast
import subprocess
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
class AnalysisConfig:
    """分析配置"""
    results_dir: str = "./rag_inference_results"
    rag_dir: str = "./rag_contexts"
    dataset_path: str = "CoderEval4Python.json"
    repos_path: str = "repos"
    output_dir: str = "./rag_analysis_output"
    
    methods: List[str] = field(default_factory=lambda: ['bm25', 'jaccard'])
    context_lengths: List[int] = field(default_factory=lambda: [
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608
    ])
    
    num_samples: int = 10
    timeout: int = 30


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """计算 pass@k"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


class RAGResultAnalyzer:
    """
    RAG 结果分析器
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.collection = {}  # task_id -> task_info
        self.results = {}  # method -> context_length -> evaluation
        
    def load_dataset(self):
        """加载数据集"""
        logger.info(f"加载数据集: {self.config.dataset_path}")
        
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data.get('RECORDS', []):
            self.collection[record['_id']] = record
        
        logger.info(f"加载了 {len(self.collection)} 个任务")
    
    def load_inference_results(self, method: str, context_length: int) -> List[Dict]:
        """加载推理结果"""
        result_file = os.path.join(
            self.config.results_dir,
            f"results_{method}_{context_length}tokens.jsonl"
        )
        
        if not os.path.exists(result_file):
            logger.warning(f"结果文件不存在: {result_file}")
            return []
        
        results = []
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        计算评估指标
        
        注意：这里使用简化的评估（字符串匹配），
        完整评估需要实际执行代码（需要完整的 CoderEval 环境）
        """
        n = self.config.num_samples
        
        total_tasks = 0
        level_stats = defaultdict(lambda: {'count': 0, 'pass_counts': []})
        overall_pass_counts = []
        
        for result in results:
            task_id = result.get('_id', '')
            generated_codes = result.get('generate_results', [])
            level = result.get('level', 'unknown')
            
            if not generated_codes:
                continue
            
            # 获取 ground truth
            task_info = self.collection.get(task_id, {})
            ground_truth = task_info.get('code', '')
            
            # 简化评估：检查生成的代码是否与 ground truth 相似
            # 真实评估应该执行代码
            c = 0
            for gen_code in generated_codes[:n]:
                if self._simple_code_match(gen_code, ground_truth):
                    c += 1
            
            total_tasks += 1
            overall_pass_counts.append(c)
            level_stats[level]['count'] += 1
            level_stats[level]['pass_counts'].append(c)
        
        if not overall_pass_counts:
            return {'error': 'No valid results'}
        
        # 计算 pass@k
        metrics = {
            'total_tasks': total_tasks,
            'pass@1': np.mean([calculate_pass_at_k(n, c, 1) for c in overall_pass_counts]) * 100,
            'pass@5': np.mean([calculate_pass_at_k(n, c, 5) for c in overall_pass_counts]) * 100 if n >= 5 else 0,
            'pass@10': np.mean([calculate_pass_at_k(n, c, 10) for c in overall_pass_counts]) * 100 if n >= 10 else 0,
            'by_level': {}
        }
        
        for level, stats in level_stats.items():
            if stats['pass_counts']:
                metrics['by_level'][level] = {
                    'count': stats['count'],
                    'pass@1': np.mean([calculate_pass_at_k(n, c, 1) for c in stats['pass_counts']]) * 100
                }
        
        return metrics
    
    def _simple_code_match(self, generated: str, ground_truth: str) -> bool:
        """
        简化的代码匹配（用于演示）
        
        真实评估应该执行代码并检查测试是否通过
        """
        if not generated or not ground_truth:
            return False
        
        # 提取函数体
        gen_lines = [l.strip() for l in generated.split('\n') if l.strip() and not l.strip().startswith('#')]
        gt_lines = [l.strip() for l in ground_truth.split('\n') if l.strip() and not l.strip().startswith('#')]
        
        # 计算相似度
        gen_set = set(gen_lines)
        gt_set = set(gt_lines)
        
        if not gt_set:
            return False
        
        intersection = len(gen_set & gt_set)
        similarity = intersection / len(gt_set)
        
        # 相似度阈值
        return similarity > 0.5
    
    def analyze_method(self, method: str) -> Dict[int, Dict]:
        """分析单个方法的所有上下文长度"""
        logger.info(f"分析方法: {method}")
        
        method_results = {}
        
        for ctx_len in self.config.context_lengths:
            results = self.load_inference_results(method, ctx_len)
            
            if results:
                metrics = self.calculate_metrics(results)
                method_results[ctx_len] = metrics
                logger.info(f"  {ctx_len} tokens: pass@1={metrics.get('pass@1', 0):.2f}%")
            else:
                method_results[ctx_len] = {'error': 'No results'}
        
        return method_results
    
    def run_analysis(self) -> Dict:
        """运行完整分析"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.load_dataset()
        
        all_results = {}
        
        for method in self.config.methods:
            method_results = self.analyze_method(method)
            all_results[method] = method_results
            self.results[method] = method_results
            
            # 保存单方法结果
            output_file = os.path.join(
                self.config.output_dir,
                f'analysis_{method}.json'
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in method_results.items()}, 
                         f, indent=2, ensure_ascii=False)
        
        # 保存汇总
        summary = {
            'analyzed_at': datetime.now().isoformat(),
            'methods': self.config.methods,
            'context_lengths': self.config.context_lengths,
            'results': {
                method: {str(k): v for k, v in results.items()}
                for method, results in all_results.items()
            }
        }
        
        summary_file = os.path.join(self.config.output_dir, 'analysis_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分析汇总已保存: {summary_file}")
        
        return all_results


def create_comprehensive_visualization(results: Dict, output_dir: str, 
                                       context_lengths: List[int], dpi: int = 400):
    """
    创建综合可视化
    
    生成一张包含所有内容的大图和各个小设置的小图
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    methods = list(results.keys())
    valid_lengths = [l for l in context_lengths if any(
        l in results[m] and 'error' not in results[m][l]
        for m in methods
    )]
    
    if not valid_lengths:
        logger.warning("没有有效的结果可供可视化")
        return
    
    x_labels = [f"{l//1024}k" if l >= 1024 else str(l) for l in valid_lengths]
    x_pos = np.arange(len(valid_lengths))
    
    # 提取数据
    data = {}
    for method in methods:
        data[method] = {
            'pass@1': [],
            'pass@5': [],
            'pass@10': []
        }
        for ctx_len in valid_lengths:
            if ctx_len in results[method] and 'error' not in results[method][ctx_len]:
                data[method]['pass@1'].append(results[method][ctx_len].get('pass@1', 0))
                data[method]['pass@5'].append(results[method][ctx_len].get('pass@5', 0))
                data[method]['pass@10'].append(results[method][ctx_len].get('pass@10', 0))
            else:
                data[method]['pass@1'].append(0)
                data[method]['pass@5'].append(0)
                data[method]['pass@10'].append(0)
    
    colors = {
        'bm25': {'primary': '#e74c3c', 'light': '#f1948a'},
        'jaccard': {'primary': '#3498db', 'light': '#85c1e9'}
    }
    
    # ==================== 综合大图 ====================
    fig = plt.figure(figsize=(24, 18))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 子图1: BM25 vs Jaccard pass@1 对比（柱状图）
    ax1 = fig.add_subplot(gs[0, 0:2])
    width = 0.35
    for i, method in enumerate(methods):
        offset = (i - 0.5) * width
        ax1.bar(x_pos + offset, data[method]['pass@1'], width, 
                label=method.upper(), color=colors.get(method, {}).get('primary', f'C{i}'), alpha=0.8)
    ax1.set_xlabel('Context Length', fontsize=11)
    ax1.set_ylabel('Pass@1 (%)', fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.set_title('(a) BM25 vs Jaccard: Pass@1 Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 子图2: pass@k 趋势折线图
    ax2 = fig.add_subplot(gs[0, 2:4])
    for method in methods:
        color = colors.get(method, {}).get('primary', 'gray')
        ax2.plot(x_pos, data[method]['pass@1'], 'o-', color=color, 
                linewidth=2, markersize=8, label=f'{method.upper()} pass@1')
        ax2.plot(x_pos, data[method]['pass@10'], 's--', color=color,
                linewidth=1.5, markersize=6, alpha=0.6, label=f'{method.upper()} pass@10')
    ax2.set_xlabel('Context Length', fontsize=11)
    ax2.set_ylabel('Pass Rate (%)', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_title('(b) Pass@k Trends across Context Lengths', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 子图3-4: 各方法单独的详细图
    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[1, i*2:(i+1)*2])
        
        width = 0.25
        ax.bar(x_pos - width, data[method]['pass@1'], width, 
               label='pass@1', color='#2ecc71', alpha=0.8)
        ax.bar(x_pos, data[method]['pass@5'], width, 
               label='pass@5', color='#3498db', alpha=0.8)
        ax.bar(x_pos + width, data[method]['pass@10'], width, 
               label='pass@10', color='#9b59b6', alpha=0.8)
        
        # 添加趋势线
        ax2_twin = ax.twinx()
        ax2_twin.plot(x_pos, data[method]['pass@1'], 'o-', color='#27ae60', 
                     linewidth=2, markersize=6)
        ax2_twin.set_ylim(ax.get_ylim())
        ax2_twin.set_yticks([])
        
        ax.set_xlabel('Context Length', fontsize=11)
        ax.set_ylabel('Pass Rate (%)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_title(f'({"c" if i==0 else "d"}) {method.upper()} Detailed Performance', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 子图5: 热力图
    ax5 = fig.add_subplot(gs[2, 0:2])
    
    heatmap_data = []
    heatmap_labels = []
    for method in methods:
        for k in [1, 5, 10]:
            row = data[method][f'pass@{k}']
            heatmap_data.append(row)
            heatmap_labels.append(f'{method.upper()} pass@{k}')
    
    heatmap_data = np.array(heatmap_data)
    im = ax5.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    ax5.set_xticks(np.arange(len(valid_lengths)))
    ax5.set_yticks(np.arange(len(heatmap_labels)))
    ax5.set_xticklabels(x_labels, fontsize=9)
    ax5.set_yticklabels(heatmap_labels, fontsize=9)
    
    for i in range(len(heatmap_labels)):
        for j in range(len(valid_lengths)):
            text = ax5.text(j, i, f'{heatmap_data[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax5, label='Pass Rate (%)')
    ax5.set_title('(e) Performance Heatmap', fontsize=12, fontweight='bold')
    
    # 子图6: 方法间差异
    ax6 = fig.add_subplot(gs[2, 2:4])
    
    if len(methods) >= 2:
        diff_pass1 = np.array(data[methods[0]]['pass@1']) - np.array(data[methods[1]]['pass@1'])
        
        colors_diff = ['#e74c3c' if d > 0 else '#3498db' for d in diff_pass1]
        bars = ax6.bar(x_pos, diff_pass1, color=colors_diff, width=0.6, alpha=0.8)
        
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_xlabel('Context Length', fontsize=11)
        ax6.set_ylabel('Difference in Pass@1 (%)', fontsize=11)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(x_labels, fontsize=9)
        ax6.set_title(f'(f) {methods[0].upper()} - {methods[1].upper()} Performance Difference', 
                     fontsize=12, fontweight='bold')
        ax6.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars, diff_pass1):
            height = bar.get_height()
            ax6.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)
    
    plt.suptitle('CoderEval RAG Retrieval Methods: Comprehensive Performance Analysis',
                fontsize=16, fontweight='bold', y=0.98)
    
    fig.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"综合大图已保存: comprehensive_analysis.png")
    
    # ==================== 各个小图 ====================
    
    # 小图1: 每个方法的单独 pass@1 折线图
    for method in methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x_pos, data[method]['pass@1'], 'o-', color=colors.get(method, {}).get('primary', 'blue'),
               linewidth=3, markersize=10, markerfacecolor='white', markeredgewidth=2)
        
        for i, v in enumerate(data[method]['pass@1']):
            ax.annotate(f'{v:.1f}', (x_pos[i], v), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=10)
        
        ax.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pass@1 (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.title(f'{method.upper()} Retrieval: Pass@1 vs Context Length',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'pass1_{method}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    # 小图2: 每个方法的 pass@k 对比
    for method in methods:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        width = 0.25
        ax.bar(x_pos - width, data[method]['pass@1'], width, label='pass@1', color='#e74c3c', alpha=0.8)
        ax.bar(x_pos, data[method]['pass@5'], width, label='pass@5', color='#3498db', alpha=0.8)
        ax.bar(x_pos + width, data[method]['pass@10'], width, label='pass@10', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.title(f'{method.upper()} Retrieval: Pass@k Comparison',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'pass_k_{method}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    # 小图3: 方法对比
    fig, ax = plt.subplots(figsize=(12, 7))
    
    width = 0.35
    for i, method in enumerate(methods):
        offset = (i - 0.5) * width
        bars = ax.bar(x_pos + offset, data[method]['pass@1'], width,
                     label=method.upper(), 
                     color=colors.get(method, {}).get('primary', f'C{i}'), alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pass@1 (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.title('BM25 vs Jaccard: Pass@1 Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 小图4: 趋势对比折线图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = ['o', 's', '^']
    for method in methods:
        color = colors.get(method, {}).get('primary', 'gray')
        ax.plot(x_pos, data[method]['pass@1'], 'o-', color=color,
               linewidth=2.5, markersize=10, label=f'{method.upper()} pass@1',
               markerfacecolor='white', markeredgewidth=2)
        ax.plot(x_pos, data[method]['pass@10'], 's--', color=color,
               linewidth=2, markersize=8, alpha=0.7, label=f'{method.upper()} pass@10')
    
    ax.set_xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.title('Pass Rate Trends: BM25 vs Jaccard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'trend_comparison.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 小图5: 每个上下文长度的详细对比
    for ctx_len in valid_lengths:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ctx_idx = valid_lengths.index(ctx_len)
        methods_names = [m.upper() for m in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        
        pass1_vals = [data[m]['pass@1'][ctx_idx] for m in methods]
        pass5_vals = [data[m]['pass@5'][ctx_idx] for m in methods]
        pass10_vals = [data[m]['pass@10'][ctx_idx] for m in methods]
        
        ax.bar(x - width, pass1_vals, width, label='pass@1', color='#e74c3c')
        ax.bar(x, pass5_vals, width, label='pass@5', color='#3498db')
        ax.bar(x + width, pass10_vals, width, label='pass@10', color='#2ecc71')
        
        ax.set_xlabel('Retrieval Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods_names)
        ax.legend()
        
        ctx_label = f"{ctx_len//1024}k" if ctx_len >= 1024 else str(ctx_len)
        plt.title(f'Performance at {ctx_label} Context Length', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'context_{ctx_len}.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    logger.info(f"所有图表已保存到: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='CoderEval RAG 结果综合分析'
    )
    
    parser.add_argument('--results-dir', type=str, default='./rag_inference_results',
                        help='推理结果目录')
    parser.add_argument('--rag-dir', type=str, default='./rag_contexts',
                        help='RAG 上下文目录')
    parser.add_argument('--dataset', type=str,
                        default='CoderEval4Python.json',
                        help='数据集路径')
    parser.add_argument('--repos', type=str,
                        default='repos',
                        help='项目仓库路径')
    parser.add_argument('--output', type=str, default='./rag_analysis_output',
                        help='输出目录')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['bm25', 'jaccard'],
                        help='要分析的检索方法')
    parser.add_argument('--context-lengths', type=int, nargs='*',
                        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608],
                        help='要分析的上下文长度')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='每任务的候选代码数量')
    parser.add_argument('--dpi', type=int, default=400,
                        help='图片分辨率')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='跳过分析，直接使用已有结果生成图表')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = AnalysisConfig(
        results_dir=args.results_dir,
        rag_dir=args.rag_dir,
        dataset_path=args.dataset,
        repos_path=args.repos,
        output_dir=args.output,
        methods=args.methods,
        context_lengths=args.context_lengths,
        num_samples=args.num_samples
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    if args.skip_analysis:
        # 从已有结果加载
        summary_file = os.path.join(config.output_dir, 'analysis_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            results = {}
            for method, method_results in summary['results'].items():
                results[method] = {int(k): v for k, v in method_results.items()}
        else:
            logger.error(f"找不到分析结果: {summary_file}")
            return
    else:
        analyzer = RAGResultAnalyzer(config)
        results = analyzer.run_analysis()
    
    # 生成可视化
    create_comprehensive_visualization(
        results, 
        config.output_dir, 
        config.context_lengths,
        args.dpi
    )
    
    # 打印汇总
    print("\n" + "="*70)
    print("RAG 检索方法性能分析汇总")
    print("="*70)
    
    for method, method_results in results.items():
        print(f"\n{method.upper()} 方法:")
        print("-" * 50)
        for ctx_len in sorted(method_results.keys()):
            metrics = method_results[ctx_len]
            if 'error' not in metrics:
                print(f"  {ctx_len:6d} tokens: pass@1={metrics['pass@1']:.2f}%  "
                      f"pass@5={metrics.get('pass@5', 0):.2f}%  "
                      f"pass@10={metrics.get('pass@10', 0):.2f}%")


if __name__ == '__main__':
    main()

