#!/usr/bin/env python3
"""
CoderEval 稀疏检索上下文生成脚本 (V2 - 防数据泄露版)
======================================================

使用 BM25 和 Jaccard 相似度从项目代码库中检索相关上下文。

防数据泄露措施：
1. 排除所有 *passk_validte*.py 测试验证文件
2. 排除目标文件（包含待补全函数的文件）
3. 精确匹配排除目标函数
4. 排除包含 ground truth 代码的片段

使用方法:
    python sparse_retrieval_context_v2.py \
        --method bm25 \
        --output ./rag_contexts_bm25 \
        --context-lengths 1024 2048 4096 8192 16384 32768 65536 131072 196608

    python sparse_retrieval_context_v2.py \
        --method jaccard \
        --output ./rag_contexts_jaccard \
        --context-lengths 1024 2048 4096 8192 16384 32768 65536 131072 196608
"""

import json
import os
import sys
import ast
import re
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import hashlib
import ast
import heapq

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """检索配置"""
    dataset_path: str = "home/travis/builds/CoderEval4Python.json"
    repos_path: str = "home/travis/builds/repos"
    output_dir: str = "./rag_contexts"
    
    # 检索方法: bm25 或 jaccard
    method: str = "bm25"
    
    # 目标上下文长度 (tokens)
    context_lengths: List[int] = field(default_factory=lambda: [
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608
    ])
    
    # BM25 参数
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # 代码片段提取参数
    min_snippet_lines: int = 3
    max_snippet_lines: int = 100


# ==================== 数据泄露过滤器 ====================

class LeakageFilter:
    """
    数据泄露过滤器
    
    确保 RAG 检索不会泄露测试数据或目标函数
    """
    
    # 需要排除的文件模式
    EXCLUDED_FILE_PATTERNS = [
        r'.*passk_validte.*\.py$',      # passk_validte 测试文件
        r'.*_passk_validate.*\.py$',    # _passk_validate 测试文件
        r'.*passk_validate.*\.py$',     # passk_validate 测试文件
        r'.*_test\.py$',                # 测试文件（可选）
        r'.*test_.*\.py$',              # 测试文件（可选）
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.EXCLUDED_FILE_PATTERNS]
    
    def should_exclude_file(self, file_path: str) -> bool:
        """检查文件是否应该被排除"""
        file_name = os.path.basename(file_path)
        
        for pattern in self.compiled_patterns:
            if pattern.match(file_name):
                return True
        
        # 检查路径中是否包含 standalone 目录（这里存放测试文件）
        if '/standalone/' in file_path or '\\standalone\\' in file_path:
            return True
        
        return False
    
    def should_exclude_snippet(self, snippet: 'CodeSnippet', 
                                target_func_name: str,
                                target_file_path: str,
                                ground_truth_code: str = "") -> bool:
        """
        检查代码片段是否应该被排除
        
        Args:
            snippet: 代码片段
            target_func_name: 目标函数名
            target_file_path: 目标文件路径
            ground_truth_code: 标准答案代码
        """
        # 1. 排除目标文件中的所有片段
        if self._is_same_file(snippet.file_path, target_file_path):
            return True
        
        # 2. 精确匹配排除目标函数
        if snippet.snippet_type in ['function', 'method']:
            # 提取函数名（处理类方法情况）
            snippet_func_name = snippet.name.split('.')[-1] if '.' in snippet.name else snippet.name
            if snippet_func_name == target_func_name:
                return True
            # 也排除以 _ 开头的同名函数（某些测试会这样命名）
            if snippet_func_name == f"_{target_func_name}":
                return True
        
        # 3. 检查是否包含 ground truth 的核心逻辑
        if ground_truth_code and self._contains_ground_truth(snippet.content, ground_truth_code):
            return True
        
        return False
    
    def _is_same_file(self, path1: str, path2: str) -> bool:
        """检查两个路径是否指向同一文件"""
        # 标准化路径
        norm1 = os.path.normpath(path1).replace('\\', '/')
        norm2 = os.path.normpath(path2).replace('\\', '/')
        
        # 直接比较
        if norm1 == norm2:
            return True
        
        # 比较文件名和部分路径
        parts1 = norm1.split('/')
        parts2 = norm2.split('/')
        
        # 比较最后3级路径
        if len(parts1) >= 3 and len(parts2) >= 3:
            if parts1[-3:] == parts2[-3:]:
                return True
        
        # 比较文件名
        if parts1[-1] == parts2[-1]:
            # 文件名相同，检查是否在同一项目中
            return True
        
        return False
    
    def _contains_ground_truth(self, snippet_content: str, ground_truth: str) -> bool:
        """
        检查片段是否包含 ground truth 的核心逻辑
        
        使用更宽松的匹配，避免漏掉变体
        """
        # 提取 ground truth 的核心行（去除空行、注释、docstring）
        gt_lines = self._extract_core_lines(ground_truth)
        snippet_lines = self._extract_core_lines(snippet_content)
        
        if not gt_lines:
            return False
        
        # 如果超过 50% 的核心行匹配，认为是泄露
        matched = 0
        for gt_line in gt_lines:
            if gt_line in snippet_lines:
                matched += 1
        
        match_ratio = matched / len(gt_lines) if gt_lines else 0
        return match_ratio > 0.5
    
    def _extract_core_lines(self, code: str) -> Set[str]:
        """提取代码的核心行（用于比较）"""
        lines = code.split('\n')
        core_lines = set()
        
        in_docstring = False
        docstring_char = None
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行
            if not stripped:
                continue
            
            # 处理 docstring
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2:
                        continue  # 单行 docstring
                    in_docstring = True
                    continue
            else:
                if docstring_char in stripped:
                    in_docstring = False
                continue
            
            # 跳过注释
            if stripped.startswith('#'):
                continue
            
            # 跳过函数定义行
            if stripped.startswith('def ') or stripped.startswith('async def '):
                continue
            
            # 跳过简单的 pass、return None 等
            if stripped in ['pass', 'return', 'return None', '...']:
                continue
            
            # 标准化后添加
            normalized = re.sub(r'\s+', ' ', stripped)
            if len(normalized) > 5:  # 只保留有意义的行
                core_lines.add(normalized)
        
        return core_lines


# ==================== 代码片段提取 ====================

class CodeSnippet:
    """代码片段"""
    def __init__(self, content: str, file_path: str, snippet_type: str, 
                 name: str = "", start_line: int = 0, end_line: int = 0):
        self.content = content
        self.file_path = file_path
        self.snippet_type = snippet_type  # function, class, module, import
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.tokens = self._tokenize(content)
        self.token_count = self._estimate_tokens(content)
        
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        text = text.lower()
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = text.replace('_', ' ')
        tokens = re.findall(r'\b[a-z]+\b|\b\d+\b', text)
        return tokens
    
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量"""
        return len(text) // 3
    
    def __hash__(self):
        return hash(self.content)
    
    def __eq__(self, other):
        return self.content == other.content


class CodeExtractor:
    """
    代码提取器
    从 Python 文件中提取代码片段（带数据泄露过滤）
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.leakage_filter = LeakageFilter()
        self.snippets_cache = {}  # project -> List[CodeSnippet]
        
    def extract_snippets_from_file(self, file_path: str) -> List[CodeSnippet]:
        """从单个文件提取代码片段"""
        snippets = []
        
        # 首先检查文件是否应该被排除
        if self.leakage_filter.should_exclude_file(file_path):
            logger.debug(f"排除文件（防泄露）: {file_path}")
            return snippets
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return snippets
            
            lines = content.split('\n')
            
            # 尝试解析 AST
            try:
                tree = ast.parse(content)
                
                # 提取 import 语句
                import_lines = []
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', start + 1)
                        import_lines.extend(range(start, end))
                
                if import_lines:
                    import_content = '\n'.join(lines[i] for i in sorted(set(import_lines)) if i < len(lines))
                    if import_content.strip():
                        snippets.append(CodeSnippet(
                            content=import_content,
                            file_path=file_path,
                            snippet_type='import',
                            name='imports',
                            start_line=min(import_lines) + 1,
                            end_line=max(import_lines) + 1
                        ))
                
                # 提取函数和类
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', start + 1)
                        func_content = '\n'.join(lines[start:end])
                        
                        if self.config.min_snippet_lines <= (end - start) <= self.config.max_snippet_lines:
                            snippets.append(CodeSnippet(
                                content=func_content,
                                file_path=file_path,
                                snippet_type='function',
                                name=node.name,
                                start_line=start + 1,
                                end_line=end
                            ))
                    
                    elif isinstance(node, ast.ClassDef):
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', start + 1)
                        class_content = '\n'.join(lines[start:end])
                        
                        if (end - start) <= self.config.max_snippet_lines:
                            snippets.append(CodeSnippet(
                                content=class_content,
                                file_path=file_path,
                                snippet_type='class',
                                name=node.name,
                                start_line=start + 1,
                                end_line=end
                            ))
                        else:
                            # 类太大，提取类头和方法签名
                            class_header = []
                            for i, child in enumerate(node.body):
                                if i == 0:
                                    header_end = getattr(child, 'end_lineno', start + 5)
                                    class_header = lines[start:min(header_end, start + 10)]
                                
                                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    method_start = child.lineno - 1
                                    method_end = getattr(child, 'end_lineno', method_start + 1)
                                    method_content = '\n'.join(lines[method_start:method_end])
                                    
                                    if self.config.min_snippet_lines <= (method_end - method_start) <= self.config.max_snippet_lines:
                                        snippets.append(CodeSnippet(
                                            content=method_content,
                                            file_path=file_path,
                                            snippet_type='method',
                                            name=f"{node.name}.{child.name}",
                                            start_line=method_start + 1,
                                            end_line=method_end
                                        ))
                            
                            if class_header:
                                snippets.append(CodeSnippet(
                                    content='\n'.join(class_header) + '\n    ...',
                                    file_path=file_path,
                                    snippet_type='class_header',
                                    name=node.name,
                                    start_line=start + 1,
                                    end_line=start + len(class_header)
                                ))
                
            except SyntaxError:
                # AST 解析失败，按行分块
                chunk_size = 30
                for i in range(0, len(lines), chunk_size):
                    chunk = '\n'.join(lines[i:i+chunk_size])
                    if chunk.strip():
                        snippets.append(CodeSnippet(
                            content=chunk,
                            file_path=file_path,
                            snippet_type='chunk',
                            name=f"chunk_{i}",
                            start_line=i + 1,
                            end_line=min(i + chunk_size, len(lines))
                        ))
                        
        except Exception as e:
            logger.debug(f"提取文件失败 {file_path}: {e}")
        
        return snippets
    
    def extract_project_snippets(self, project_dir: str, 
                                  exclude_file: str = None) -> List[CodeSnippet]:
        """
        提取整个项目的代码片段
        
        Args:
            project_dir: 项目目录
            exclude_file: 要排除的文件路径（目标文件）
        """
        cache_key = f"{project_dir}:{exclude_file or ''}"
        if cache_key in self.snippets_cache:
            return self.snippets_cache[cache_key]
        
        snippets = []
        excluded_count = 0
        
        for root, dirs, files in os.walk(project_dir):
            # 跳过常见的非代码目录
            dirs[:] = [d for d in dirs if d not in [
                '__pycache__', '.git', 'node_modules', 'venv', 'env',
                '.tox', 'build', 'dist', 'egg-info', '.eggs',
                'standalone'  # 排除 standalone 目录
            ]]
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                
                file_path = os.path.join(root, file)
                
                # 检查是否应排除此文件
                if self.leakage_filter.should_exclude_file(file_path):
                    excluded_count += 1
                    continue
                
                # 排除目标文件
                if exclude_file and self._is_target_file(file_path, exclude_file):
                    excluded_count += 1
                    continue
                
                file_snippets = self.extract_snippets_from_file(file_path)
                snippets.extend(file_snippets)
        
        logger.info(f"从 {project_dir} 提取了 {len(snippets)} 个片段，排除了 {excluded_count} 个文件")
        
        self.snippets_cache[cache_key] = snippets
        return snippets
    
    def _is_target_file(self, file_path: str, target_file: str) -> bool:
        """检查是否是目标文件"""
        # 标准化路径
        file_path = os.path.normpath(file_path).replace('\\', '/')
        target_file = os.path.normpath(target_file).replace('\\', '/')
        
        # 直接比较
        if file_path.endswith(target_file) or target_file.endswith(file_path):
            return True
        
        # 比较文件名
        if os.path.basename(file_path) == os.path.basename(target_file):
            # 检查是否在相似路径中
            fp_parts = file_path.split('/')
            tf_parts = target_file.split('/')
            
            # 比较最后几级目录
            common = 0
            for i in range(1, min(len(fp_parts), len(tf_parts)) + 1):
                if fp_parts[-i] == tf_parts[-i]:
                    common += 1
                else:
                    break
            
            return common >= 2  # 至少2级目录相同
        
        return False
    
    def clear_cache(self):
        """清除缓存"""
        self.snippets_cache.clear()


# ==================== 检索算法 ====================

class BM25Retriever:
    """BM25 检索器"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.avg_doc_len = 0
        self.corpus_size = 0
        self.snippets = []
        self.indexed = False
        
    def index(self, snippets: List[CodeSnippet]):
        """建立索引"""
        self.snippets = snippets
        self.corpus_size = len(snippets)
        
        if self.corpus_size == 0:
            self.indexed = True
            return
        
        self.doc_freqs = Counter()
        self.doc_lens = []
        
        for snippet in snippets:
            tokens = snippet.tokens
            self.doc_lens.append(len(tokens))
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0
        self.indexed = True
    
    def _idf(self, term: str) -> float:
        """计算 IDF"""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
    
    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算 BM25 分数"""
        snippet = self.snippets[doc_idx]
        doc_tokens = snippet.tokens
        doc_len = self.doc_lens[doc_idx]
        
        tf = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            
            term_freq = tf[term]
            idf = self._idf(term)
            
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            
            score += idf * numerator / denominator
        
        return score
    
    def retrieve(self, query: str, top_k: Optional[int] = 100) -> List[Tuple[CodeSnippet, float]]:
        """检索最相关的代码片段"""
        if not self.indexed or self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx in range(self.corpus_size):
            score = self._score(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))

        if not scores:
            return []

        # top_k=None 或 top_k<=0 => 返回全部（仍需排序）
        if top_k is None or top_k <= 0 or top_k >= len(scores):
            scores.sort(key=lambda x: x[1], reverse=True)
            return [(self.snippets[idx], score) for idx, score in scores]

        # top_k 较小：只取最大的 top_k，避免全量 sort
        top = heapq.nlargest(top_k, scores, key=lambda x: x[1])
        top.sort(key=lambda x: x[1], reverse=True)
        return [(self.snippets[idx], score) for idx, score in top]
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        text = text.lower()
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = text.replace('_', ' ')
        tokens = re.findall(r'\b[a-z]+\b|\b\d+\b', text)
        return tokens


class JaccardRetriever:
    """Jaccard 相似度检索器"""
    
    def __init__(self):
        self.snippets = []
        self.snippet_token_sets = []
        self.indexed = False
    
    def index(self, snippets: List[CodeSnippet]):
        """建立索引"""
        self.snippets = snippets
        self.snippet_token_sets = [set(s.tokens) for s in snippets]
        self.indexed = True
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """计算 Jaccard 相似度"""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def retrieve(self, query: str, top_k: Optional[int] = 100) -> List[Tuple[CodeSnippet, float]]:
        """检索最相关的代码片段（Jaccard）"""
        if not self.indexed or not self.snippets:
            return []

        query_tokens = self._tokenize(query)
        query_set = set(query_tokens)
        if not query_set:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, token_set in enumerate(self.snippet_token_sets):
            similarity = self._jaccard_similarity(query_set, token_set)
            if similarity > 0:
                scores.append((idx, similarity))

        if not scores:
            return []

        # top_k=None 或 top_k<=0 => 返回全部（仍需排序）
        if top_k is None or top_k <= 0 or top_k >= len(scores):
            scores.sort(key=lambda x: x[1], reverse=True)
            return [(self.snippets[idx], score) for idx, score in scores]

        # top_k 较小：只取最大的 top_k，避免全量 sort
        top = heapq.nlargest(top_k, scores, key=lambda x: x[1])
        top.sort(key=lambda x: x[1], reverse=True)
        return [(self.snippets[idx], score) for idx, score in top]
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        text = text.lower()
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = text.replace('_', ' ')
        tokens = re.findall(r'\b[a-z]+\b|\b\d+\b', text)
        return tokens


# ==================== 上下文构建器 ====================

class SparseRetrievalContextBuilder:
    """
    稀疏检索上下文构建器（防数据泄露版）
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.extractor = CodeExtractor(config)
        self.leakage_filter = LeakageFilter()
        
        if config.method == 'bm25':
            self.retriever = BM25Retriever(config.bm25_k1, config.bm25_b)
        else:
            self.retriever = JaccardRetriever()
        
        self.current_project = None
        self.current_exclude_file = None
        
    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量"""
        return len(text) // 3
    
    def _build_query(self, task: Dict) -> str:
        """构建检索查询"""
        name = task.get('name', '')
        docstring = task.get('docstring', '')
        file_path = task.get('file_path', '')
        
        module_parts = file_path.replace('.py', '').replace('/', '.').split('.')
        module_context = ' '.join(module_parts[-3:])
        
        query = f"{name} {docstring} {module_context}"
        return query
    
    def _get_function_signature(self, task: Dict) -> str:
        """获取函数签名"""
        code = task.get('code', '')
        name = task.get('name', '')
        docstring = task.get('docstring', '')
        
        lines = code.strip().split('\n')
        signature_lines = []
        
        for line in lines:
            signature_lines.append(line)
            if line.strip().endswith(':'):
                break
        
        signature = '\n'.join(signature_lines)
        
        if docstring:
            signature += f'\n    """\n    {docstring}\n    """'
        
        return signature


    # ---- useful 标注：oracle + ground_truth（两者 OR）----

    _PY_KEYWORDS = {
        "False","None","True","and","as","assert","async","await","break","class","continue",
        "def","del","elif","else","except","finally","for","from","global","if","import","in",
        "is","lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield",
    }
    _COMMON_BUILTINS = {
        "abs","all","any","ascii","bin","bool","bytearray","bytes","callable","chr","classmethod",
        "compile","complex","delattr","dict","dir","divmod","enumerate","eval","exec","filter",
        "float","format","frozenset","getattr","globals","hasattr","hash","help","hex","id",
        "input","int","isinstance","issubclass","iter","len","list","locals","map","max",
        "memoryview","min","next","object","oct","open","ord","pow","print","property","range",
        "repr","reversed","round","set","setattr","slice","sorted","staticmethod","str","sum",
        "super","tuple","type","vars","zip",
    }

    def _safe_json_loads(self, s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            return None

    def _maybe_parse_listlike(self, v: Any) -> List[str]:
        """
        oracle_context 里经常出现：
        - 真 list
        - 字符串形式的 python list: "['a','b']"
        - 空格分隔字符串: "typing errno os"
        """
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            txt = v.strip()
            if not txt:
                return []
            # python list 形式
            if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("(") and txt.endswith(")")):
                try:
                    parsed = ast.literal_eval(txt)
                    if isinstance(parsed, (list, tuple, set)):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            # 否则按空白切分
            return [t for t in re.split(r"\s+", txt) if t]
        return [str(v)]

    def _extract_oracle_symbols(self, task: Dict) -> Set[str]:
        """
        支持两类 oracle 格式：
        1) {"apis": "...", "classes": "...", "vars": "..."}
        2) {"import": "...", "file": "...", "class": "..."}  (你贴的 all_context 长这样，但有时也有人把它当 oracle 用)
        我们只要能抽出 token 列表即可。
        """
        symbols: Set[str] = set()

        oracle_raw = task.get("oracle_context", None)
        if not oracle_raw:
            return symbols

        oracle_obj = oracle_raw
        if isinstance(oracle_raw, str):
            oracle_obj = self._safe_json_loads(oracle_raw)
            if oracle_obj is None:
                return symbols

        if not isinstance(oracle_obj, dict):
            return symbols

        # 首选 apis/classes/vars
        for k in ("apis", "classes", "vars"):
            if k in oracle_obj:
                for t in self._maybe_parse_listlike(oracle_obj.get(k)):
                    tt = t.strip().strip("'").strip('"')
                    if tt:
                        symbols.add(tt)

        # 兼容另一类字段命名
        for k in ("import", "file", "class"):
            if k in oracle_obj:
                for t in self._maybe_parse_listlike(oracle_obj.get(k)):
                    tt = t.strip().strip("'").strip('"')
                    if tt:
                        symbols.add(tt)

        # 清理：去掉明显无意义/过短/关键字
        cleaned = set()
        for s in symbols:
            if len(s) < 2:
                continue
            if s in self._PY_KEYWORDS:
                continue
            cleaned.add(s)
        return cleaned

    def _get_ground_truth_code(self, task: Dict) -> str:
        # 你数据里有的叫 code，有的叫 ground_truth，这里自动兼容
        gt = task.get("ground_truth", "") or task.get("code", "") or ""
        return gt

    def _extract_gt_symbols(self, task: Dict) -> Set[str]:
        """
        从 ground_truth 抽符号：
        - import/from import 的名字
        - 调用：foo(...), obj.bar(...)
        - Name / Attribute
        优先 AST，AST 失败则正则 fallback
        """
        gt = self._get_ground_truth_code(task)
        if not gt.strip():
            return set()

        symbols: Set[str] = set()

        # AST 优先
        try:
            tree = ast.parse(gt)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name:
                            symbols.add(alias.name.split(".")[-1])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name:
                            symbols.add(alias.name)
                elif isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Name):
                        symbols.add(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        symbols.add(fn.attr)
                elif isinstance(node, ast.Attribute):
                    symbols.add(node.attr)
                elif isinstance(node, ast.Name):
                    symbols.add(node.id)
        except Exception:
            # Regex fallback：抓调用与属性
            # foo( , obj.bar( , .attr
            call_names = re.findall(r"\b([A-Za-z_]\w*)\s*\(", gt)
            attr_names = re.findall(r"\.([A-Za-z_]\w*)\b", gt)
            name_tokens = re.findall(r"\b([A-Za-z_]\w*)\b", gt)
            for t in call_names + attr_names + name_tokens:
                symbols.add(t)

        # 过滤噪声
        cleaned = set()
        for s in symbols:
            if len(s) < 2:
                continue
            if s in self._PY_KEYWORDS:
                continue
            # builtins 不必全丢，但很多任务会把这些当作“关键 API”（例如 divmod/map）
            # 这里我不丢 builtins，只是保留，后面命中会更宽松
            cleaned.add(s)
        return cleaned

    def _find_symbol_hits(self, text: str, symbols: Set[str]) -> List[str]:
        """
        在 snippet.content 里做“标识符边界”匹配，避免 Time 命中 Timestamp。
        """
        if not text or not symbols:
            return []
        hits = []
        for sym in symbols:
            # 对类似 "os.path" 的符号，既匹配整体也匹配最后段
            candidates = {sym}
            if "." in sym:
                candidates.add(sym.split(".")[-1])

            for c in candidates:
                if not c or c in self._PY_KEYWORDS:
                    continue
                # 标识符边界：前后不是字母数字下划线
                pat = rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])"
                if re.search(pat, text):
                    hits.append(sym)
                    break
        # 去重并稳定排序
        return sorted(set(hits)
    def build_context(self, task: Dict, target_tokens: int) -> Tuple[str, Dict]:
        project = task.get('project', '')
        file_path = task.get('file_path', '')
        target_func = task.get('name', '')

        # 关键：统一 ground_truth 入口（兼容不同数据字段命名）
        ground_truth = task.get('ground_truth', '') or task.get('code', '') or ''

        project_dir = os.path.join(
            self.config.repos_path,
            project.replace('/', '---')
        )

        if self.current_project != project or self.current_exclude_file != file_path:
            if os.path.exists(project_dir):
                snippets = self.extractor.extract_project_snippets(
                    project_dir,
                    exclude_file=file_path
                )

                filtered_snippets = []
                for s in snippets:
                    if self.leakage_filter.should_exclude_snippet(
                        s, target_func, file_path, ground_truth
                    ):
                        continue
                    filtered_snippets.append(s)

                logger.debug(f"过滤后剩余 {len(filtered_snippets)} 个片段（原 {len(snippets)}）")

                self.retriever.index(filtered_snippets)
                self.current_project = project
                self.current_exclude_file = file_path
            else:
                logger.warning(f"项目目录不存在: {project_dir}")
                self.retriever.index([])
                self.current_project = project
                self.current_exclude_file = file_path

        query = self._build_query(task)
        retrieved = self.retriever.retrieve(query, top_k=500)

        # ===== context scale stats (no index_size needed) =====
        filtered_pool_size = len(filtered_snippets) if "filtered_snippets" in locals() else None

        # retriever 内部索引规模（兼容 BM25/Jaccard）
        retriever_index_size = getattr(self.retriever, "corpus_size", None)
        if retriever_index_size is None and hasattr(self.retriever, "snippets"):
            retriever_index_size = len(self.retriever.snippets)

        retrieved_candidates = len(retrieved)

        cand_token_counts = []
        for snip, _score in retrieved:
            tc = getattr(snip, "token_count", None)
            if tc is not None:
                cand_token_counts.append(int(tc))

        avg_cand_tokens = (sum(cand_token_counts) / len(cand_token_counts)) if cand_token_counts else None
        max_fillable_tokens_est = (retrieved_candidates * avg_cand_tokens) if avg_cand_tokens is not None else None

        logger.info(
            f"[ctx-stats] pool={filtered_pool_size} index={retriever_index_size} "
            f"retrieved={retrieved_candidates} avail={available_tokens} "
            f"avgCandTok={avg_cand_tokens} maxFillEst={max_fillable_tokens_est}"
        )
        

        func_signature = self._get_function_signature(task)
        signature_tokens = self._estimate_tokens(func_signature)

        prompt_overhead = 200
        # 关键：防止出现负数
        available_tokens = max(0, target_tokens - signature_tokens - prompt_overhead)
        budget_insufficient = (available_tokens == 0)

        # ---------------------------
        # 构造“可对齐”的分段上下文
        # ---------------------------
        retrieved_context = ""
        current_tokens = 0

        retrieved_info = []          # 全量（用于后续分析）
        snippet_char_spans = []      # 每个 snippet 在 full_context 里的字符 span（后面会加偏移）

        def _format_snip_block(idx: int, snippet, score: float, content: str, truncated: bool) -> str:
            meta = {
                "idx": idx,
                "file": snippet.file_path,
                "type": snippet.snippet_type,
                "name": snippet.name,
                "score": float(score),
                "truncated": bool(truncated),
            }
            header = f"<<SNIP_META {json.dumps(meta, ensure_ascii=False)}>>\n"
            footer = f"\n<<END_SNIP_{idx}>>"

            body = f"# From: {snippet.file_path}\n{content}"
            if truncated:
                body += "\n# ... (truncated)"
            return header + body + footer

        snip_idx = 0

        # useful 标注所需符号集合（你已经实现了这两个 helper）
        oracle_syms = self._extract_oracle_symbols(task)
        gt_syms = self._extract_gt_symbols(task)

        for snippet, score in retrieved:
            if self.leakage_filter.should_exclude_snippet(
                snippet, target_func, file_path, ground_truth
            ):
                continue

            snippet_tokens = snippet.token_count

            # 判断是否需要截断
            if current_tokens + snippet_tokens > available_tokens:
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 50:
                    truncated_content = snippet.content[:remaining_tokens * 3]
                    if truncated_content.strip():
                        block = _format_snip_block(
                            snip_idx, snippet, score, truncated_content, truncated=True
                        )

                        if retrieved_context:
                            retrieved_context += "\n\n"
                        start = len(retrieved_context)
                        retrieved_context += block
                        end = len(retrieved_context)

                        # 截断内容上做命中（与实际喂给模型一致）
                        oracle_hits = self._find_symbol_hits(truncated_content, oracle_syms)
                        gt_hits = self._find_symbol_hits(truncated_content, gt_syms)
                        hit_oracle = len(oracle_hits) > 0
                        hit_gt = len(gt_hits) > 0
                        useful_or = hit_oracle or hit_gt

                        retrieved_info.append({
                            'idx': snip_idx,
                            'file': snippet.file_path,
                            'type': snippet.snippet_type,
                            'name': snippet.name,
                            'score': score,
                            'truncated': True,
                            'hit_oracle': hit_oracle,
                            'hit_gt': hit_gt,
                            'useful_or': useful_or,
                            'oracle_hits': oracle_hits,
                            'gt_hits': gt_hits,
                            'oracle_hit_count': len(oracle_hits),
                            'gt_hit_count': len(gt_hits),
                        })
                        snippet_char_spans.append({
                            'idx': snip_idx,
                            'start_char_in_retrieved_context': start,
                            'end_char_in_retrieved_context': end
                        })
                        snip_idx += 1
                break

            # 不截断：完整加入
            block = _format_snip_block(
                snip_idx, snippet, score, snippet.content, truncated=False
            )

            if retrieved_context:
                retrieved_context += "\n\n"
            start = len(retrieved_context)
            retrieved_context += block
            end = len(retrieved_context)

            current_tokens += snippet_tokens

            # 关键：不截断分支也打 useful 标注（字段结构一致）
            oracle_hits = self._find_symbol_hits(snippet.content, oracle_syms)
            gt_hits = self._find_symbol_hits(snippet.content, gt_syms)
            hit_oracle = len(oracle_hits) > 0
            hit_gt = len(gt_hits) > 0
            useful_or = hit_oracle or hit_gt

            retrieved_info.append({
                'idx': snip_idx,
                'file': snippet.file_path,
                'type': snippet.snippet_type,
                'name': snippet.name,
                'score': score,
                'truncated': False,
                'hit_oracle': hit_oracle,
                'hit_gt': hit_gt,
                'useful_or': useful_or,
                'oracle_hits': oracle_hits,
                'gt_hits': gt_hits,
                'oracle_hit_count': len(oracle_hits),
                'gt_hit_count': len(gt_hits),
            })
            snippet_char_spans.append({
                'idx': snip_idx,
                'start_char_in_retrieved_context': start,
                'end_char_in_retrieved_context': end
            })
            snip_idx += 1

        # full_context 里 Retrieved Context 的前缀
        retrieved_prefix = "=== Retrieved Context ===\n"
        target_prefix = "\n\n=== Target Function ===\n"

        full_context = (
            f"{retrieved_prefix}"
            f"{retrieved_context}"
            f"{target_prefix}"
            f"{func_signature}\n"
            f"    # TODO: Implement this function\n"
        )

        # 把 retrieved_context 内的 char span 转成 full_context 内的 char span（加偏移）
        retrieved_offset = len(retrieved_prefix)
        snippet_char_spans_full = []
        for s in snippet_char_spans:
            snippet_char_spans_full.append({
                'idx': s['idx'],
                'start_char': s['start_char_in_retrieved_context'] + retrieved_offset,
                'end_char': s['end_char_in_retrieved_context'] + retrieved_offset
            })

        # metadata = {
        #     'method': self.config.method,
        #     'target_tokens': target_tokens,
        #     'actual_tokens': self._estimate_tokens(full_context),
        #     'num_retrieved': len(retrieved_info),

        #     # 全量 meta（后续做 useful/冗余标注与 attention 聚合要用）
        #     'retrieved_snippets': retrieved_info,

        #     # 显式 span：你后面可以用 tokenizer offset mapping 映射到 token span
        #     'snippet_char_spans': snippet_char_spans_full,
        #         # ---- 新增：预算拆解与状态 ----
        #     'prompt_overhead_tokens_est': prompt_overhead,
        #     'signature_tokens_est': signature_tokens,
        #     'available_tokens_est': available_tokens,
        #     'budget_insufficient': budget_insufficient,
        #     'budget_status': (
        #         'ok' if not budget_insufficient
        #         else 'insufficient_budget_for_retrieved_context'
        #     ),

        #     # 可选：方便排查为什么为 0
        #     'budget_note': (
        #         '' if not budget_insufficient
        #         else f"target_tokens({target_tokens}) <= signature_tokens_est({signature_tokens}) + prompt_overhead_est({prompt_overhead})"
        #     ),

        #     'retrieved_snippets': retrieved_info,
        #     'snippet_char_spans': snippet_char_spans_full,
        #     'retrieved_snippets_preview': retrieved_info[:20],
        # }

        metadata = {
            "method": self.config.method,
            "target_tokens": target_tokens,
            "actual_tokens": self._estimate_tokens(full_context),
            "num_retrieved": len(retrieved_info),

            # 全量 meta（后续做 useful/冗余标注与 attention 聚合要用）
            "retrieved_snippets": retrieved_info,

            # 显式 span：后面可用 tokenizer offset mapping 映射到 token span
            "snippet_char_spans": snippet_char_spans_full,

            # 预算拆解与状态（估算值）
            "prompt_overhead_tokens_est": prompt_overhead,
            "signature_tokens_est": signature_tokens,
            "available_tokens_est": available_tokens,
            "budget_insufficient": budget_insufficient,
            "budget_status": (
                "ok" if not budget_insufficient
                else "insufficient_budget_for_retrieved_context"
            ),
            "budget_note": (
                "" if not budget_insufficient
                else f"target_tokens({target_tokens}) <= signature_tokens_est({signature_tokens}) + prompt_overhead_est({prompt_overhead})"
            ),

            "retrieved_snippets_preview": retrieved_info[:20],
        }

        metadata["stats"] = {
            "filtered_pool_size": filtered_pool_size,
            "retriever_index_size": retriever_index_size,
            "retrieved_candidates": retrieved_candidates,
            "available_tokens_est": available_tokens,
            "filled_tokens": current_tokens,
            "avg_candidate_tokens": avg_cand_tokens,
            "max_fillable_tokens_est": max_fillable_tokens_est,
        }
        

        return full_context, metadata









# ==================== 数据集生成器 ====================

class RAGDatasetGenerator:
    """RAG 数据集生成器（防数据泄露版）"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.context_builder = SparseRetrievalContextBuilder(config)
        
    def load_dataset(self) -> List[Dict]:
        """加载 CoderEval 数据集"""
        logger.info(f"加载数据集: {self.config.dataset_path}")
        
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        records = data.get('RECORDS', [])
        logger.info(f"加载了 {len(records)} 个任务")
        
        return records
    
    def build_prompt(self, task: Dict, context: str) -> str:
        """构建完整的模型输入 prompt"""
        name = task.get('name', '')
        docstring = task.get('docstring', '')
        
        prompt = f"""Complete the following Python function based on the retrieved context.

{context}

Implement the function `{name}` according to its docstring:
{docstring}

=== Your Implementation ===
"""
        return prompt
    
    def generate_for_length(self, tasks: List[Dict], target_tokens: int) -> List[Dict]:
        """为指定长度生成数据"""
        results = []
        total = len(tasks)
        
        logger.info(f"生成 {target_tokens} tokens 的上下文...")
        
        # 清除缓存，确保每次都重新过滤
        self.context_builder.extractor.clear_cache()
        
        for idx, task in enumerate(tasks):
            if (idx + 1) % 50 == 0:
                logger.info(f"  进度: {idx + 1}/{total}")
            
            try:
                context, metadata = self.context_builder.build_context(task, target_tokens)
                prompt = self.build_prompt(task, context)
                
                result = {
                    '_id': task.get('_id', ''),
                    'name': task.get('name', ''),
                    'project': task.get('project', ''),
                    'file_path': task.get('file_path', ''),
                    'level': task.get('level', ''),
                    'docstring': task.get('docstring', ''),
                    'ground_truth': task.get('code', ''),
                    'retrieval_method': self.config.method,
                    'target_tokens': target_tokens,
                    'actual_tokens': metadata['actual_tokens'],
                    'num_retrieved': metadata['num_retrieved'],
                    'retrieved_snippets': metadata['retrieved_snippets'],
                    'context': context,
                    'prompt': prompt
                }
                results.append(result)
                
            except Exception as e:
                logger.warning(f"任务 {task.get('_id', '')} 处理失败: {e}")
                minimal_context = f"# Function to implement\ndef {task.get('name', 'unknown')}():\n    pass"
                prompt = self.build_prompt(task, minimal_context)
                
                result = {
                    '_id': task.get('_id', ''),
                    'name': task.get('name', ''),
                    'project': task.get('project', ''),
                    'file_path': task.get('file_path', ''),
                    'level': task.get('level', ''),
                    'docstring': task.get('docstring', ''),
                    'ground_truth': task.get('code', ''),
                    'retrieval_method': self.config.method,
                    'target_tokens': target_tokens,
                    'actual_tokens': len(prompt) // 3,
                    'num_retrieved': 0,
                    'retrieved_snippets': [],
                    'context': minimal_context,
                    'prompt': prompt,
                    'error': str(e)
                }
                results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], target_tokens: int):
        """保存结果"""
        output_file = os.path.join(
            self.config.output_dir,
            f"rag_{self.config.method}_{target_tokens}tokens.jsonl"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"已保存: {output_file}")
    
    def run(self):
        """运行数据集生成"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        tasks = self.load_dataset()
        
        all_stats = {}
        
        for target_tokens in self.config.context_lengths:
            results = self.generate_for_length(tasks, target_tokens)
            self.save_results(results, target_tokens)
            
            avg_tokens = sum(r['actual_tokens'] for r in results) / len(results)
            avg_retrieved = sum(r['num_retrieved'] for r in results) / len(results)
            
            all_stats[target_tokens] = {
                'total': len(results),
                'avg_actual_tokens': avg_tokens,
                'avg_num_retrieved': avg_retrieved
            }
        
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'method': self.config.method,
            'dataset_path': self.config.dataset_path,
            'total_tasks': len(tasks),
            'context_lengths': self.config.context_lengths,
            'statistics': all_stats,
            'leakage_protection': {
                'excluded_file_patterns': LeakageFilter.EXCLUDED_FILE_PATTERNS,
                'exclude_target_file': True,
                'exclude_target_function': True,
                'exclude_ground_truth_matches': True
            }
        }
        
        metadata_file = os.path.join(self.config.output_dir, f'metadata_{self.config.method}.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"元数据已保存: {metadata_file}")
        
        print("\n" + "="*60)
        print(f"RAG 数据集生成完成 (方法: {self.config.method})")
        print("="*60)
        print(f"\n输出目录: {self.config.output_dir}")
        print(f"任务总数: {len(tasks)}")
        print(f"\n防泄露措施:")
        print(f"  - 排除 *passk_validte*.py 文件")
        print(f"  - 排除 standalone 目录")
        print(f"  - 排除目标文件")
        print(f"  - 排除目标函数")
        print(f"  - 排除包含 ground truth 的片段")
        print(f"\n生成的文件:")
        for tokens in self.config.context_lengths:
            stats = all_stats[tokens]
            print(f"  - rag_{self.config.method}_{tokens}tokens.jsonl")
            print(f"      平均实际 tokens: {stats['avg_actual_tokens']:.0f}")
            print(f"      平均检索片段数: {stats['avg_num_retrieved']:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='CoderEval 稀疏检索上下文生成 (V2 防泄露版)'
    )
    
    parser.add_argument('--dataset', type=str,
                        default='home/travis/builds/CoderEval4Python.json',
                        help='数据集路径')
    parser.add_argument('--repos', type=str,
                        default='home/travis/builds/repos',
                        help='项目仓库路径')
    parser.add_argument('--output', type=str,
                        default='./rag_contexts',
                        help='输出目录')
    parser.add_argument('--method', type=str, choices=['bm25', 'jaccard'],
                        default='bm25',
                        help='检索方法')
    parser.add_argument('--context-lengths', type=int, nargs='+',
                        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608],
                        help='目标上下文长度列表')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = RetrievalConfig(
        dataset_path=args.dataset,
        repos_path=args.repos,
        output_dir=args.output,
        method=args.method,
        context_lengths=args.context_lengths
    )
    
    generator = RAGDatasetGenerator(config)
    generator.run()


if __name__ == '__main__':
    main()

