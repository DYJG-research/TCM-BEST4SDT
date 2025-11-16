#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器

生成详细的评测报告，包含HTML格式的可视化报告
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import html

logger = logging.getLogger(__name__)

# 标准维度顺序
STANDARD_DIMENSION_ORDER = [
    "证型", "病性", "病位", "治则治法",
    "病因", "病机",
    "方证契合度", "方剂配伍规律", "配伍禁忌", "药材安全性分析", "妊娠禁忌",
    "CoT内容完备性", "CoT准确性",
    "煎服方法", "注意事项", "随症加减"
]

# 展示名映射：仅用于报告显示，不影响数据键
DISPLAY_NAME_MAP = {
    "安全问题": "大模型内容安全",
}

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        """初始化报告生成器"""
        pass
    
    def generate_report(self, final_results: Dict[str, Any], 
                       detailed_results: List[Dict[str, Any]], 
                       output_path: str,
                       general_assessment_task_results: Dict[str, List[Dict[str, Any]]] | None = None):
        """
        生成完整的评测报告
        
        Args:
            final_results: 最终评测结果
            detailed_results: 详细评测结果
            output_path: 输出路径
        """
        try:
            # 生成HTML报告（不再生成 summary.json）
            html_content = self._generate_html_report(final_results, detailed_results, general_assessment_task_results)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"评测报告已生成: {output_path}")
            
        except Exception as e:
            logger.error(f"生成报告时出错: {e}")
            raise
    
    def _generate_html_report(self, final_results: Dict[str, Any], 
                             detailed_results: List[Dict[str, Any]],
                             general_assessment_task_results: Dict[str, List[Dict[str, Any]]] | None = None) -> str:
        """
        生成HTML格式的报告
        
        Args:
            final_results: 最终结果
            detailed_results: 详细结果
            
        Returns:
            HTML内容
        """
        # 参与的类别
        participating_classes = final_results.get('participating_classes', [])
        tcm_participating = (
            isinstance(participating_classes, list) and 
            ('中医辨证论治' in participating_classes)
        )
        has_tcm_cases = bool(detailed_results)

        # 构建TCM维度区块
        tcm_sections_html = ""
        if tcm_participating and has_tcm_cases:
            tcm_sections_html = f"""
        <div class="category-section">
            <h3>客观题评测</h3>
            <div class="dimension-grid">
                {self._generate_dimension_cards(final_results, ["证型", "病位", "治则治法", "病性"])}
            </div>
        </div>
        
        <div class="category-section">
            <h3>奖励模型评分</h3>
            <div class="dimension-grid">
                {self._generate_dimension_cards(final_results, ["方证契合度"])}
            </div>
        </div>
        
        <div class="category-section">
            <h3>LLM评分</h3>
            <div class="dimension-grid">
                {self._generate_llm_dimension_cards(final_results)}
            </div>
        </div>
            """

        # 统计分析区块移除
        stats_section_html = ""

        # 构建新增类别区块
        new_class_scores = final_results.get('general_assessment_tasks', {}) or {}
        # 统计通用评测任务的样本数
        new_class_counts = {}
        if general_assessment_task_results:
            try:
                temp_counts = {}
                for cls, items in (general_assessment_task_results or {}).items():
                    # 原始键 -> 显示名映射，保证与 new_class_scores 的键一致
                    display_key = DISPLAY_NAME_MAP.get(cls, cls)
                    temp_counts[display_key] = len(items or [])
                new_class_counts = temp_counts
            except Exception:
                new_class_counts = {}
        new_class_section_html = ""
        new_class_summary_table_html = ""
        if new_class_scores:
            new_class_section_html = f"""
        <h2>🧩 通用评测任务</h2>
        <div class=\"dimension-grid\">
            {self._generate_new_class_cards(new_class_scores, new_class_counts)}
        </div>
            """
            # 汇总表格
            rows = "".join([
                f"<tr><td>{DISPLAY_NAME_MAP.get(cls, cls)}</td><td>{score:.4f}</td><td>{int(new_class_counts.get(cls, 0))}</td></tr>"
                for cls, score in new_class_scores.items()
            ])
            new_class_summary_table_html = f"""
        <h3>通用评测任务汇总</h3>
        <table class=\"stats-table\">
            <tr><th>类别</th><th>平均分</th><th>样本数</th></tr>
            {rows}
        </table>
            """

        # 构建详细案例区块
        cases_section_html = ""
        if has_tcm_cases:
            cases_section_html = f"""
        <h2>📋 辨证论治任务详细案例结果</h2>
        <p>点击案例标题查看详细信息</p>
        {self._generate_case_details(detailed_results)}
            """

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCM-BSET4SDT评测报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }}
        .score-large {{
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .score-subtitle {{
            font-size: 18px;
            opacity: 0.9;
        }}
        .dimension-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .dimension-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .dimension-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .dimension-score {{
            font-size: 24px;
            font-weight: bold;
            color: #27ae60;
            margin: 5px 0;
        }}
        .dimension-weight {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .stats-table th, .stats-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .stats-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f, #27ae60);
            transition: width 0.3s ease;
        }}
        .case-details {{
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .case-header {{
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            cursor: pointer;
        }}
        .case-content {{
            padding: 15px;
            display: none;
        }}
        .case-content.active {{
            display: block;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: center;
            margin-top: 30px;
        }}
        .category-section {{
            margin: 30px 0;
            padding: 20px;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
        }}
    </style>
    <script>
        function toggleCase(caseId) {{
            const content = document.getElementById('case-content-' + caseId);
            content.classList.toggle('active');
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>TCM-BSET4SDT评测报告</h1>
        
        <div class="summary-box">
            <div class="score-subtitle">总体评分</div>
            <div class="score-large">{final_results['total_score']:.4f}</div>
            <div class="score-subtitle">评测案例数: {final_results.get('num_cases', 0)}</div>
        </div>
        
        <h2>📊 辨证论治任务评测维度得分</h2>
        {tcm_sections_html}
        
        {new_class_section_html}
        {new_class_summary_table_html}
        
        {stats_section_html}
        
        {cases_section_html}
        
        <div class="timestamp">
            报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _generate_dimension_cards(self, final_results: Dict[str, Any], 
                                 dimensions: List[str]) -> str:
        """
        生成维度卡片HTML
        
        Args:
            final_results: 最终结果
            dimensions: 维度列表
            
        Returns:
            HTML内容
        """
        cards_html = ""
        
        for dimension in dimensions:
            score = final_results['dimension_scores'].get(dimension, 0.0)
            meta_info = f"得分: {score:.4f}"
            
            # 计算进度条宽度
            progress_width = min(score * 100, 100)
            
            cards_html += f"""
            <div class="dimension-card">
                <div class="dimension-header">{dimension}</div>
                <div class="dimension-score">{score:.4f}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress_width}%"></div>
                </div>
                <div class="dimension-weight">{meta_info}</div>
            </div>
            """
        
        return cards_html
    
    def _generate_llm_dimension_cards(self, final_results: Dict[str, Any]) -> str:
        """
        生成LLM判分维度卡片HTML（动态判断是否包含Think内容完备性和CoT准确性）
        
        Args:
            final_results: 最终结果
            
        Returns:
            LLM维度卡片HTML
        """
        dimension_scores = final_results.get("dimension_scores", {})
        
        # LLM判分维度的标准顺序
        llm_dimensions_order = [
            "病因", "病机",
            "CoT内容完备性", "CoT准确性",
            # 处方相关四维度（展示在 LLM评分）
            "方剂配伍规律", "药材安全性分析", "配伍禁忌", "妊娠禁忌",
            "煎服方法", "注意事项", "随症加减"
        ]
        
        # 只显示实际存在的维度
        llm_dimensions = [dim for dim in llm_dimensions_order if dim in dimension_scores]
        
        return self._generate_dimension_cards(final_results, llm_dimensions)
    
    def _generate_new_class_cards(self, new_class_scores: Dict[str, float], new_class_counts: Dict[str, int]) -> str:
        """
        生成新增类别的卡片HTML
        
        Args:
            new_class_scores: 类别到平均分的映射（0-1）
        Returns:
            HTML内容
        """
        cards_html = ""
        for class_name, score in new_class_scores.items():
            display_name = DISPLAY_NAME_MAP.get(class_name, class_name)
            progress_width = min(max(score, 0.0) * 100, 100)
            cards_html += f"""
            <div class=\"dimension-card\">
                <div class=\"dimension-header\">{display_name}</div>
                <div class=\"dimension-score\">{score:.4f}</div>
                <div class=\"progress-bar\">
                    <div class=\"progress-fill\" style=\"width: {progress_width}%\"></div>
                </div>
                <div class=\"dimension-weight\">平均准确率: {score:.4f} | 样本数: {int(new_class_counts.get(class_name, 0))}</div>
            </div>
            """
        return cards_html
    
    def _generate_case_details(self, detailed_results: List[Dict[str, Any]]) -> str:
        """
        生成案例详情HTML
        
        Args:
            detailed_results: 详细结果
            
        Returns:
            HTML内容
        """
        cases_html = ""
        
        for i, case_result in enumerate(detailed_results):
            case_id = case_result.get('case_id', f'case_{i}')
            instruction = case_result.get('instruction', '')
            diagnosis = case_result.get('diagnosis', '未知')
            
            # 计算案例总分
            dimension_scores = case_result.get('dimension_scores', {})
            case_total = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0
            
            # 生成维度得分表格（按标准顺序）
            dimensions_table = "<table class='stats-table'><tr><th>维度</th><th>得分</th></tr>"
            for dim in STANDARD_DIMENSION_ORDER:
                if dim in dimension_scores:
                    score = dimension_scores[dim]
                    dimensions_table += f"<tr><td>{dim}</td><td>{score:.4f}</td></tr>"
            dimensions_table += "</table>"
            
            cases_html += f"""
            <div class="case-details">
                <div class="case-header" onclick="toggleCase('{case_id}')">
                    {case_id} - {diagnosis} (总分: {case_total:.4f})
                </div>
                <div id="case-content-{case_id}" class="case-content">
                    <p><strong>病例描述:</strong></p>
                    <div style="white-space: pre-wrap;">{html.escape(instruction)}</div>
                    <p><strong>诊断:</strong> {diagnosis}</p>
                    <h4>各维度得分:</h4>
                    {dimensions_table}
                </div>
            </div>
            """
        
        return cases_html
    
    # 已移除未使用的统计与摘要/建议生成功能
