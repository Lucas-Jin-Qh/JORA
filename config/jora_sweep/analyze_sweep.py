#!/usr/bin/env python3
"""
JORA 超参数扫描结果分析脚本
分析实验结果，生成论文所需的汇总表格

用法:
    # 分析所有结果
    python analyze_sweep.py
    
    # 分析特定阶段的实验
    python analyze_sweep.py --phase 1
    
    # 生成论文表格
    python analyze_sweep.py --generate-tables
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import csv
import pandas as pd
import numpy as np

# 路径配置
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
CONFIG_DIR = SCRIPT_DIR / "config"


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    name: str
    config_file: str
    phase: int
    status: str
    trainable_params: Optional[int]
    
    # 评测指标
    arc_c_score: Optional[float]
    arc_c_delta: Optional[float]
    gsm8k_score: Optional[float]
    gsm8k_delta: Optional[float]
    mmlu_score: Optional[float]
    mmlu_delta: Optional[float]
    
    # 效率指标
    tokens_per_second: Optional[float]
    peak_vram_gb: Optional[float]
    
    # 额外信息
    notes: str


class SweepAnalyzer:
    """扫描结果分析器"""
    
    def __init__(self, results_file: Path = None):
        self.results_file = results_file or self._find_latest_results_file()
        self.data = self._load_results()
        
    def _find_latest_results_file(self) -> Path:
        """找到最新的结果文件"""
        if not RESULTS_DIR.exists():
            return None
        
        csv_files = list(RESULTS_DIR.glob("sweep_results_*.csv"))
        if not csv_files:
            return None
        
        return max(csv_files, key=lambda f: f.stat().st_mtime)
    
    def _load_results(self) -> pd.DataFrame:
        """加载结果数据"""
        if not self.results_file or not self.results_file.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.results_file)
        return df
    
    def filter_by_phase(self, phase: int) -> pd.DataFrame:
        """按阶段筛选"""
        return self.data[self.data['phase'] == phase]
    
    def filter_by_config(self, config_file: str) -> pd.DataFrame:
        """按配置文件筛选"""
        return self.data[self.data['config_file'] == config_file]
    
    def get_completed_experiments(self) -> pd.DataFrame:
        """获取完成的实验"""
        return self.data[self.data['status'] == 'completed']
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """计算统计数据"""
        if df.empty:
            return {}
        
        stats = {
            'count': len(df),
            'mean': {},
            'std': {},
            'max': {},
            'min': {},
        }
        
        metric_cols = ['arc_c_score', 'gsm8k_score', 'mmlu_score', 'tokens_per_second']
        
        for col in metric_cols:
            if col in df.columns:
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    stats['mean'][col] = valid_data.mean()
                    stats['std'][col] = valid_data.std() if len(valid_data) > 1 else 0
                    stats['max'][col] = valid_data.max()
                    stats['min'][col] = valid_data.min()
        
        return stats
    
    def generate_summary_table(self, output_file: Path = None):
        """生成汇总表格"""
        completed = self.get_completed_experiments()
        
        if completed.empty:
            print("No completed experiments to analyze.")
            return
        
        # 按配置分组
        summary = completed.groupby('config_file').agg({
            'experiment_name': 'count',
            'trainable_params': 'mean',
            'arc_c_score': 'mean',
            'gsm8k_score': 'mean',
            'mmlu_score': 'mean',
            'tokens_per_second': 'mean',
        }).round(4)
        
        summary.columns = ['n_experiments', 'avg_params', 'ARC-C', 'GSM8K', 'MMLU', 'tokens/s']
        
        print("\n" + "="*80)
        print("SUMMARY TABLE BY CONFIG")
        print("="*80)
        print(summary.to_string())
        
        if output_file:
            summary.to_csv(output_file)
            print(f"\nSaved to: {output_file}")
    
    def generate_ablation_table(self, output_file: Path = None):
        """生成消融实验表格"""
        completed = self.get_completed_experiments()
        ablation = completed[completed['config_file'] == 'ablation_study.json']
        
        if ablation.empty:
            print("No ablation experiments completed.")
            return
        
        # 按实验名排序
        ablation = ablation.sort_values('experiment_name')
        
        table = ablation[['experiment_name', 'trainable_params', 'arc_c_delta', 'gsm8k_delta', 'tokens_per_second']].copy()
        table = table.rename(columns={
            'experiment_name': 'Configuration',
            'trainable_params': 'Params',
            'arc_c_delta': 'ARC-C Δ',
            'gsm8k_delta': 'GSM8K Δ',
            'tokens_per_second': 'tokens/s'
        })
        
        print("\n" + "="*80)
        print("ABLATION STUDY TABLE")
        print("="*80)
        print(table.to_string(index=False, float_format='%.4f'))
        
        if output_file:
            table.to_csv(output_file, index=False)
            print(f"\nSaved to: {output_file}")
    
    def generate_sweep_curves(self, x_col: str, y_col: str, output_file: Path = None):
        """生成扫描曲线（用于绘制 S/k 影响曲线）"""
        completed = self.get_completed_experiments()
        
        if completed.empty:
            print("No completed experiments to plot.")
            return
        
        # 提取 S 和 k 值
        def extract_sk(name):
            import re
            s_match = re.search(r'S(\d+)', name)
            k_match = re.search(r'k(\d+)', name)
            s = int(s_match.group(1)) if s_match else None
            k = int(k_match.group(1)) if k_match else None
            return s, k
        
        completed['S'] = completed['experiment_name'].apply(lambda x: extract_sk(x)[0])
        completed['k'] = completed['experiment_name'].apply(lambda x: extract_sk(x)[1])
        
        # 过滤有效数据
        valid = completed.dropna(subset=['S', 'k', y_col])
        
        if valid.empty:
            print(f"No valid data for {y_col}")
            return
        
        # 按 S 分组统计
        grouped = valid.groupby('k')[y_col].agg(['mean', 'std'])
        
        print("\n" + "="*80)
        print(f"SCAN CURVES: {y_col} vs k (grouped by S)")
        print("="*80)
        print(grouped.to_string())
        
        if output_file:
            grouped.to_csv(output_file)
            print(f"\nSaved to: {output_file}")
    
    def generate_paper_table1_main_results(self, output_file: Path = None):
        """生成论文 Table 1: Main Results"""
        completed = self.get_completed_experiments()
        
        # 只选取主实验配置
        main_configs = ['main_config.json', 'block_s_k_sweep.json']
        main = completed[completed['config_file'].isin(main_configs)]
        
        if main.empty:
            print("No main experiments completed.")
            return
        
        # 选取最佳配置
        if 'gsm8k_delta' in main.columns:
            best = main.loc[main['gsm8k_delta'].idxmax()] if main['gsm8k_delta'].notna().any() else main.iloc[0]
        else:
            best = main.iloc[0]
        
        table = f"""
| Method | Params | ARC-C (Δ) | GSM8K (Δ) | MMLU (Δ) | Avg Δ | Tokens/s |
|--------|--------|-----------|-----------|----------|-------|----------|
| JORA(block,S=96,k=24) | {best.get('trainable_params', 'N/A'):,} | - | - | - | - | - |
"""
        
        print("\n" + "="*80)
        print("TABLE 1: MAIN RESULTS (Template)")
        print("="*80)
        print(table)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(table)
            print(f"\nSaved to: {output_file}")
    
    def export_all_tables(self, output_dir: Path):
        """导出所有表格"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generate_summary_table(output_dir / "summary_by_config.csv")
        self.generate_ablation_table(output_dir / "ablation_results.csv")
        self.generate_sweep_curves(output_dir / "sweep_curves_k_vs_gsm8k.csv")
        self.generate_paper_table1_main_results(output_dir / "table1_main_results.md")


def main():
    parser = argparse.ArgumentParser(description="JORA Sweep Result Analyzer")
    parser.add_argument("--results_file", type=str, help="Path to results CSV file")
    parser.add_argument("--phase", type=int, choices=[1,2,3,4,5,6], 
                       help="Filter by phase")
    parser.add_argument("--config", type=str, 
                       help="Filter by config file name")
    parser.add_argument("--generate-tables", action="store_true",
                       help="Generate all paper tables")
    parser.add_argument("--output_dir", type=str, default="paper_tables",
                       help="Output directory for tables")
    
    args = parser.parse_args()
    
    # 创建分析器
    results_file = Path(args.results_file) if args.results_file else None
    analyzer = SweepAnalyzer(results_file)
    
    if analyzer.data.empty:
        print("No results found!")
        print(f"Please run sweep experiments first.")
        return
    
    print(f"Loaded results from: {analyzer.results_file}")
    print(f"Total experiments: {len(analyzer.data)}")
    print(f"Completed: {len(analyzer.get_completed_experiments())}")
    
    # 根据参数执行分析
    if args.generate_tables:
        output_dir = Path(args.output_dir)
        analyzer.export_all_tables(output_dir)
    elif args.phase:
        filtered = analyzer.filter_by_phase(args.phase)
        print(f"\nPhase {args.phase} experiments:")
        print(filtered[['experiment_name', 'status', 'trainable_params']].to_string(index=False))
    elif args.config:
        filtered = analyzer.filter_by_config(args.config)
        print(f"\nConfig {args.config} experiments:")
        print(filtered[['experiment_name', 'status', 'trainable_params']].to_string(index=False))
    else:
        # 默认显示汇总
        analyzer.generate_summary_table()


if __name__ == "__main__":
    main()

