#!/usr/bin/env python3
"""
JORA Hyperparameter Sweep Automation Script
============================================

批量执行 JORA 超参数扫描实验，自动生成配置文件并记录结果。

使用方式:
    # 执行所有扫描实验
    python scripts/jora_sweep.py --run-all

    # 仅生成配置文件
    python scripts/jora_sweep.py --generate-only

    # 执行特定阶段的实验
    python scripts/jora_sweep.py --phase 1

    # 从配置文件运行实验
    python scripts/jora_sweep.py --config config/jora_sweep/block_s_k_sweep.json

作者: JORA Team
日期: 2024
"""

import os
import sys
import json
import argparse
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "jora_sweep"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "jora_sweep"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# 默认训练参数
DEFAULT_TRAINING_ARGS = {
    "model_name_or_path": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "tatsu-lab/alpaca",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-4,
    "torch_dtype": "bfloat16",
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 2,
}


class JORASweepRunner:
    """JORA 超参数扫描运行器"""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        dry_run: bool = False,
        parallel: bool = False,
        n_workers: int = 4,
    ):
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.dry_run = dry_run
        self.parallel = parallel
        self.n_workers = n_workers
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # 实验记录
        self.experiment_log = []
        
    def generate_peft_config(self, exp_config: Dict, fixed_params: Dict) -> Dict:
        """生成完整的 PEFT 配置文件"""
        config = fixed_params.copy()
        
        # 添加实验特定的参数
        for key in ["S_L", "S_R", "k", "core", "block_size", "selection", 
                    "ema_beta", "update_interval", "magnitude", "oer_temperature",
                    "rotation_param", "ema_update_interval"]:
            if key in exp_config:
                config[key] = exp_config[key]
        
        # 设置 target_modules
        config["target_modules"] = fixed_params.get("target_modules", ["q_proj", "v_proj"])
        
        return config
    
    def generate_training_command(
        self, 
        exp_name: str, 
        peft_config: Dict,
        training_args: Dict = None
    ) -> str:
        """生成训练命令"""
        if training_args is None:
            training_args = DEFAULT_TRAINING_ARGS.copy()
        
        cmd_parts = [
            "python", str(PROJECT_ROOT / "examples" / "sft" / "train.py"),
            f"--seed 42",
            f"--model_name_or_path {training_args['model_name_or_path']}",
            f"--dataset_name {training_args['dataset_name']}",
            "--chat_template_format none",
            "--add_special_tokens False",
            "--append_concat_token False",
            "--splits train",
            f"--torch_dtype {training_args['torch_dtype']}",
            f"--num_train_epochs {training_args['num_train_epochs']}",
            "--logging_steps 10",
            "--log_level info",
            "--logging_strategy steps",
            "--eval_strategy epoch",
            "--packing False",
            f"--learning_rate {training_args['learning_rate']}",
            "--lr_scheduler_type cosine",
            "--weight_decay 0.01",
            f"--warmup_ratio {training_args['warmup_ratio']}",
            "--max_grad_norm 1.0",
            f"--output_dir {self.output_dir / 'results' / exp_name}",
            f"--per_device_train_batch_size {training_args['per_device_train_batch_size']}",
            f"--gradient_accumulation_steps {training_args['gradient_accumulation_steps']}",
            "--gradient_checkpointing",
            "--use_reentrant",
            "--dataset_text_field text",
        ]
        
        # JORA 特有参数
        cmd_parts.extend([
            "--use_peft_jora",
            f"--jora_s_l {peft_config['S_L']}",
            f"--jora_s_r {peft_config['S_R']}",
            f"--jora_k {peft_config['k']}",
            f"--jora_rotation_param {peft_config['rotation_param']}",
            f"--jora_selection_type {peft_config['selection']}",
            f"--jora_magnitude {peft_config['magnitude']}",
            f"--jora_update_interval {peft_config['update_interval']}",
        ])
        
        # 添加 EMA 参数
        if "ema_beta" in peft_config:
            cmd_parts.append(f"--jora_ema_beta {peft_config['ema_beta']}")
        if "ema_update_interval" in peft_config:
            cmd_parts.append(f"--jora_ema_update_interval {peft_config['ema_update_interval']}")
        
        # 添加温度参数
        if "oer_temperature" in peft_config:
            cmd_parts.append(f"--jora_oer_temperature {peft_config['oer_temperature']}")
            
        # 添加 Core 参数
        if peft_config.get("core") == "block":
            cmd_parts.append(f"--jora_block_size {peft_config['block_size']}")
        
        # DDP 参数
        cmd_parts.extend([
            "--ddp_find_unused_parameters True",
            "--ddp_timeout 1800",
        ])
        
        return " \\\n    ".join(cmd_parts)
    
    def save_config_file(self, exp_name: str, config: Dict) -> Path:
        """保存 PEFT 配置文件"""
        config_path = self.output_dir / "configs" / f"{exp_name}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return config_path
    
    def run_experiment(
        self, 
        exp_name: str, 
        config: Dict, 
        training_args: Dict = None,
        dry_run: bool = False
    ) -> Dict:
        """运行单个实验"""
        # 保存配置
        config_path = self.save_config_file(exp_name, config)
        
        # 生成命令
        command = self.generate_training_command(exp_name, config, training_args)
        
        # 保存命令
        cmd_path = self.output_dir / "logs" / f"{exp_name}.sh"
        with open(cmd_path, 'w') as f:
            f.write(f"#!/bin/bash\n# Experiment: {exp_name}\n# Generated at: {datetime.now()}\n\n{command}\n")
        
        result = {
            "exp_name": exp_name,
            "config_path": str(config_path),
            "command_path": str(cmd_path),
            "command": command,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "metrics": {},
        }
        
        if dry_run:
            result["status"] = "dry_run"
            print(f"[DRY RUN] {exp_name}")
            print(f"  Config: {config_path}")
            print(f"  Command: {cmd_path}")
        else:
            print(f"[QUEUED] {exp_name}")
            result["status"] = "queued"
            
        return result
    
    def run_sweep(self, sweep_config: Dict, training_args: Dict = None) -> List[Dict]:
        """执行扫描实验"""
        results = []
        
        fixed_params = sweep_config.get("fixed_params", {})
        experiments = sweep_config.get("experiments", [])
        
        print(f"\n{'='*60}")
        print(f"Running Sweep: {sweep_config.get('name', 'Unknown')}")
        print(f"Total Experiments: {len(experiments)}")
        print(f"{'='*60}\n")
        
        for exp in experiments:
            exp_name = exp.get("name", f"exp_{len(results)}")
            
            # 合并配置
            config = self.generate_peft_config(exp, fixed_params)
            
            # 运行实验
            result = self.run_experiment(exp_name, config, training_args, self.dry_run)
            results.append(result)
            
        # 保存实验列表
        log_path = self.output_dir / "experiment_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                "sweep_name": sweep_config.get("name", "Unknown"),
                "created_at": datetime.now().isoformat(),
                "experiments": results,
            }, f, indent=2)
        
        return results
    
    def run_from_config_file(self, config_path: str, training_args: Dict = None):
        """从配置文件执行扫描"""
        with open(config_path, 'r') as f:
            sweep_config = json.load(f)
        
        return self.run_sweep(sweep_config, training_args)
    
    def run_all_phases(self, training_args: Dict = None):
        """执行所有扫描阶段"""
        # 执行顺序
        phases = [
            CONFIG_DIR / "main_config.json",
            CONFIG_DIR / "block_s_k_sweep.json",
            CONFIG_DIR / "block_size_sweep.json",
            CONFIG_DIR / "diag_baseline.json",
            CONFIG_DIR / "selection_refine.json",
            CONFIG_DIR / "magnitude_compare.json",
            CONFIG_DIR / "rotation_compare.json",
        ]
        
        all_results = []
        for phase_config in phases:
            if phase_config.exists():
                print(f"\n{'#'*60}")
                print(f"# Processing: {phase_config.name}")
                print(f"{'#'*60}")
                results = self.run_from_config_file(str(phase_config), training_args)
                all_results.extend(results)
            else:
                print(f"[SKIP] {phase_config} not found")
        
        return all_results


def generate_results_template(output_dir: Path = None):
    """生成实验结果记录模板"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    template_data = {
        "experiment_info": {
            "exp_name": "",
            "phase": "",
            "config_file": "",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": "Llama-2-7B",
            "dataset": "Alpaca",
        },
        "jora_config": {
            "core": "block",
            "block_size": 4,
            "S_L": 96,
            "S_R": 96,
            "k": 24,
            "selection": "topk_ema",
            "ema_beta": 0.95,
            "update_interval": 50,
            "magnitude": "oer_softmax",
            "oer_temperature": 2.0,
            "rotation_param": "cayley",
        },
        "training_info": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "trainable_params": 0,
            "training_time_hours": 0,
        },
        "results": {
            "ARC-Challenge": {"base": 0.0, "ft": 0.0, "delta": 0.0},
            "GSM8K": {"base": 0.0, "ft": 0.0, "delta": 0.0},
            "MMLU": {"base": 0.0, "ft": 0.0, "delta": 0.0},
            "HellaSwag": {"base": 0.0, "ft": 0.0, "delta": 0.0},
        },
        "throughput": {
            "tokens_per_second": 0.0,
            "peak_vram_gb": 0.0,
        },
        "notes": "",
    }
    
    return template_data


def main():
    parser = argparse.ArgumentParser(
        description="JORA Hyperparameter Sweep Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 干运行，显示所有命令但不执行
    python scripts/jora_sweep.py --dry-run --run-all

    # 执行所有扫描实验
    python scripts/jora_sweep.py --run-all

    # 执行特定阶段的实验
    python scripts/jora_sweep.py --phase 1

    # 从指定配置文件运行
    python scripts/jora_sweep.py --config config/jora_sweep/block_s_k_sweep.json

    # 生成实验结果模板
    python scripts/jora_sweep.py --generate-template
        """
    )
    
    parser.add_argument(
        "--run-all", 
        action="store_true",
        help="运行所有扫描实验"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="运行特定阶段的实验"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="指定配置文件路径"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行，只生成配置和命令，不执行"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="并行执行实验（需要配置SLURM或PBS）"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="并行工作数（默认: 4）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录"
    )
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="生成实验结果模板"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="模型名称或路径"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="数据集名称"
    )
    
    args = parser.parse_args()
    
    # 设置训练参数
    training_args = DEFAULT_TRAINING_ARGS.copy()
    training_args["model_name_or_path"] = args.model
    training_args["dataset_name"] = args.dataset
    
    # 初始化运行器
    runner = JORASweepRunner(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        parallel=args.parallel,
        n_workers=args.n_workers,
    )
    
    if args.generate_template:
        template = generate_results_template(runner.output_dir)
        template_path = runner.output_dir / "results" / "results_template.json"
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"[DONE] Template saved to: {template_path}")
        return
    
    if args.dry_run:
        print("="*60)
        print("DRY RUN MODE - 只生成配置和命令，不执行训练")
        print("="*60)
    
    if args.run_all:
        runner.run_all_phases(training_args)
    elif args.phase:
        phase_files = {
            1: "block_s_k_sweep.json",
            2: "block_size_sweep.json",
            3: "selection_refine.json",
            4: "magnitude_compare.json",
            5: "rotation_compare.json",
        }
        config_path = CONFIG_DIR / phase_files[args.phase]
        if config_path.exists():
            runner.run_from_config_file(str(config_path), training_args)
        else:
            print(f"[ERROR] Config not found: {config_path}")
    elif args.config:
        runner.run_from_config_file(args.config, training_args)
    else:
        parser.print_help()
    
    if not args.dry_run and runner.experiment_log:
        print("\n" + "="*60)
        print("实验已排队完成！")
        print(f"输出目录: {runner.output_dir}")
        print(f"日志文件: {runner.output_dir / 'experiment_log.json'}")
        print("="*60)


if __name__ == "__main__":
    main()

