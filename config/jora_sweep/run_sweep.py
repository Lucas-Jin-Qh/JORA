#!/usr/bin/env python3
"""
JORA 超参数自动化扫描脚本
自动遍历 config/jora_sweep 下的所有配置文件并执行训练

用法:
    # 执行所有实验
    python run_sweep.py

    # 只执行特定阶段
    python run_sweep.py --phase 1
    
    # 只执行特定配置
    python run_sweep.py --config block_s_k_sweep.json
    
    # dry run（不执行，只显示命令）
    python run_sweep.py --dry-run
    
    # 断点续传
    python run_sweep.py --resume
"""

import json
import os
import sys
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import csv

# 路径配置
SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR / "config"
OUTPUT_BASE_DIR = SCRIPT_DIR / "outputs"
RESULTS_DIR = SCRIPT_DIR / "results"

# 训练脚本路径
TRAIN_SCRIPT = SCRIPT_DIR / "examples" / "sft" / "train.py"


@dataclass
class ExperimentConfig:
    """单次实验配置"""
    name: str
    peft_config: dict
    extra_args: dict = field(default_factory=dict)


@dataclass  
class SweepResult:
    """实验结果"""
    experiment_name: str
    config_file: str
    phase: int
    status: str  # running/completed/failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    trainable_params: Optional[int] = None
    
    # 评测指标
    arc_c_score: Optional[float] = None
    arc_c_delta: Optional[float] = None
    gsm8k_score: Optional[float] = None
    gsm8k_delta: Optional[float] = None
    mmlu_score: Optional[float] = None
    mmlu_delta: Optional[float] = None
    
    # 效率指标
    tokens_per_second: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    
    # 实验信息
    notes: str = ""
    log_file: Optional[str] = None


class JoraSweepRunner:
    """JORA 超参数扫描运行器"""
    
    def __init__(self, 
                 model_path: str = "meta-llama/Llama-2-7b-hf",
                 dataset_name: str = "tatsu-lab/alpaca",
                 output_dir: str = None,
                 dry_run: bool = False,
                 resume: bool = False):
        
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.dry_run = dry_run
        self.resume = resume
        
        # 输出目录
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_BASE_DIR / f"sweep_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果记录
        self.results_file = RESULTS_DIR / f"sweep_results_{self.timestamp}.csv"
        self.completed_experiments = self._load_completed_experiments() if resume else set()
        
        # 日志
        self.log_file = self.output_dir / "sweep_log.txt"
        
    def _load_completed_experiments(self) -> set:
        """加载已完成的实验列表（用于断点续传）"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                reader = csv.DictReader(f)
                return {row['experiment_name'] for row in reader if row['status'] == 'completed'}
        return set()
    
    def parse_sweep_config(self, config_file: Path) -> List[ExperimentConfig]:
        """解析扫描配置文件"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        experiments = []
        fixed_params = config.get('fixed_params', {})
        
        for exp in config.get('experiments', []):
            # 合并配置
            peft_config = fixed_params.copy()
            peft_config.update({k: v for k, v in exp.items() 
                              if k not in ['name', 'notes', 'expected_params', 'coverage_rate']})
            
            # 添加 target_modules（如果不存在）
            if 'target_modules' not in peft_config:
                peft_config['target_modules'] = ["q_proj", "v_proj"]
            
            experiments.append(ExperimentConfig(
                name=exp['name'],
                peft_config=peft_config,
                extra_args={'notes': exp.get('notes', '')}
            ))
        
        return experiments
    
    def generate_peft_config_file(self, exp: ExperimentConfig, output_dir: Path) -> Path:
        """生成 PEFT 配置文件"""
        config_file = output_dir / f"{exp.name}_peft_config.json"
        
        # 构建完整的配置文件
        full_config = {
            "peft_type": "JORA",
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "ddp_allow_unused_parameters": True,
            **exp.peft_config
        }
        
        with open(config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        return config_file
    
    def build_train_command(self, 
                           exp: ExperimentConfig, 
                           peft_config_file: Path,
                           exp_output_dir: Path) -> List[str]:
        """构建训练命令"""
        cmd = [
            "python", str(TRAIN_SCRIPT),
            "--seed", "42",
            "--model_name_or_path", self.model_path,
            "--dataset_name", self.dataset_name,
            "--chat_template_format", "none",
            "--add_special_tokens", "False",
            "--append_concat_token", "False",
            "--splits", "train",
            "--torch_dtype", "bfloat16",
            "--num_train_epochs", "3",
            "--logging_steps", "10",
            "--log_level", "info",
            "--logging_strategy", "steps",
            "--eval_strategy", "epoch",
            "--save_strategy", "epoch",
            "--save_total_limit", "1",
            "--packing", "False",
            "--learning_rate", "2e-4",
            "--lr_scheduler_type", "cosine",
            "--weight_decay", "0.01",
            "--warmup_ratio", "0.03",
            "--max_grad_norm", "1.0",
            "--output_dir", str(exp_output_dir),
            "--per_device_train_batch_size", "4",
            "--gradient_accumulation_steps", "8",
            "--gradient_checkpointing",
            "--use_reentrant",
            "--dataset_text_field", "text",
            "--use_peft_jora",
            "--lora_target_modules", "q_proj,v_proj",
            f"--jora_s_l", str(exp.peft_config.get('S_L', 96)),
            f"--jora_s_r", str(exp.peft_config.get('S_R', 96)),
            f"--jora_k", str(exp.peft_config.get('k', 24)),
        ]
        
        # 添加 JORA 特有参数
        if 'block_size' in exp.peft_config:
            cmd.extend(["--jora_block_size", str(exp.peft_config['block_size'])])
        
        if 'selection' in exp.peft_config:
            cmd.extend([
                f"--jora_selection_type", exp.peft_config['selection'],
                f"--jora_ema_update_interval", str(exp.peft_config.get('update_interval', 50)),
            ])
            if exp.peft_config['selection'] == 'topk_ema':
                cmd.extend([f"--jora_selection_group_size", str(exp.peft_config.get('ema_beta', 0.95))])
        
        if 'magnitude' in exp.peft_config:
            cmd.extend([
                f"--jora_magnitude", exp.peft_config['magnitude'],
            ])
            if exp.peft_config.get('oer_temperature'):
                cmd.extend([f"--jora_oer_temperature", str(exp.peft_config['oer_temperature'])])
        
        if 'rotation_param' in exp.peft_config:
            cmd.extend([f"--jora_rotation_param", exp.peft_config['rotation_param']])
        
        # DDP 参数
        cmd.extend([
            "--ddp_find_unused_parameters", "True",
            "--ddp_timeout", "1800",
        ])
        
        return cmd
    
    def run_experiment(self, exp: ExperimentConfig, config_file: Path) -> SweepResult:
        """运行单次实验"""
        result = SweepResult(
            experiment_name=exp.name,
            config_file=str(config_file),
            phase=0,
            status="running"
        )
        
        # 创建实验输出目录
        exp_output_dir = self.output_dir / exp.name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建命令
        cmd = self.build_train_command(exp, config_file, exp_output_dir)
        
        # 打印命令
        cmd_str = " \\\n    ".join(cmd)
        print(f"\n{'='*80}")
        print(f"Running experiment: {exp.name}")
        print(f"{'='*80}")
        print(f"Command:\n{cmd_str}")
        print(f"{'='*80}")
        
        if self.dry_run:
            print("[DRY RUN - Not executing]")
            result.status = "dry_run"
            return result
        
        # 执行训练
        log_file = exp_output_dir / "train.log"
        result.log_file = str(log_file)
        
        start_time = datetime.datetime.now()
        result.start_time = start_time.isoformat()
        
        try:
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # 实时打印输出
                for line in proc.stdout:
                    print(line, end='')
                    f.write(line)
                
                proc.wait()
                
                if proc.returncode == 0:
                    result.status = "completed"
                else:
                    result.status = "failed"
                    
        except Exception as e:
            result.status = "failed"
            print(f"Error: {e}")
        
        end_time = datetime.datetime.now()
        result.end_time = end_time.isoformat()
        
        # 尝试从日志中提取参数数量
        result.trainable_params = self._extract_params_from_log(log_file)
        
        return result
    
    def _extract_params_from_log(self, log_file: Path) -> Optional[int]:
        """从日志中提取可训练参数数量"""
        if not log_file.exists():
            return None
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # 查找 trainable parameters
            import re
            match = re.search(r'trainable params: ([\d,]+)', content)
            if match:
                return int(match.group(1).replace(',', ''))
        except Exception:
            pass
        
        return None
    
    def save_result(self, result: SweepResult):
        """保存实验结果到 CSV"""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 创建/追加 CSV
        file_exists = self.results_file.exists()
        
        with open(self.results_file, 'a', newline='') as f:
            fieldnames = [
                'experiment_name', 'config_file', 'phase', 'status',
                'start_time', 'end_time', 'trainable_params',
                'arc_c_score', 'arc_c_delta',
                'gsm8k_score', 'gsm8k_delta',
                'mmlu_score', 'mmlu_delta',
                'tokens_per_second', 'peak_vram_gb',
                'notes', 'log_file'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'experiment_name': result.experiment_name,
                'config_file': result.config_file,
                'phase': result.phase,
                'status': result.status,
                'start_time': result.start_time,
                'end_time': result.end_time,
                'trainable_params': result.trainable_params,
                'arc_c_score': result.arc_c_score,
                'arc_c_delta': result.arc_c_delta,
                'gsm8k_score': result.gsm8k_score,
                'gsm8k_delta': result.gsm8k_delta,
                'mmlu_score': result.mmlu_score,
                'mmlu_delta': result.mmlu_delta,
                'tokens_per_second': result.tokens_per_second,
                'peak_vram_gb': result.peak_vram_gb,
                'notes': result.notes,
                'log_file': result.log_file
            })
    
    def run_phase(self, config_file: Path) -> List[SweepResult]:
        """运行单个配置文件中的所有实验"""
        experiments = self.parse_sweep_config(config_file)
        results = []
        
        for exp in experiments:
            # 跳过已完成的实验（断点续传）
            if exp.name in self.completed_experiments:
                print(f"⏭ Skipping completed experiment: {exp.name}")
                continue
            
            # 生成配置
            peft_config_file = self.generate_peft_config_file(exp, self.output_dir)
            
            # 运行实验
            result = self.run_experiment(exp, peft_config_file)
            results.append(result)
            
            # 保存结果
            self.save_result(result)
        
        return results
    
    def run_all(self, config_files: List[Path] = None):
        """运行所有配置的实验"""
        if config_files is None:
            # 自动发现所有配置文件
            config_files = sorted(CONFIG_DIR.glob("*.json"))
        
        print(f"\n{'#'*80}")
        print(f"JORA Hyperparameter Sweep")
        print(f"{'#'*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total configurations: {len(config_files)}")
        print(f"{'#'*80}\n")
        
        all_results = []
        for config_file in config_files:
            print(f"\n{'='*80}")
            print(f"Processing: {config_file.name}")
            print(f"{'='*80}")
            
            results = self.run_phase(config_file)
            all_results.extend(results)
        
        # 汇总结果
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: List[SweepResult]):
        """打印实验汇总"""
        print(f"\n{'='*80}")
        print("SWEEP SUMMARY")
        print(f"{'='*80}")
        
        completed = [r for r in results if r.status == 'completed']
        failed = [r for r in results if r.status == 'failed']
        
        print(f"Total experiments: {len(results)}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        print(f"Dry runs: {len([r for r in results if r.status == 'dry_run'])}")
        
        if completed:
            print(f"\nCompleted experiments:")
            for r in completed:
                params = f"{r.trainable_params:,}" if r.trainable_params else "N/A"
                print(f"  ✓ {r.name}: {params} params")
        
        if failed:
            print(f"\nFailed experiments:")
            for r in failed:
                print(f"  ✗ {r.name}")
        
        print(f"\nResults saved to: {self.results_file}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="JORA Hyperparameter Sweep Runner")
    parser.add_argument("--phase", type=int, choices=[1,2,3,4,5], 
                       help="Run specific phase only")
    parser.add_argument("--config", type=str, 
                       help="Run specific config file (e.g., block_s_k_sweep.json)")
    parser.add_argument("--model_path", type=str, 
                       default="meta-llama/Llama-2-7b-hf",
                       help="Model path or name")
    parser.add_argument("--dataset", type=str, 
                       default="tatsu-lab/alpaca",
                       help="Dataset name")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show commands without executing")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run")
    
    args = parser.parse_args()
    
    # 确定要运行的配置文件
    config_files = []
    if args.config:
        config_path = CONFIG_DIR / args.config
        if config_path.exists():
            config_files = [config_path]
        else:
            print(f"Config file not found: {config_path}")
            sys.exit(1)
    elif args.phase:
        # 根据阶段选择配置
        phase_map = {
            1: "block_s_k_sweep.json",
            2: "block_size_sweep.json", 
            3: "selection_refine.json",
            4: "magnitude_compare.json",
            5: "rotation_compare.json"
        }
        if args.phase in phase_map:
            config_files = [CONFIG_DIR / phase_map[args.phase]]
    else:
        # 默认运行所有配置（按推荐顺序）
        config_files = [
            CONFIG_DIR / "main_config.json",
            CONFIG_DIR / "block_s_k_sweep.json",
            CONFIG_DIR / "block_size_sweep.json",
            CONFIG_DIR / "diag_baseline.json",
            CONFIG_DIR / "selection_refine.json",
            CONFIG_DIR / "magnitude_compare.json",
            CONFIG_DIR / "rotation_compare.json",
        ]
        config_files = [f for f in config_files if f.exists()]
    
    # 创建运行器并执行
    runner = JoraSweepRunner(
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        resume=args.resume
    )
    
    runner.run_all(config_files)


if __name__ == "__main__":
    main()

