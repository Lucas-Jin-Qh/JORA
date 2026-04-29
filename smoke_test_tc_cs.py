#!/home/jqh/miniconda3/envs/lf/bin/python
"""Wrapper that sanitizes sys.path before importing peft."""
import sys
sys.path = [p for p in sys.path if not p.startswith('/opt/ros')]
sys.path = [p for p in sys.path if '/python3.10/' not in p]

import runpy, os, warnings
warnings.filterwarnings("ignore")

os.chdir('/home/jqh/Workshop/JORA')
runpy.run_path('/home/jqh/Workshop/JORA/smoke_test_tc_cs_impl.py', run_name='__main__')
