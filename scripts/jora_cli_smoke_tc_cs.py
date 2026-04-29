#!/home/jqh/miniconda3/envs/lf/bin/python
"""Wrapper: sanitize sys.path then run the actual script."""
import sys
sys.path = [p for p in sys.path if not p.startswith('/opt/ros')]
sys.path = [p for p in sys.path if '/python3.10/' not in p]

import runpy, os
os.chdir('/home/jqh/Workshop/JORA')
sys.argv = ['jora_cli_smoke_tc_cs.py'] + sys.argv[1:]
sys.exit(runpy.run_path('/home/jqh/Workshop/JORA/scripts/jora_cli_smoke_tc_cs_impl.py', run_name='__main__'))
