#!/bin/bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/sycophancy_experiments/260823_domain_baseline_followup_wordings
python generate_followups.py 2>&1 | tee generation_fu_$(hostname).log
