#!/bin/bash
cd /workspace/sycophancy_experiments/260923_improve_probe_performance
PY=/workspace/obfuscation-atlas/.venv/bin/python
H=$(hostname)
$PY extract.py --variant base --positions all > logs/extract_base_$H.log 2>&1; echo "EXIT $?" >> logs/extract_base_$H.log
$PY extract.py --variant pool --n-train-per-class '{"bias": 1454, "deception_yes": 964, "deception_no": 490, "bias_followed": 1454}' > logs/extract_pool_$H.log 2>&1; echo "EXIT $?" >> logs/extract_pool_$H.log
$PY extract.py --variant pool_nu --train-filter none_unanimous --n-train-per-class '{"bias": 576, "deception_yes": 406, "deception_no": 170, "bias_followed": 576}' > logs/extract_pool_nu_$H.log 2>&1; echo "EXIT $?" >> logs/extract_pool_nu_$H.log
