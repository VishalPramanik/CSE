#!/bin/bash
# Ablation studies (Tables 5-6)
set -e
DEVICE=${1:-cuda}
OUT=./results/ablation

for TAU in 0.70 0.85 0.95; do
    python main.py --mode single --backbone resnet18 --dataset cifar10 \
        --forget airplane --nontarget bird ship --tau-cov $TAU --device $DEVICE --output-dir $OUT
done

for FRAC in 0.05 0.10 0.15; do
    python main.py --mode single --backbone resnet18 --dataset cifar10 \
        --forget airplane --nontarget bird ship --nontarget-fraction $FRAC --device $DEVICE --output-dir $OUT
done

echo "Done. Results in $OUT"
