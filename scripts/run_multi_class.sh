#!/bin/bash
# Multi-class forgetting on CIFAR-100 (Table 2)
set -e
BACKBONE=${1:-resnet18}
DEVICE=${2:-cuda}
OUT=./results/multi_class

python main.py --mode multi --backbone $BACKBONE --dataset cifar100 \
    --forget castle keyboard --num-seeds 3 --device $DEVICE --output-dir $OUT

python main.py --mode multi --backbone $BACKBONE --dataset cifar100 \
    --forget castle keyboard telephone --num-seeds 3 --device $DEVICE --output-dir $OUT

python main.py --mode multi --backbone $BACKBONE --dataset cifar100 \
    --forget castle keyboard telephone television --num-seeds 3 --device $DEVICE --output-dir $OUT

python main.py --mode multi --backbone $BACKBONE --dataset cifar100 \
    --forget castle keyboard telephone television lawn_mower --num-seeds 3 --device $DEVICE --output-dir $OUT

echo "Done. Results in $OUT"
