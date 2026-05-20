#!/bin/bash
# Single-class cross-dataset unlearning (Table 1)
set -e
BACKBONE=${1:-resnet18}
DEVICE=${2:-cuda}
OUT=./results/single_class

python main.py --mode single --backbone $BACKBONE --dataset cifar10 \
    --forget airplane --nontarget bird ship --num-seeds 3 --device $DEVICE --output-dir $OUT

python main.py --mode single --backbone $BACKBONE --dataset cifar10 \
    --forget truck --nontarget automobile --num-seeds 3 --device $DEVICE --output-dir $OUT

python main.py --mode single --backbone $BACKBONE --dataset cifar100 \
    --forget shark --nontarget aquarium_fish flatfish ray trout --num-seeds 3 --device $DEVICE --output-dir $OUT

echo "Done. Results in $OUT"
