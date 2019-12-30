#!/usr/bin/env bash
version=$1
source activate pytorch1.3_python3.6
cd /home/etvuz/project3/siamrcnn2

python tools/track_got10k.py --config_file=experiments/got10k_v$version/e2e_faster_rcnn_R_50_FPN_1x.yaml \
--gpu=$2 --start=$3 --end=$4 \
--output_dir=results/got10k_v$version \
--weight=results/got10k_v$version/model_0100000.pth