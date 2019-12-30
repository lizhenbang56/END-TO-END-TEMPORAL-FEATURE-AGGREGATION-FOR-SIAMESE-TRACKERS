#!/usr/bin/env bash
version=$1
gpu=$2
start=$3
end=$4
epoch=$5
weight=results/got10k_v$version/model_0$epoch.pth
echo $weight
source activate pytorch1.3_python3.6
cd /home/etvuz/project3/siamrcnn2

python tools/get_got10k_proposals.py --config_file=experiments/got10k_v$version/e2e_faster_rcnn_R_50_FPN_1x.yaml \
--gpu=$gpu --start=$start --end=$end \
--output_dir=results/got10k_v$version/$epoch \
--epoch=$epoch --phase=run_model \
--weight=$weight