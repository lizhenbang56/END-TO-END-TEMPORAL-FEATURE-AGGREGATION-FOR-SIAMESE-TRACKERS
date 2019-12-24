#!/usr/bin/env bash
version=$1
source activate pytorch1.3_python3.6
cd /home/etvuz/project3/siamrcnn2
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--config-file=experiments/got10k_v$version/e2e_faster_rcnn_R_50_FPN_1x.yaml \
OUTPUT_DIR /home/etvuz/project3/siamrcnn2/results/got10k_v$version \
SOLVER.IMS_PER_BATCH 8 \
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 4000  # 注意：对应yaml中的BATCH_SIZE
