source activate py37-pytorch1.4
gpus=0,1,2,3,4,5,6,7
NGPUS=8
images_per_gpu=3
FPN_POST_NMS_TOP_N_TRAIN=$[images_per_gpu*1000]
IMS_PER_BATCH=$[NGPUS*images_per_gpu]
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --gpus $gpus MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN $FPN_POST_NMS_TOP_N_TRAIN  SOLVER.IMS_PER_BATCH $IMS_PER_BATCH