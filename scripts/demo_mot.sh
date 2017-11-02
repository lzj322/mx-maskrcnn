export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
export PYTHONPATH=incubator-mxnet/python/
MODEL_PATH=/model/
RESULT_PATH=data/MOT/results/
PREFIX=${MODEL_PATH}final
DATASET=Mot
TEST_SET=MOT17-02
DATASET_PATH=/root/Share/MOT17/train/
mkdir ${RESULT_PATH}
python demo_mask.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --iamge_set ${TEST_SET} \
    --dataset_path ${DATASET_PATH} \
    --prefix ${PREFIX} \
    --result_path ${RESULT_PATH} \
    --has_rpn \
    --epoch 0 \
    --gpu 3
    
    
