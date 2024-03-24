export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    tools/train.py \
    plugin/fudet/configs/lidar_cam/lidar_0075v_cam_vov.py \
    --seed 0 \
    --launcher pytorch

export CUDA_VISIBLE_DEVICES=2,3

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    tools/test.py \
    plugin/fudet/configs/lidar_cam/lidar_0075v_cam_vov_robodrive.py \
    checkpoint/lidar_0075_cam_vov.pth \
    --eval bbox \
    --launcher pytorch