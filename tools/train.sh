export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="29500" \
    --nproc_per_node=1 \
    tools/train.py \
    plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov.py \
    --launcher pytorch \
    --seed 0 \
    --cfg-options runner.max_epochs=12 \
    data.samples_per_gpu=10 \
    load_from='work_dirs/lidar_0075v_cam_vov/epoch_5.pth' \
    checkpoint_config.max_keep_ckpts=12 \

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    tools/test.py \
    plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov_robodrive.py \
    checkpoint/lidar_0075_cam_vov.pth \
    --eval bbox \
    --launcher pytorch