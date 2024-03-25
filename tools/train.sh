export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="39500" \
    --nproc_per_node=2 \
    tools/train.py \
    plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov.py \
    --seed 0 \
    --cfg-options runner.max_epochs=36 \
    data.samples_per_gpu=13 \
    load_from='checkpoint/lidar_0075_cam_vov.pth' \
    --launcher pytorch

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=4 \
    tools/train.py \
    plugin/futr3d/configs/lidar_cam/unibev_200x200_spatial_adaptive_fusion.py \
    --seed 0 \
    --cfg-options runner.max_epochs=12 \
    data.samples_per_gpu=8 \
    checkpoint_config.max_keep_ckpts=12 \
    load_from='checkpoint/lidar_0075_cam_vov.pth' \
    --launcher pytorch

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    tools/train.py \
    plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py \
    --seed 0 \
    --cfg-options runner.max_epochs=6 data.samples_per_gpu=15\
    --launcher pytorch

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