python -m torch.distributed.launch tools/train.py \
    --nproc_per_node=2 \
    plugin/fudet/configs/lidar_cam/lidar_0075v_cam_vov.py \