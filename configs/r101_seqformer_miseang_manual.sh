#!/usr/bin/env bash

set -x

python3 -u main_miseang.py \
    --dataset_file Miseang \
     --epochs 15 \
    --lr 2e-4 \
    --lr_drop 4 10\
    --batch_size 1 \
    --num_workers 2 \
    --coco_path ../coco \
    --ytvis_path ../ytvis \
    --miseang_path ../miseang_vis_data/vis_format \
    --num_queries 300 \
    --num_frames 5 \
    --with_box_refine \
    --masks \
    --rel_coord \
    --backbone resnet101 \
    --pretrain_weights weights/seqformer_r101_joint.pth \
    --output_dir 220918_r101_miseang/CDVA_MAX_220925 \
    --manual_cp \
    --category \
    --position \
    --GD \
    --all_frames \
    --insert \
    --MAX_CLIPS_PER_CLASS \
#    --VideoMix \
#    --video_transform \
#    --PD \
#    --simple_cp \
#    --simple_cp_one \
#    --img_copy_paste_diff \
#    --img_copy_paste_same \


