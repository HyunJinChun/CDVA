
# ------------------------------------------------------------------------
# Training script of SeqFormer
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import matplotlib.pyplot as plt
from datasets.transforms_copy_paste_incomplete import Denormalize


def get_args_parser():
    parser = argparse.ArgumentParser('SeqFormer', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int, nargs='+')
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--pretrain_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use: [resnet50, resnet101, resnext101_32x8d, swin_l_p4w12]")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_out_stride', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='Miseang') ## coco
    parser.add_argument('--coco_path', default='../coco', type=str)
    parser.add_argument('--ytvis_path', default='../ytvis', type=str)
    parser.add_argument('--miseang_path', default='../miseang_vis_dataset', type=str)  ##
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # evaluation options
    parser.add_argument('--dataset_type', default='original')
    parser.add_argument('--eval_types', default='')
    parser.add_argument('--visualize', default='')

    # multi-frame
    parser.add_argument('--num_frames', default=1, type=int, help='number of frames')

    parser.add_argument('--rel_coord', default=False, action='store_true')

    parser.add_argument('--jointfinetune', action='store_true',
                        help="keep all weight when load joint training model")

    # Video Context Augmentation
    parser.add_argument('--video_transform', action='store_true')  # Video Transform 유무 (좌우반전, resize)

    parser.add_argument('--simple_cp', action='store_true') # simple cp 개체들 여러 개 **B-Aug**
    parser.add_argument('--simple_cp_one', action='store_true') # simple cp 개체 1개만

    parser.add_argument('--img_copy_paste_same', action='store_true') # 동일한 프레임 5장 -> 단일 프레임
    parser.add_argument('--img_copy_paste_diff', action='store_true') # 다른 프레임 5장 -> 독립된 프레임들

    parser.add_argument('--manual_cp', action='store_true')  # using_getitem에서 target clip의 빈 공간을 계산한 후 진행
    parser.add_argument('--category', action='store_true') # 인물 카테고리
    parser.add_argument('--human_size', action='store_true') # 인물 크기
    parser.add_argument('--size', action='store_true') # 최대 빈 영역 계산 - 120%
    parser.add_argument('--position', action='store_true') # 인물 위치

    parser.add_argument('--GD', action='store_true')  # Geometric Distortions : Flipping, Resize
    parser.add_argument('--PD', action='store_true')  # Photometric Distortions : Brightness, Contrast, Hue ...
    parser.add_argument('--all_frames', action='store_true') # MVDA : one frame or all frames
    parser.add_argument('--foreground', action='store_true')  # calculate foreground humans

    parser.add_argument('--replace', action='store_true') # Instance Replacement
    parser.add_argument('--insert', action='store_true')  # Instance Insertion **MVDA**

    parser.add_argument('--VideoMix', action='store_true')  # **VideoMix**
    parser.add_argument('--ObjectMix', action='store_true')  # **ObjectMix**

    parser.add_argument('--MAX_CLIPS_PER_CLASS', action='store_true')  # MAX_CLIPS_PER_CLASS


    return parser

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train = build_dataset(image_set='train', args=args)
    save_idx = 0
    de_normalization = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for save_idx, (sample, target) in enumerate(dataset_train):
        # print(save_idx)
        # sample = de_normalization(sample)

        if save_idx == 31:
            break

        sample_list = torch.split(sample, 3)

        for i, img in enumerate(sample_list):
            img = de_normalization(img)
            img = img.transpose(1, 0).transpose(2, 1)

            if save_idx >= 29:
                plt.imshow(img)
                plt.axis('off')
                plt.savefig('../vis_test/hyeon/test_copy_paste/test_230103/ObjectMix/MAX/test_{}_{}.png'.format(save_idx, i), bbox_inches='tight', pad_inches=0)
                print('test_{}_{}.png'.format(save_idx, i))
            # plt.show()
        print()

    '''
    print("Start training")
    train_stats = train_one_epoch(
        model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SeqFormer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
