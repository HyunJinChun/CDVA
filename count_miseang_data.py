
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
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import matplotlib.pyplot as plt
from datasets.transforms_copy_paste_incomplete import Denormalize

import datasets.samplers_miseang as samplers_miseang
import datasets.samplers as samplers
from tqdm import tqdm

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


def count_class(dataset_train):
    count_class_dict = {}
    instance_class_dict = {}

    for i in range(1, 8):
        count_class_dict[i] = 0
        instance_class_dict[i] = 0

    for save_idx, (sample, target) in enumerate(tqdm(dataset_train, desc='cnt_cls')):
        labels = target['labels'].tolist()
        for label in labels:
            instance_class_dict[label] += 1

        masks = torch.split(target['masks'], 5)
        for f, mask_clip in enumerate(masks):
            for mask_frame in mask_clip:
                if torch.count_nonzero(mask_frame) != 0:
                    count_class_dict[labels[f]] += 1

    return count_class_dict, instance_class_dict


def generate_graph(instance_class_dict, count_class_dict, mode='train', path='./graph_result'):
    category = ['janggeurae', 'ohsangsik', 'kimdongsik', 'jangbaekki', 'anyoungyi', 'hanseokyul', 'someone']

    if instance_class_dict is not None :
        # instance_class_dict
        x = np.arange(len(instance_class_dict.keys()))
        x_data = []
        y_data = []

        for ins_class in instance_class_dict.keys():
            x_data.append(category[ins_class-1])
            y_data.append(instance_class_dict[ins_class])

        bar = plt.bar(x, y_data)
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % height, ha='center', va='bottom', size=12)
        plt.xticks(x, x_data)
        plt.savefig('{}/{}_count_instance.png'.format(path, mode))
        plt.show()

    # count_class_dict
    x = np.arange(len(count_class_dict.keys()))
    x_data = []
    y_data = []

    for count_class in count_class_dict.keys():
        x_data.append(category[count_class-1])
        y_data.append(count_class_dict[count_class])

    bar = plt.bar(x, y_data)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % height, ha='center', va='bottom', size=12)
    plt.xticks(x, x_data)
    plt.savefig('{}/{}_count_class.png'.format(path, mode))
    plt.show()


def main(args):
    start = time.time()

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ## 데이터 집합 데이터 개수 그래프 생성 ##
    # print('====> Start build_dataset')
    # dataset_train = tqdm(build_dataset(image_set='train', args=args), desc='build_dataset')
    #
    # print('====> Start count_class')
    # count_class_dict, instance_class_dict = count_class(dataset_train)
    #
    # print('====> Start generate_graph')
    # generate_graph(instance_class_dict=instance_class_dict, count_class_dict=count_class_dict, mode='VideoMix_MAX2', path='./graph_result/220917')

    dataset_train = build_dataset(image_set='train', args=args)

    for save_idx, (sample, target) in enumerate(tqdm(dataset_train, desc='load_dataset')):
        continue

    print()
    print('count_class: ', dataset_train.count_class())
    print('** Time :', time.time() - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SeqFormer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
