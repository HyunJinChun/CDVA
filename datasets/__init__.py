# ------------------------------------------------------------------------
# SeqFormer
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection
from .ytvos import YTVOSDataset as YTVOSDataset
from .miseang import MiseangDataset as MiseangDataset  ##

from .coco import build as build_coco
from .coco2seq import build as build_seq_coco
from .concat_dataset import build as build_joint
from .ytvos import build as build_ytvs

from .miseang import build as build_miseang  ##
from .miseang_using_getitem import build as build_miseang_using_getitem ## 220526
from .miseang_img import build as build_miseang_img ## 220623
from .miseang_target_base import build as build_miseang_target_base ## 220815
from .miseang_replace import build as build_miseang_replace ## 220830
from .miseang_insert import build as build_miseang_insert ## 220901
from .miseang_VideoMix import build as build_miseang_VideoMix
from .miseang_ObjectMix import build as build_miseang_ObjectMix
from .miseang_VideoMix_MAX2 import build as build_miseang_VideoMix_MAX2
from .miseang_ObjectMix_MAX import build as build_miseang_ObjectMix_MAX
from .miseang_MAX import build as build_miseang_B_AUG_MAX
from .miseang_CDVA_MAX import build as build_miseang_CDVA_MAX

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    if isinstance(dataset, YTVOSDataset):
        return dataset.ytvos


### build_type only works for YoutubeVIS ###
## + Miseang
def build_dataset(image_set, args):
    if args.dataset_file == 'YoutubeVIS':
        return build_ytvs(image_set, args)

    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'Seq_coco':
        return build_seq_coco(image_set, args)
    if args.dataset_file == 'jointcoco':
        return build_joint(image_set, args)

    if args.dataset_file == 'Miseang' and args.VideoMix is True and args.MAX_CLIPS_PER_CLASS is True:
        return build_miseang_VideoMix_MAX2(image_set, args)
    if args.dataset_file == 'Miseang' and args.ObjectMix is True and args.MAX_CLIPS_PER_CLASS is True:
        return build_miseang_ObjectMix_MAX(image_set, args)
    if args.dataset_file == 'Miseang' and args.manual_cp is True and args.insert is True and args.MAX_CLIPS_PER_CLASS is True:
        return build_miseang_CDVA_MAX(image_set, args)
    if args.dataset_file == 'Miseang' and args.simple_cp is True and args.MAX_CLIPS_PER_CLASS is True:
        return build_miseang_B_AUG_MAX(image_set, args)

    #==================

    if args.dataset_file == 'Miseang' and args.ObjectMix is True:
        return build_miseang_ObjectMix(image_set, args)
    if args.dataset_file == 'Miseang' and args.VideoMix is True:
        return build_miseang_VideoMix(image_set, args)
    if args.dataset_file == 'Miseang' and args.manual_cp is True and args.insert is True:
        return build_miseang_insert(image_set, args)
    if args.dataset_file == 'Miseang' and args.manual_cp is True and args.replace is True:
        return build_miseang_replace(image_set, args)
    if args.dataset_file == 'Miseang' and args.manual_cp is True: # Manual Method
        # return build_miseang_using_getitem(image_set, args)
        return build_miseang_target_base(image_set, args)
    if args.dataset_file == 'Miseang' and (args.img_copy_paste_diff is True or args.img_copy_paste_same is True): # Img Based
        return build_miseang_img(image_set, args)
    if args.dataset_file == 'Miseang': # No Transforms, Only Transforms, Simple CP, Simple CP One
        return build_miseang(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')

