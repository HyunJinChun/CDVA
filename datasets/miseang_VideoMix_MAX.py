# ------------------------------------------------------------------------
# SeqFormer data loader
# ------------------------------------------------------------------------
# Modified from Deformable VisTR (https://github.com/Epiphqny/VisTR)
# ------------------------------------------------------------------------
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import datasets.transforms_clip as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
import math
import time
import numpy as np
# from datasets import transforms_copy_paste_incomplete
from datasets import transforms_VideoMix_MAX

class MiseangDataset:
    def __init__(self, img_folder, ann_file, transforms, no_transforms, return_masks, num_frames, copy_paste_transforms):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.no_transforms = no_transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.copy_paste_transforms = copy_paste_transforms

        self.prepare = ConvertCocoPolysToMaskFace(return_masks)
        # self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))
        print('\n video num:', len(self.vid_ids), '  clip num:', len(self.img_ids))
        print('\n')

        # ============

        self.REQUIRED_CATEGORIES = [2, 3, 4, 5, 6]
        self.MAX_CLIPS_PER_CLASS = 2700
        self.aug_vids = {}
        self.countClips = 0 ######## 될까>>?  # 생성한 모든 데이터를 사용할 수 있도록 idx 불러올 때 이용
        self.requiredClips = []

        ## 장그래 개수 세는 용
        # vid_list, num_frames = self.get_clip_with_targetClass(1)  # targetClass가 속한 클립들
        # print(len(vid_list))

        # count_list = []  # 데이터 세는 용
        for targetClass in self.REQUIRED_CATEGORIES:
            vid_list, num_frames = self.get_clip_with_targetClass(targetClass)  # targetClass가 속한 클립들
            # count_list.append((targetClass, len(vid_list), requiredClips))
            none_vid_list = self.get_clip_without_targetClass(targetClass)  # 부족 인물 클래스들에 속한 어떤 인물도 포함되어 있지 않은 클립들
            requiredClips = self.MAX_CLIPS_PER_CLASS - len(vid_list)  # 보강을 통해 추가로 채워야 할 클립 수
            self.requiredClips.append(requiredClips)
            self.aug_vids[targetClass] = {"vid_list": vid_list, "none_vid_list": none_vid_list}
            self.countClips += requiredClips
            print('class', targetClass, ':', len(vid_list), num_frames)
            print('---')

    def __len__(self):
        return len(self.img_ids)
        # return self.countClips

    def __getitem__(self, idx):
        print('idx=',idx)
        if idx < self.requiredClips[0]:  # 2. 오상식
            if len(self.aug_vids[2]['vid_list']) < idx:
                target_vid_id = self.vid_infos[self.aug_vids[2]['vid_list'][idx]]['id']
                vid = self.aug_vids[2]['vid_list'][self.requiredClips[0] - len(self.aug_vids[2]['vid_list'])]
            else:
                target_vid_id = self.vid_infos[self.aug_vids[2]['vid_list'][idx]]['id']
                vid = self.aug_vids[2]['vid_list'][idx]
            b_vid_id = self.vid_infos[self.aug_vids[2]['none_vid_list'][idx]]['id']
            b_vid = self.aug_vids[2]['none_vid_list'][idx]
            targetClass = 2
        elif self.requiredClips[0] <= idx < self.requiredClips[1]:  # 3. 김동식
            if len(self.aug_vids[3]['vid_list']) < idx:
                target_vid_id = self.vid_infos[self.aug_vids[3]['vid_list'][idx]]['id']
                vid = self.aug_vids[3]['vid_list'][self.requiredClips[1] - len(self.aug_vids[3]['vid_list'])]
            else:
                target_vid_id = self.vid_infos[self.aug_vids[3]['vid_list'][idx]]['id']
                vid = self.aug_vids[3]['vid_list'][idx]
            b_vid_id = self.vid_infos[self.aug_vids[3]['none_vid_list'][idx]]['id']
            b_vid = self.aug_vids[3]['none_vid_list'][idx]
            targetClass = 3
        elif sum(self.requiredClips[0:2]) <= idx < self.requiredClips[2]:  # 4. 장백기
            if len(self.aug_vids[4]['vid_list']) < idx:
                target_vid_id = self.vid_infos[self.aug_vids[4]['vid_list'][idx]]['id']
                vid = self.aug_vids[4]['vid_list'][self.requiredClips[2] - len(self.aug_vids[4]['vid_list'])]
            else:
                target_vid_id = self.vid_infos[self.aug_vids[4]['vid_list'][idx]]['id']
                vid = self.aug_vids[4]['vid_list'][idx]
            b_vid_id = self.vid_infos[self.aug_vids[4]['none_vid_list'][idx]]['id']
            b_vid = self.aug_vids[4]['none_vid_list'][idx]
            targetClass = 4
        elif sum(self.requiredClips[0:3]) <= idx < self.requiredClips[3]:  # 5. 안영이
            if len(self.aug_vids[5]['vid_list']) < idx:
                target_vid_id = self.vid_infos[self.aug_vids[5]['vid_list'][idx]]['id']
                vid = self.aug_vids[5]['vid_list'][self.requiredClips[3] - len(self.aug_vids[5]['vid_list'])]
            else:
                target_vid_id = self.vid_infos[self.aug_vids[5]['vid_list'][idx]]['id']
                vid = self.aug_vids[5]['vid_list'][idx]
            b_vid_id = self.vid_infos[self.aug_vids[5]['none_vid_list'][idx]]['id']
            b_vid = self.aug_vids[5]['none_vid_list'][idx]
            targetClass = 5
        else:  # 6. 한석율
            if len(self.aug_vids[6]['vid_list']) < idx:
                target_vid_id = self.vid_infos[self.aug_vids[6]['vid_list'][idx]]['id']
                vid = self.aug_vids[6]['vid_list'][self.requiredClips[4] - len(self.aug_vids[6]['vid_list'])]
            else:
                target_vid_id = self.vid_infos[self.aug_vids[6]['vid_list'][idx]]['id']
                vid = self.aug_vids[6]['vid_list'][idx]
            b_vid_id = self.vid_infos[self.aug_vids[6]['none_vid_list'][idx]]['id']
            b_vid = self.aug_vids[6]['none_vid_list'][idx]
            targetClass = 6

        inds = list(range(self.num_frames)) ## [0,1,2,3,4]

        target_img = []
        for i in inds:
            img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][i])
            target_img.append(Image.open(img_path).convert('RGB'))
        ann_ids = self.ytvos.getAnnIds(vidIds=[target_vid_id])
        target_ann = self.ytvos.loadAnns(ann_ids)
        target_ann = {'video_id': vid, 'annotations': target_ann}
        target_ann = self.prepare(target_img[0], target_ann, inds, sample_inds=inds)

        background_img = []
        for i in inds:
            img_path = os.path.join(str(self.img_folder), self.vid_infos[b_vid]['file_names'][i])
            background_img.append(Image.open(img_path).convert('RGB'))
        ann_ids = self.ytvos.getAnnIds(vidIds=[b_vid_id])
        background_ann = self.ytvos.loadAnns(ann_ids)
        background_ann = {'video_id': b_vid, 'annotations': background_ann}
        background_ann = self.prepare(background_img[0], background_ann, inds, sample_inds=inds)

        augClip_img, augClip_target = self.copy_paste_transforms(background_img, background_ann, target_img, target_ann, self.num_frames, targetClass)  # VideoMix CopyPaste

        augClip_target['boxes'] = augClip_target['boxes'].clamp(1e-6)
        return torch.cat(augClip_img, dim=0), augClip_target


    def get_clip_with_targetClass(self, targetClass):
        '''
        :param targetClass:
        :return:
        '''
        vid_list = []
        total_num = 0

        for idx, (vid, frame_id) in enumerate(self.img_ids):
            vid_id = self.vid_infos[vid]['id']
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)
            for t in target:
                if targetClass == t['category_id']:
                    vid_list.append(vid)
                    if 'detected_face' in t.keys() and t['detected_face'][frame_id] is True:
                        total_num += 1

        return vid_list, total_num

    def get_clip_without_targetClass(self, targetClass):
        '''
        :param targetClass:
        :return:
        '''
        vid_list = []

        for idx, (vid, frame_id) in enumerate(self.img_ids):
            flag = False

            vid_id = self.vid_infos[vid]['id']
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)
            for t in target:
                if targetClass == t['category_id']:
                    flag = True
            if flag is False:
                        vid_list.append(vid)

        return vid_list




def convert_coco_poly_to_mask(segmentations, height, width, is_crowd):
    masks = []
    for i, seg in enumerate(segmentations):
        if not seg:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            if not is_crowd[i]:
                seg = coco_mask.frPyObjects(seg, height, width)
            mask = coco_mask.decode(seg)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, target_inds, sample_inds):
        w, h = image.size
        video_id = target['video_id']
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        video_len = len(anno[0]['bboxes'])
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []

        # add valid flag for bboxes
        for i, ann in enumerate(anno):
            classes.append(ann["category_id"])
            for id in target_inds:
                bbox = ann['bboxes'][sample_inds[id]]
                areas = ann['areas'][sample_inds[id]]
                segm = ann['segmentations'][sample_inds[id]]
                # clas = ann["category_id"]
                # for empty boxes
                if bbox is None:
                    bbox = [0, 0, 0, 0]
                    areas = 0
                    valid.append(0)

                else:
                    valid.append(1)
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes.append(bbox)
                area.append(areas)
                segmentations.append(segm)
                # classes.append(clas)
                iscrowd.append(crowd)

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w, iscrowd)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks

        image_id = [sample_inds[id] + video_id * 1000 for id in target_inds]
        image_id = torch.tensor(image_id)
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area)
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return target


class ConvertCocoPolysToMaskFace(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, target_inds, sample_inds):
        w, h = image.size
        video_id = target['video_id']
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        video_len = len(anno[0]['bboxes']) # no usage

        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []

        # add valid flag for bboxes
        ## changed for face_detected
        for i, ann in enumerate(anno):
            category = False ##
            for id in target_inds:
                bbox = ann['bboxes'][sample_inds[id]]
                areas = ann['areas'][sample_inds[id]]
                segm = ann['segmentations'][sample_inds[id]]

                if 'detected_face' in ann.keys():
                    face_detected = ann['detected_face'][sample_inds[id]]  ## get face_detected

                    if face_detected and ann["category_id"] != 7:
                        category = True

                # for empty boxes
                if bbox is None:
                    bbox = [0, 0, 0, 0]
                    areas = 0
                    valid.append(0)
                else:
                    valid.append(1)
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes.append(bbox)
                area.append(areas)
                segmentations.append(segm)
                iscrowd.append(crowd)
            if category: ##
                classes.append(ann["category_id"])
            else:
                classes.append(7)

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w, iscrowd)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks

        image_id = [sample_inds[id] + video_id * 1000 for id in target_inds]
        image_id = torch.tensor(image_id)
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area)
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [296, 328, 360, 392]
    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=768),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=768),
                    T.Check(),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_miseang_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [296, 328, 360, 392]
    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.Compose([
                T.RandomResize(scales, max_size=768),
                T.Check(),
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def no_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([
        normalize,
    ])


def make_copy_paste_transforms():
    cp_normalize = T.Compose([
        transforms_VideoMix_MAX.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    normalize = normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]
    flip_transforms = T.RandomHorizontalFlip()
    resize_transforms = transforms_VideoMix_MAX.RandomResize(max_size=768)
    check = T.Check()
    cp_transforms = transforms_VideoMix_MAX.CopyAndPaste()
    cp_compose = transforms_VideoMix_MAX.CopyPasteCompose(flip_transforms, resize_transforms, check, cp_transforms,
                                                        cp_normalize, normalize, scales)

    return cp_compose


def build(image_set, args):
    root = Path(args.miseang_path)
    assert root.exists(), f'provided Miseang path {root} does not exist'

    if args.dataset_file == 'Miseang':
        mode = 'instances'
        PATHS = {
            # "train": (root / "train/frame", root / "annotations" / f'{mode}_train_sub_face_5frames.json'),
            # # "train": (root / "train/60frame", root / "annotations" / f'{mode}_train_sub_face.json'),
            # "val": (root / "val/frame", root / "annotations" / f'{mode}_val_sub.json'),
            "train": (root / "train/frame", root / 'train/json/train.json'),
            "val": (root / "validation/frame", root / 'validation/json/validation.json'),
        }
        img_folder, ann_file = PATHS[image_set]
        print('use Miseang dataset - miseang_VideoMix_MAX.py')

        print('VideoMix :', args.VideoMix, 'MAX :', args.MAX_CLIPS_PER_CLASS)

        if args.VideoMix is True and image_set == 'train':
            copy_paste_tfs = make_copy_paste_transforms()
            print('make VideoMix transforms')
        else:
            copy_paste_tfs = None

        dataset = MiseangDataset(img_folder, ann_file, transforms=make_miseang_transforms(image_set), no_transforms=no_transforms(), return_masks=args.masks,
                                 num_frames=args.num_frames, copy_paste_transforms=copy_paste_tfs)

    return dataset