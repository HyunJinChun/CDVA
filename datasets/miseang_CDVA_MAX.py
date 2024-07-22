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
from datasets import transforms_copy_paste_CDVA_MAX


class MiseangDataset:
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames, copy_paste_transforms, category, size, position, human_size):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.copy_paste_transforms = copy_paste_transforms
        self.category = category
        self.size = size
        self.position = position
        self.human_size = human_size

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
        self.MAX_CLIPS_PER_CLASS = 1800
        self.aug_vids = {}
        self.requiredClips = []  # 보강을 통해 추가로 채워야 할 클립 수 517, 944 ...
        # self.existClips = [997, 721, 590, 687, 24] # 2, 3, 4, 5, 6 순서로 존재하는 클립의 개수
        self.targetClass = 2  # 첫 시작 : 오상식
        self.countClass = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

        self.add_frame = 0
        for j, targetClass in enumerate(self.REQUIRED_CATEGORIES):
            v_info, add_frame = self.get_clips_with_targetClass(targetClass)  # targetClass가 속한 클립들
            requiredClips = add_frame
            self.requiredClips.append(requiredClips)
            self.add_frame += add_frame
            self.aug_vids[targetClass] = {"vid_list": v_info}

        self.none_vid_list, self.none_vid_len, self.only_none_vid = self.get_clips_without_targetClasses(
            self.add_frame)  # 부족 인물 클래스들에 속한 어떤 인물도 포함되어 있지 않은 클립들
        self.augClips = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # targetClass 삽입 몇 번 됐는지
        self.add_clips = []  # 더 추가할 경우
        self.a_flag = False
        self.t_flag = False

        # print('target:', self.targetClass, 'requiredClips: ', self.requiredClips)
        # print('none_vid_list', len(self.none_vid_list))

    def __len__(self):
        return len(self.img_ids) + 3000 ##1304

    def __getitem__(self, idx):
        instance_check = False

        while not instance_check:
            ## 원하는 클래스의 인물들이 한 명도 없다면 Target Clip 삽입하기
            # if idx % 1000 == 0:
            #     # print(self.countClass)
            if idx < len(self.img_ids):
                vid, frame_id = self.img_ids[idx]
            else:  # 추가 보강
                vid, frame_id = self.none_vid_list[idx - len(self.img_ids)]
            vid_id = self.vid_infos[vid]['id']
            background_img = []
            vid_len = len(self.vid_infos[vid]['file_names'])
            inds = list(range(self.num_frames))  ## [0,1,2,3,4]
            num_frames = self.num_frames  ## 5 (T from miseang.sh)
            # random sparse sample
            sample_indx = [frame_id]  ## [32]
            # local sample
            samp_id_befor = randint(1, 3)
            samp_id_after = randint(1, 3)
            local_indx = [max(0, frame_id - samp_id_befor), min(vid_len - 1, frame_id + samp_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])

                # 전체 frame의 수가 5개를 초과하는 경우에 대한 처리 방법
                elif vid_len >= global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            for j in range(self.num_frames):
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[j]])
                background_img.append(Image.open(img_path).convert('RGB'))
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            background_ann = self.ytvos.loadAnns(ann_ids)

            background_ann = {'video_id': vid, 'annotations': background_ann}
            background_inds = inds
            background_ann = self.prepare(background_img[0], background_ann, background_inds,
                                          sample_inds=sample_indx)

            #### ==== test 확인용 ==== ####
            if len(background_ann['labels']) != 0:
                instance_check = True
            else:
                np.random.seed(int(time.time()))
                idx = random.randint(0, self.__len__()-1)

        # 원하는대로 보강되는지 확인용
        for label in background_ann['labels']:
            self.countClass[label.tolist()] += 1

        if idx >= len(self.img_ids) and sum(self.requiredClips) == 0:  # MAX_ 도달 안했으면 될 때까지
            f = False
            for key, value in self.countClass.items():
                if 2 <= key <= 6:
                    self.add_clips.append(value)
                    if value < self.MAX_CLIPS_PER_CLASS and f is False:
                        self.targetClass = key
                        f = True
            self.a_flag = True

        if vid not in self.only_none_vid or (
                sum(self.requiredClips) == 0 and sum(self.add_clips) == 9000):  # 배경에 targetClass 인물이 있거나 보강이 완료되면 CP X
            background_img, background_ann = self._transforms(background_img, background_ann, num_frames)
            background_ann['boxes'] = background_ann['boxes'].clamp(1e-6)
            return torch.cat(background_img, dim=0), background_ann

        if self.a_flag is True and self.targetClass == 7:
            background_img, background_ann = self._transforms(background_img, background_ann, num_frames)
            background_ann['boxes'] = background_ann['boxes'].clamp(1e-6)
            return torch.cat(background_img, dim=0), background_ann

        # 삽입할 target clip 찾기
        if self.a_flag is True:
            target_vid = self.aug_vids[self.targetClass]['vid_list'][self.add_clips[self.targetClass] - 1]
        else:
            target_vid = self.aug_vids[self.targetClass]['vid_list'][self.augClips[self.targetClass]]
        target_img, target_ann = self.get_target_info(target_vid)

        # background_img, background_ann = self._transforms(background_img, background_ann, num_frames)
        # background_ann['boxes'] = background_ann['boxes'].clamp(1e-6)
        # return torch.cat(background_img, dim=0), background_ann

        ########### 확인용으로 주석처리

        augClip_img, augClip_target = self.copy_paste_transforms(background_img, background_ann, target_img,
                                                                 target_ann, self.num_frames,
                                                                 self.targetClass)  # CDVA CopyPaste

        augClip_target['boxes'] = augClip_target['boxes'].clamp(1e-6)
        return torch.cat(augClip_img, dim=0), augClip_target

    def get_target_info(self, target_vid):
        instance_check = False

        while not instance_check:
            vid, frame_id = target_vid
            vid_id = self.vid_infos[vid]['id']
            target_img = []
            vid_len = len(self.vid_infos[vid]['file_names'])
            inds = list(range(self.num_frames))  ## [0,1,2,3,4]
            num_frames = self.num_frames  ## 5 (T from miseang.sh)
            # random sparse sample
            sample_indx = [frame_id]  ## [32]
            # local sample
            samp_id_befor = randint(1, 3)
            samp_id_after = randint(1, 3)
            local_indx = [max(0, frame_id - samp_id_befor), min(vid_len - 1, frame_id + samp_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            # 5개 frame을 넘는 경우에 대해서 처리 필요
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >= global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            for j in range(self.num_frames):
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[j]])
                target_img.append(Image.open(img_path).convert('RGB'))

            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target_ann = self.ytvos.loadAnns(ann_ids)

            target_ann = {'video_id': vid, 'annotations': target_ann}
            target_inds = inds  ## [0,1,2,3,4]
            target_ann = self.prepare(target_img[0], target_ann, target_inds, sample_inds=sample_indx)

            if self.targetClass in target_ann['labels']:
                instance_check = True
            else:
                np.random.seed(int(time.time()))
                idx = random.randint(0, len(self.aug_vids[self.targetClass]['vid_list']) - 1)
                target_vid = self.aug_vids[self.targetClass]['vid_list'][idx]

        if target_ann is not None:
            target_idx = self.REQUIRED_CATEGORIES.index(self.targetClass)
            if self.t_flag is False:
                self.requiredClips[target_idx] -= 1
            self.augClips[self.targetClass] += 1  # 보강 확인용
            self.countClass[self.targetClass] += 1  # 전체 데이터 확인용

            if self.requiredClips[target_idx] == 0:
                self.targetClass += 1
                # print(' target:', self.targetClass, 'requiredClips: ', self.requiredClips)
                if sum(self.requiredClips) == 0:  ## 추가
                    # print(self.countClass)
                    self.TargetClass = 2
                    self.t_flag = True

            if sum(self.requiredClips) == 0 and sum(self.add_clips) != 0:
                self.add_clips[target_idx] += 1

        return target_img, target_ann

    def get_clips_with_targetClass(self, targetClass):
        '''
        :param targetClass:
        :param requiredClips:
        :return:
        '''
        vid_list = []
        # vid_yn = [] # vid 중복 여부 확인

        for idx, (vid, frame_id) in enumerate(self.img_ids):
            vid_id = self.vid_infos[vid]['id']
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)
            vid_len = len(self.vid_infos[vid]['file_names'])
            for i, t in enumerate(target):
                if targetClass == t['category_id']:  # and vid not in vid_yn
                    vid_list.append((vid, frame_id))
                    # vid_yn.append(vid)
        random.shuffle(vid_list)
        # print('vid-list :', len(vid_list), end=' -> ')
        before_len = len(vid_list)

        for i in range(self.MAX_CLIPS_PER_CLASS - len(vid_list)):
            vid_list.append(vid_list[i])

        # print('append vid-list :', len(vid_list))

        return vid_list, len(vid_list) - before_len

    def get_clips_without_targetClasses(self, add_frame):
        '''
        :return:
        '''
        vid_list = []
        only_vid_list = []

        for idx, (vid, frame_id) in enumerate(self.img_ids):
            flag = False  # targetClasses가 없음
            vid_id = self.vid_infos[vid]['id']
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)
            for t in target:
                if t['category_id'] in self.REQUIRED_CATEGORIES:
                    flag = True  # targetClasses가 있음
            if not flag:
                vid_list.append((vid, frame_id))
                if vid not in only_vid_list:
                    only_vid_list.append(vid)

        # print('none-vid-list :', len(vid_list), end=' -> ')
        before_len = len(vid_list)

        for i in range(add_frame - before_len):
            vid_list.append(vid_list[i])

        # print('append none-vid-list :', len(vid_list))

        return vid_list, before_len, only_vid_list

    def count_class(self):
        print(self.countClass)

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


def make_copy_paste_transforms(size, position, GD, PD, all_frames, foreground):
    cp_normalize = T.Compose([
        transforms_copy_paste_CDVA_MAX.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    normalize = normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]
    flip_transforms = T.RandomHorizontalFlip()
    resize_transforms = transforms_copy_paste_CDVA_MAX.RandomResize(max_size=768)
    check = T.Check()
    hsv_transforms = transforms_copy_paste_CDVA_MAX.PhotometricDistort()
    cp_transforms = transforms_copy_paste_CDVA_MAX.CopyAndPaste()
    cp_compose = transforms_copy_paste_CDVA_MAX.CopyPasteCompose(flip_transforms, resize_transforms, check, hsv_transforms,
                                                                      cp_transforms, cp_normalize, normalize, scales, size, position, GD, PD, all_frames, foreground)

    return cp_compose


def build(image_set, args):
    root = Path(args.miseang_path)
    assert root.exists(), f'provided Miseang path {root} does not exist'

    if args.dataset_file == 'Miseang':
        mode = 'instances'
        PATHS = {
            "train": (root / "train/frame", root / 'train/json/train.json'),
            "val": (root / "validation/frame", root / 'validation/json/validation.json'),
        }
        img_folder, ann_file = PATHS[image_set]
        print('use Miseang dataset - CDVA_MAX.py')

        if args.manual_cp is True and image_set == 'train':
            copy_paste_tfs = make_copy_paste_transforms(size=args.size, position=args.position, GD=args.GD, PD=args.PD, all_frames=args.all_frames, foreground=args.foreground)
            print('make copy paste transforms - CDVA_MAX')
        else:
            copy_paste_tfs = None

        print('category :', args.category, 'human_size :', args.human_size, ', size :', args.size, ', position :', args.position)
        print('cal_foreground :', args.foreground)
        print('GD :', args.GD, ', PD :', args.PD)
        print()

        dataset = MiseangDataset(img_folder, ann_file, transforms=make_miseang_transforms(image_set), return_masks=args.masks,
                                 num_frames=args.num_frames, copy_paste_transforms=copy_paste_tfs, category=args.category,
                                 size=args.size, position=args.position, human_size=args.human_size)

    return dataset
