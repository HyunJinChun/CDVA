# ------------------------------------------------------------------------
# SeqFormer data loader
# ------------------------------------------------------------------------
# Modified from Deformable VisTR (https://github.com/Epiphqny/VisTR)
# ------------------------------------------------------------------------
import copy
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
from datasets import transforms_copy_paste_insert
from tqdm import tqdm


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

        # 220815 보강 데이터 개수 지정
        count_class_dict = {}
        for i in range(1, 8):
            count_class_dict[i] = 0
        self.count_class = count_class_dict
        self.data_max_num = 13000

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
        print('data max num:', self.data_max_num) ##
        print('\n')


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # ****** target : 논문 상 타겟 클립을 의미, background_target : 논문 상 배경 클립을 의미
        instance_check = False
        while not instance_check:
            vid,  frame_id = self.img_ids[idx] ## ex) vid:0, frame_id:32
            vid_id = self.vid_infos[vid]['id']
            img = []
            vid_len = len(self.vid_infos[vid]['file_names'])
            inds = list(range(self.num_frames)) ## [0,1,2,3,4]
            num_frames = self.num_frames ## 5 (T from miseang.sh)
            # random sparse sample
            sample_indx = [frame_id] ## [32]
            #local sample
            samp_id_befor = randint(1,3)
            samp_id_after = randint(1,3)
            local_indx = [max(0, frame_id - samp_id_befor), min(vid_len - 1, frame_id + samp_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)]+all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)),global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])

                # 전체 frame의 수가 5개를 초과하는 경우에 대한 처리 방법
                elif vid_len >=global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len),global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    select_id = random.sample(range(vid_len),global_n - vid_len)+list(range(vid_len))
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            for j in range(self.num_frames):
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[j]])
                img.append(Image.open(img_path).convert('RGB'))
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)

            target = {'video_id': vid, 'annotations': target}
            target_inds = inds ## [0,1,2,3,4]
            target = self.prepare(img[0], target, target_inds, sample_inds=sample_indx)

            # 데이터 보강할 클래스
            valid_category = [2, 3, 4, 5, 6]

            # 인물 중 장그래와 someone을 제외한 인물들이 있는지 확인
            instance_choice = []

            for i, label in enumerate(target['labels']):
                if self.category is True:
                    if label in valid_category and self.count_class[label.tolist()] <= self.data_max_num:
                        instance_choice.append(i)
                else:
                    instance_choice.append(i)

            if not instance_choice: # 원하는 클래스의 인물이 없으면 클립 재선택
                idx = random.randint(0, self.__len__() - 1)
                continue

            # # 적용 확률
            r = random.random()
            p = 0.5

            # if self._transforms is not None and self.copy_paste_transforms is not None and r >= p: # 50% 확률
            if self._transforms is not None and self.copy_paste_transforms is None: # 100% 확률
                img, target = self._transforms(img, target, num_frames)

            # if self._transforms is not None and self.copy_paste_transforms is not None and r < p: # 50% 확률
            if self._transforms is not None and self.copy_paste_transforms is not None: # 100% 확률
                ## 220608 재수정 - blank_space를 여기서 계산하면 resize, flip의 문제가 발생하여 transforms에서 하도록 변경함
                background_idx = random.randint(0, self.__len__() - 1)  # background clip 랜덤 선택
                target = self.split_target(target)  # num_frames로 boxes랑 masks 나눴음 -> 나중에 copy_paste_transforms에서 다시 stack함

                background_img, background_target, target = self.get_background_target(background_idx, target, instance_choice) # 빈 공간을 고려하지 않고 background clip 인물의 크기 정보만을 이용한 get_background_target


                # 3. background clip에 target clip의 tracklet 삽입
                img, target = self.copy_paste_transforms(background_img, background_target, img, target, num_frames)

            if len(target['labels']) == 0: # None instance
                idx = random.randint(0, self.__len__()-1)
            else:
                # 학습 시 사용되는 등장인물별 데이터 개수 세기
                labels = target['labels'].tolist()
                masks = torch.split(target['masks'], 5)
                for f, mask_clip in enumerate(masks):
                    for mask_frame in mask_clip:
                        if torch.count_nonzero(mask_frame) != 0:
                            self.count_class[labels[f]] += 1
                instance_check = True

        target['boxes'] = target['boxes'].clamp(1e-6)
        return torch.cat(img, dim=0), target

    # num_frames로 boxes랑 masks 나누기
    def split_target(self, target):
        target_key = ['boxes', 'masks']
        target_list_dict = {}

        for key in target_key:
            instance_cnt = int(len(target[key]) / self.num_frames)
            target_list_dict[key] = []

            for i in range(self.num_frames):
                target_list_dict[key].append([])
                for j in range(instance_cnt):
                    target_list_dict[key][i].append(target[key][i + self.num_frames * j])

                target_list_dict[key][i] = torch.stack(target_list_dict[key][i], dim=0)

        # transforms_copy_paste에서 valid, area, iscrowd는 주석 처리되어 있으므로 여기서도 주석 처리함
        target_list_dict['labels'] = target['labels']
        target_list_dict['image_id'] = target['image_id']
        # target_list_dict['valid'] = target['valid']
        # target_list_dict['area'] = target['area']
        # target_list_dict['iscrowd'] = target['iscrowd']
        target_list_dict['orig_size'] = target['orig_size']
        target_list_dict['size'] = target['size']

        return target_list_dict

    # background clip을 선택하는 get_background_target - blank_space 고려 X, background의 인물 크기만 고려 후 선택
    def get_background_target(self, idx, target_list_dict, instance_choice):
        instance_check = False
        while not instance_check:
            vid, frame_id = self.img_ids[idx]  ## ex) vid:0, frame_id:32
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
                background_img.append(Image.open(img_path).convert('RGB'))
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            background_target = self.ytvos.loadAnns(ann_ids)

            background_target = {'video_id': vid, 'annotations': background_target}
            target_inds = inds  ## [0,1,2,3,4]
            background_target = self.prepare(background_img[0], background_target, target_inds, sample_inds=sample_indx)

            if self.category is False: ## a.k.a ObjectMix (모든 인물, 원래 자리에)
                return background_img, background_target, target_list_dict

            # target 인물이 1명만 있을 때, background에 동일 인물이 있으면 필터링 (여러 명이면, select_instance에서 뭐가 선택될 지 모르니 그 이후에 필터링)
            invalid_category = [target_list_dict['labels'][i].tolist() for i in instance_choice]
            background_category = background_target['labels'].tolist()

            if (len(invalid_category) == 1 and invalid_category[0] in background_category) or len(background_category) == 0:
                idx = random.randint(0, self.__len__() - 1)
                continue

            background_target = self.split_target(background_target)

            # background clip과 target clip 내 인물 고려 -> 출력 : target clip의 인물 1명 선택
            selected_target = self.select_instance(target_list_dict, background_target, instance_choice)

            # 적절한 인물이 없는 경우 다음으로 넘기기
            if selected_target is None:
                idx = random.randint(0, self.__len__() - 1)
                continue
            else:
                background_target = self.unsplit_target(background_target)
                instance_check = True

        return background_img, background_target, selected_target

    # select_instance_allbox와 동일, but blank_space 계산 X
    # def select_instance_allbox_no_blankspace(self, pasted_target_list_dict, target_list_dict, instance_choice):
    def select_instance(self, target_list_dict, background_list_dict, instance_choice):
        background_category = background_list_dict['labels'].tolist()
        target_key = ['boxes', 'masks']
        target_key_ = ['image_id', 'orig_size', 'size']

        # background clip 인물과 target clip 인물 크기 비교
        selected_idx = -1  # 적당한 인물이 선택되면 해당 index로 변경, for문 이후에도 -1이면 괜찮은 인물이 없다는 의미
        if self.human_size is True: # 인물 크기 비교
            selected = False
            background_instance_cnt = len(background_list_dict['boxes'][0]) # 인물이 1명일 때랑 여러 명일 때랑 형식이 달라서 구분해줘야 함
            image_width = target_list_dict['size'][1]

            # background clip 전체 프레임 속 모든 인물의 box width 비율 정보를 리스트에 저장
            b_width_list = [0] * background_instance_cnt # 평균 계산 후 image_width로 나눠서 비율 저장
            for frame, target in enumerate(background_list_dict['boxes']): # target[0-4] : tensor([target_instance_cnt, 4])
                for i, box in enumerate(target):
                    b_width_list[i] += box[2] - box[0] # target의 가로 추출
                    if frame == self.num_frames - 1:
                        b_width_list[i] /= self.num_frames
                        b_width_list[i] = (b_width_list[i] / image_width) * 100

            # target clip 전체 프레임 속 모든 인물의 box width 정보를 리스트에 저장
            t_width_list = [0] * len(instance_choice)
            for i, idx in enumerate(instance_choice):
                for frame in range(self.num_frames):
                    t_width_list[i] += target_list_dict['boxes'][frame][idx][2] - target_list_dict['boxes'][frame][idx][0]
                    if frame == self.num_frames - 1:
                        t_width_list[i] /= self.num_frames
                        t_width_list[i] = (t_width_list[i] / image_width) * 100

            for idx, t_width_percent in enumerate(t_width_list):
                for b_width_percent in b_width_list: # background 인물 크기 차례로 비교
                    # 2. 인물(들)이 있으면 target clip 내 인물(들)의 크기와 비교
                    # 비율 계산 후 5% 차이 이하인 경우를 선택
                    # if abs(b_width_percent - t_width_percent) <= 10:
                    if abs(b_width_percent - t_width_percent) <= 5:
                        # 2. END
                        selected_idx = instance_choice[idx]
                        # target 인물이 background에 있으면 선택 X
                        if target_list_dict['labels'][selected_idx].tolist() not in background_category :
                            selected = True
                            break
                    # 크기가 5% 차이 이상인 경우는 continue
                if selected:
                    break
        else: # 인물 크기 비교 X
            tmp = []
            for i in instance_choice:
                if target_list_dict['labels'][i] not in background_category:
                    tmp.append(i)
            if len(tmp) != 0:
                selected_idx = random.choice(tmp)

        if selected_idx == -1: # source clip에 적당한 인물이 없는 경우임
            return None

        # 최종 선택된 인물을 choice_target_list_dict로 return
        choice_target_list_dict = {}

        for key in target_key:
            # choice_target_list_dict[key] = []
            tmp_list = []

            for i in range(self.num_frames):
                tmp_list.append(target_list_dict[key][i][selected_idx])

            # 기존의 target 모습으로 변경
            tmp_list = torch.stack(tmp_list, dim=0)
            choice_target_list_dict[key] = tmp_list

        # label
        choice_target_list_dict['labels'] = []
        choice_target_list_dict['labels'].append(target_list_dict['labels'][selected_idx])
        choice_target_list_dict['labels'] = torch.stack(choice_target_list_dict['labels'], dim=0)

        # others
        for key_ in target_key_:
            choice_target_list_dict[key_] = target_list_dict[key_]

        return choice_target_list_dict

    # target unsplit
    def unsplit_target(self, target):
        target_key = ['boxes', 'masks']
        instance_cnt = len(target['labels'])

        for key in target_key:
            # choice_target_list_dict[key] = []
            tmp_list = []

            for cnt in range(instance_cnt):
                for i in range(self.num_frames):
                    tmp_list.append(target[key][i][cnt])
            tmp_list = torch.stack(tmp_list, dim=0)
            target[key] = tmp_list
        return target


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
        transforms_copy_paste_insert.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    normalize = normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]
    flip_transforms = T.RandomHorizontalFlip()
    resize_transforms = transforms_copy_paste_insert.RandomResize(max_size=768)
    check = T.Check()
    hsv_transforms = transforms_copy_paste_insert.PhotometricDistort()
    cp_transforms = transforms_copy_paste_insert.CopyAndPaste()
    cp_compose = transforms_copy_paste_insert.CopyPasteCompose(flip_transforms, resize_transforms, check, hsv_transforms,
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
        print('use Miseang dataset - miseang_insert.py')

        if args.manual_cp is True and image_set == 'train':
            copy_paste_tfs = make_copy_paste_transforms(size=args.size, position=args.position, GD=args.GD, PD=args.PD, all_frames=args.all_frames, foreground=args.foreground)
            print('make copy paste transforms - insert')
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