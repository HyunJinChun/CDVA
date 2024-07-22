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
from datasets import transforms_img_copy_paste

class MiseangDataset:
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames, copy_paste_transforms, status):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.copy_paste_transforms = copy_paste_transforms
        self.status = status

        self.prepare = ConvertCocoPolysToMaskFace(return_masks)
        self.prepare_diff = ConvertCocoPolysToMaskFace_diff(return_masks)
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

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
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

            if self._transforms is not None and self.copy_paste_transforms is None:
                img, target = self._transforms(img, target, num_frames)

            if self._transforms is not None and self.copy_paste_transforms is not None:
                np.random.seed(int(time.time()))
                paste_idx = random.randint(0, self.__len__() - 1)
                if self.status == 'same':
                    pasted_img, pasted_target = self.get_paste_img_target_same(paste_idx)
                elif self.status == 'diff':
                    pasted_img, pasted_target = self.get_paste_img_target_diff(paste_idx)
                img, target = self.copy_paste_transforms(img, target, pasted_img, pasted_target, num_frames)

            if len(target['labels']) == 0: # None instance
                idx = random.randint(0, self.__len__()-1)
            else:
                instance_check = True

        target['boxes'] = target['boxes'].clamp(1e-6)
        return torch.cat(img, dim=0), target

    def get_paste_img_target_same(self, idx):
        instance_check = False
        while not instance_check:
            vid, frame_id = self.img_ids[idx]  ## ex) vid:0, frame_id:32
            vid_id = self.vid_infos[vid]['id']
            img = []
            vid_len = len(self.vid_infos[vid]['file_names'])
            # inds = list(range(self.num_frames))  ## [0,1,2,3,4]
            inds = [0] * self.num_frames ## [0, 0, 0, 0, 0]
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

            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)

            for j in inds:
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[j]])
                img.append(Image.open(img_path).convert('RGB'))

            target = {'video_id': vid, 'annotations': target}
            target_inds = inds
            target = self.prepare(img[0], target, target_inds, sample_inds=sample_indx)

            if len(target['labels']) == 0: # None instance
                idx = random.randint(0, self.__len__()-1)
            else:
                instance_check = True

        return img, target

    def get_paste_img_target_diff(self, idx):
        frame_num = 0
        inst_label = -1
        flag = False

        vid_list = []
        sample_indx_list = []

        img = []
        bboxes = []
        segmentations = []
        areas = []
        detected_face = []

        new_target = {}

        while frame_num < self.num_frames:
            vid, frame_id = self.img_ids[idx]  ## ex) vid:0, frame_id:32
            vid_id = self.vid_infos[vid]['id']
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

            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)

            if inst_label == -1:
                target = random.choice(target)
                new_target['height'] = target['height']
                new_target['width'] = target['width']
                new_target['length'] = target['length']
                new_target['iscrowd'] = target['iscrowd']

                bboxes.append(target['bboxes'][sample_indx[0]])
                segmentations.append(target['segmentations'][sample_indx[0]])
                areas.append(target['areas'][sample_indx[0]])
                if 'detected_face' in target.keys():
                    detected_face.append(target['detected_face'][sample_indx[0]])
                else:
                    detected_face.append(None)
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[0]])
                img.append(Image.open(img_path).convert('RGB'))

                frame_num += 1
                inst_label = target['category_id']
                vid_list.append(vid)
                sample_indx_list.append(sample_indx[0])
            else:
                for i, t in enumerate(target):
                    if frame_num == self.num_frames:
                        flag = True
                        break
                    if inst_label == t['category_id'] and t['bboxes'][sample_indx[0]] is not None:
                        bboxes.append(t['bboxes'][sample_indx[0]])
                        segmentations.append(t['segmentations'][sample_indx[0]])
                        areas.append(t['areas'][sample_indx[0]])
                        if 'detected_face' in t.keys():
                            detected_face.append(t['detected_face'][sample_indx[0]])
                        else:
                            detected_face.append(None)
                        img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[0]])
                        img.append(Image.open(img_path).convert('RGB'))

                        frame_num += 1
                        vid_list.append(vid)
                        sample_indx_list.append(sample_indx[0])
                    else:
                        continue

            if flag is True: # frame_num == 5인 경우 while문 빠져나가기 (for문 때문에 혹시 모르니까 한 번 잡아주는거임)
                break
            idx = random.randint(0, self.__len__() - 1)

        new_target['segmentations'] = segmentations
        new_target['bboxes'] = bboxes
        new_target['areas'] = areas
        new_target['category_id'] = inst_label
        new_target['detected_face'] = detected_face

        target = {'video_id': vid_list, 'annotations': [new_target]}
        target_inds = inds  ## [0,1,2,3,4]
        target = self.prepare_diff(img[0], target, target_inds, sample_inds=sample_indx_list)

        return img, target

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


class ConvertCocoPolysToMaskFace_diff(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, target_inds, sample_inds):
        w, h = image.size
        video_id = target['video_id']
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        video_len = len(anno[0]['bboxes'])  # no usage

        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []

        # add valid flag for bboxes
        ## changed for face_detected
        for i, ann in enumerate(anno):
            category = False  ##
            for id in target_inds:
                bbox = ann['bboxes'][id]
                areas = ann['areas'][id]
                segm = ann['segmentations'][id]

                if 'detected_face' in ann.keys():
                    face_detected = ann['detected_face'][id]  ## get face_detected

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
            if category:  ##
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

        image_id = [id + video_id[id] * 1000 for id in target_inds]
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


def make_copy_paste_transforms():
    cp_normalize = T.Compose([
        transforms_img_copy_paste.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    normalize = normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]
    flip_transforms = T.RandomHorizontalFlip()
    resize_transforms = transforms_img_copy_paste.RandomResize(max_size=768)
    check = T.Check()
    cp_transforms = transforms_img_copy_paste.CopyAndPaste()
    cp_compose = transforms_img_copy_paste.CopyPasteCompose(flip_transforms, resize_transforms, check, cp_transforms,
                                                        cp_normalize, normalize, scales)

    return cp_compose


def build(image_set, args):
    root = Path(args.miseang_path)
    assert root.exists(), f'provided Miseang path {root} does not exist'

    if args.dataset_file == 'Miseang':
        mode = 'instances'
        PATHS = {
            "train": (root / "train/frame", root / "annotations" / f'{mode}_train_sub_face_5frames.json'),
            # "train": (root / "train/60frame", root / "annotations" / f'{mode}_train_sub_face.json'),
            "val": (root / "val/frame", root / "annotations" / f'{mode}_val_sub.json'),
        }
        img_folder, ann_file = PATHS[image_set]
        print('use Miseang dataset - miseang_img.py')

        if args.simple_cp is True and image_set == 'train':
            copy_paste_tfs = make_copy_paste_transforms()
            print('make copy paste transforms')
            if args.img_copy_paste_same is True:
                status = 'same'
                print('status - same')
            elif args.img_copy_paste_diff is True:
                status = 'diff'
                print('status - diff')
        else:
            copy_paste_tfs = None


        dataset = MiseangDataset(img_folder, ann_file, transforms=make_miseang_transforms(image_set), return_masks=args.masks,
                                 num_frames=args.num_frames, copy_paste_transforms=copy_paste_tfs, status=status)

    return dataset