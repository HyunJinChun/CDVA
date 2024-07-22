import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh, box_iou
from util.misc import interpolate
import numpy as np
from numpy import random as rand
from PIL import Image
import cv2
from skimage.filters import gaussian
from datasets.transforms_clip import resize


class CopyPasteCompose(object):
    def __init__(self, flip_transforms, resize_transforms, check, cp_transforms, cp_normalize, normalize, scales):
        self.flip_transforms = flip_transforms
        self.resize_transforms = resize_transforms
        self.check = check
        self.cp_transforms = cp_transforms
        self.cp_normalize = cp_normalize
        self.normalize = normalize
        self.scales = scales

    def __call__(self, image, target, pasted_image, pasted_target, now_frames, targetClass):
        size = random.choice(self.scales)

        # random flip
        image, target = self.flip_transforms(image, target, now_frames)
        pasted_image, pasted_target = self.flip_transforms(pasted_image, pasted_target, now_frames)

        # random resize
        image, target = self.resize_transforms(image, size, target, now_frames)
        image, target = self.check(image, target, now_frames)

        pasted_image, pasted_target = self.resize_transforms(pasted_image, size, pasted_target, now_frames)
        pasted_image, pasted_target = self.check(pasted_image, pasted_target, now_frames)

        # copy paste
        image, target = self.cp_transforms(image, target, pasted_image, pasted_target, now_frames, targetClass)

        # normalize
        if isinstance(image[0], np.ndarray):
            image, target = self.cp_normalize(image, target, now_frames)

        else:
            image, target = self.normalize(image, target, now_frames)

        return image, target


class RandomResize(object):
    def __init__(self, max_size=None):
        self.max_size = max_size

    def __call__(self, img, size, target=None, now_frames=None):
        return resize(img, target, size, self.max_size)


class CopyAndPaste(object):
    def __init__(self):
        self.target_key = get_target_key()

    def __call__(self, image, target, pasted_image, pasted_target, now_frames, targetClass):
        cp_image_list, new_target = apply_copy_paste(image, target, pasted_image, pasted_target, now_frames, self.target_key, targetClass)
        return cp_image_list, new_target


def get_target_key():
    # return ['boxes', 'masks', 'valid']
    return ['boxes', 'masks']


def do_normalize(clip, target):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = []
    for im in clip:
        img.append(F.to_tensor(im).to(torch.float32))

    image = []
    for im in img:
        image.append(F.normalize(im, mean=mean, std=std))

    if target is None:
        return image, None

    target = target.copy()
    h, w = image[0].shape[-2:]

    if "boxes" in target:
        boxes = target["boxes"]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        target["boxes"] = boxes

    return image, target


def apply_copy_paste(image_list, target, pasted_image_list, pasted_target, now_frames, target_key, targetClass):
    cp_image_list = []
    cp_mask_list = []
    cp_boxes_list = []

    # target과 pasted_target을 now_frames으로 나누기(tensor -> list of tensor)
    target_list_dict, pasted_target_list_dict = split_targets(target, pasted_target, now_frames, target_key)

    # Paste 대상이 되는 객체 선택
    choice_target_list_dict = select_pasted_instance(pasted_target_list_dict, target_key, now_frames, targetClass)

    if choice_target_list_dict is None:
        new_target = {
            'boxes': target['boxes'],
            'labels': target['labels'],
            'masks': target['masks'],
            'image_id': target['image_id'],
            # 'valid': target_list_dict['valid'],
            # 'area': target_list_dict['area'],
            # 'iscrowd': target_list_dict['iscrowd'],
            'orig_size': target['orig_size'],
            'size': target['size']
        }
        return image_list, new_target

    # Alpha(선택된 모든 객체를 binary한 tensor) 계산
    for frame_num in range(now_frames):

        mask_list = []
        for i in range(len(choice_target_list_dict['masks'][frame_num])):
            mask_list.append(choice_target_list_dict['masks'][frame_num][i])

        alpha = mask_list[0] > 0
        for mask in mask_list[1:]:
            alpha += mask > 0

        alpha = alpha.numpy()

        # Image에 copy and paste 적용
        image = image_list[frame_num]
        pasted_image = pasted_image_list[frame_num]

        cp_image = image_copy_paste(image, pasted_image, alpha)
        cp_image_list.append(cp_image)

        # Mask에 copy and paste 적용
        masks = []
        for i in range(len(target_list_dict['masks'][frame_num])):
            masks.append(target_list_dict['masks'][frame_num][i].numpy())

        paste_masks = []
        for i in range(len(choice_target_list_dict['masks'][frame_num])):
            paste_masks.append(choice_target_list_dict['masks'][frame_num][i].numpy())

        cp_mask = masks_copy_paste(masks, paste_masks, alpha)
        cp_mask_list.append(cp_mask)

        # Bbox에 copy and pasted 적용
        boxes = []
        for i in range(len(target_list_dict['boxes'][frame_num])):
            boxes.append(target_list_dict['boxes'][frame_num][i].numpy())

        paste_boxes = []
        for i in range(len(choice_target_list_dict['boxes'][frame_num])):
            paste_boxes.append(choice_target_list_dict['boxes'][frame_num][i].numpy())

        cp_boxes = bboxes_copy_paste(boxes, paste_boxes, masks, paste_masks, alpha)
        cp_boxes_list.append(cp_boxes)

    # label extend
    cp_labels = torch.cat([target_list_dict['labels'], choice_target_list_dict['labels']], dim=0)

    # frame format -> instance format
    tmp_cp_boxes_list = []
    tmp_cp_masks_list = []

    instance_cnt = len(cp_boxes_list[0])

    for i in range(instance_cnt):
        tmp_cp_boxes_list.append([])
        tmp_cp_masks_list.append([])

        for frame_num in range(now_frames):
            tmp_cp_boxes_list[-1].append(cp_boxes_list[frame_num][i])
            tmp_cp_masks_list[-1].append(cp_mask_list[frame_num][i])

        tmp_cp_boxes_list[-1] = torch.tensor(tmp_cp_boxes_list[-1], dtype=torch.float32)
        tmp_cp_masks_list[-1] = torch.tensor(tmp_cp_masks_list[-1], dtype=torch.float32)

    tmp_cp_boxes_list = torch.cat(tmp_cp_boxes_list, dim=0)
    tmp_cp_masks_list = torch.cat(tmp_cp_masks_list, dim=0)

    new_target = {
        'boxes': tmp_cp_boxes_list,
        'labels': cp_labels,
        'masks': tmp_cp_masks_list,
        'image_id': target['image_id'],
        # 'valid': target_list_dict['valid'],
        # 'area': target_list_dict['area'],
        # 'iscrowd': target_list_dict['iscrowd'],
        'orig_size': target['orig_size'],
        'size': target['size']
    }

    return cp_image_list, new_target


def split_targets(target, pasted_target, now_frames, target_key):
    '''
    초기 target(source clip)과 pasted_target(destination clip) 5개는 하나의 행렬로 구성
    N : clip내의 frame 개수
    I : clip내의 instance 개수
    target['boxes'] = NI * 4 (frame 순서대로, frame 내에서는 instance 순서대로 정렬)
    len_bbox / now_frame = number of instances
    행렬을 frame 순서로 나눔

    :param target:
    :param pasted_target:
    :param now_frames:
    :param target_key:
    :return:
    '''
    target_list_dict = {}
    pasted_target_list_dict = {}

    for key in target_key:
        instance_cnt = int(len(target[key]) / now_frames)
        pasted_instance_cnt = int(len(pasted_target[key]) / now_frames)

        target_list_dict[key] = []
        pasted_target_list_dict[key] = []

        for i in range(now_frames):
            target_list_dict[key].append([])
            pasted_target_list_dict[key].append([])

            for j in range(instance_cnt):
                target_list_dict[key][i].append(target[key][i+now_frames*j])
            for j in range(pasted_instance_cnt):
                pasted_target_list_dict[key][i].append(pasted_target[key][i + now_frames * j])

            # target_list_dict[key][i] = torch.tensor(target_list_dict[key][i], dtype=torch.float32)
            target_list_dict[key][i] = torch.stack(target_list_dict[key][i], dim=0)
            pasted_target_list_dict[key][i] = torch.stack(pasted_target_list_dict[key][i], dim=0)

    target_list_dict['labels'] = target['labels']
    pasted_target_list_dict['labels'] = pasted_target['labels']

    return target_list_dict, pasted_target_list_dict


def select_pasted_instance(pasted_target_list_dict, target_key, now_frames, targetClass):
    '''

    :param pasted_target_list_dict:
    :param target_key:
    :return:
    '''
    instance_choice = [x for x, i in enumerate(pasted_target_list_dict['labels']) if i == targetClass]
    if len(instance_choice) > 1:
        instance_choice = [random.choice(instance_choice)]
    choice_target_list_dict = {}

    for key in target_key:
        choice_target_list_dict[key] = []

        for i in range(now_frames):
            choice_target_list_dict[key].append([])

            for idx in instance_choice:
                choice_target_list_dict[key][i].append(pasted_target_list_dict[key][i][idx])

            if len(choice_target_list_dict[key][i]) > 0:
                if key == 'boxes':
                    choice_target_list_dict[key][i] = torch.stack(choice_target_list_dict[key][i], dim=0)

                if key == 'masks':
                    choice_target_list_dict[key][i] = torch.stack(choice_target_list_dict[key][i], dim=0)

    # label
    if len(instance_choice) > 0:
        choice_target_list_dict['labels'] = []

        for idx in instance_choice:
            choice_target_list_dict['labels'].append(pasted_target_list_dict['labels'][idx])

        choice_target_list_dict['labels'] = torch.stack(choice_target_list_dict['labels'], dim=0)

    else:
        choice_target_list_dict = None

    return choice_target_list_dict


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    img = F.to_tensor(img).to(torch.float32).numpy()
    paste_img = F.to_tensor(paste_img).to(torch.float32).numpy()

    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        alpha = alpha[None, ...]

        img = paste_img * alpha + img * (1 - alpha)

    return img


def masks_copy_paste(masks, paste_masks, alpha):
    if alpha is not None:
        # eliminate pixels that will be pasted over
        masks = [
            np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.uint8) for mask in masks
        ]

    new_paste_masks = [
        (paste_mask > 0).astype(np.uint8) for paste_mask in paste_masks
    ]

    new_masks = masks + new_paste_masks
    return new_masks


def extract_bboxes(masks):
    bboxes = []
    # allow for case of no masks
    if len(masks) == 0:
        return bboxes

    h, w = masks[0].shape
    for mask in masks:
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1
            y1 /= w
            y2 /= w
            x1 /= h
            x2 /= h
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0

        bboxes.append((y1, x1, y2, x2))

    return bboxes


def bboxes_copy_paste(bboxes, paste_bboxes, masks, paste_masks, alpha):
    if paste_bboxes is not None:
        masks = masks_copy_paste(masks, paste_masks=[], alpha=alpha)
        adjusted_bboxes = extract_bboxes(masks)

        # only keep the bounding boxes for objects listed in bboxes
        # mask_indices = [box[-1] for box in bboxes]
        # adjusted_bboxes = [adjusted_bboxes[idx] for idx in mask_indices]
        # append bbox tails (classes, etc.)
        # adjusted_bboxes = [bbox + tail[4:] for bbox, tail in zip(adjusted_bboxes, bboxes)]

        # adjust paste_bboxes mask indices to avoid overlap
        if len(masks) > 0:
            max_mask_index = len(masks)
        else:
            max_mask_index = 0

        adjusted_paste_bboxes = extract_bboxes(paste_masks)

        bboxes = adjusted_bboxes + adjusted_paste_bboxes

    return bboxes


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = F.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)


class ToTensor(object):
    def __call__(self, clip, target, now_frames):
        img = []
        for im in clip:
            img.append(torch.tensor(im, dtype=torch.float32))

        return img, target