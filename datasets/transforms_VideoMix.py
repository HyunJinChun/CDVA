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
from datasets.transforms_clip import resize, Check


class CopyPasteCompose(object):
    def __init__(self, flip_transforms, resize_transforms, check, cp_transforms, cp_normalize, normalize, scales):
        self.flip_transforms = flip_transforms
        self.resize_transforms = resize_transforms
        self.check = check
        self.cp_transforms = cp_transforms
        self.cp_normalize = cp_normalize
        self.normalize = normalize
        self.scales = scales

    def __call__(self, image, target, pasted_image, pasted_target, now_frames):
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
        image, target = self.cp_transforms(image, target, pasted_image, pasted_target, now_frames)

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

    def __call__(self, image, target, pasted_image, pasted_target, now_frames):
        cp_image_list, new_target = apply_copy_paste(image, target, pasted_image, pasted_target, now_frames, self.target_key)
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


def apply_copy_paste(image_list, target, pasted_image_list, pasted_target, now_frames, target_key):
    cp_image_list = []
    cp_mask_list = []
    cp_boxes_list = []

    # Paste 대상이 되는 객체 선택
    pasted_image_list, pasted_target, region = VideoMix(pasted_image_list, pasted_target, now_frames, alpha=8.0)

    if pasted_target['boxes'].nelement() == 0:
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

    # target과 pasted_target을 now_frames으로 나누기(tensor -> list of tensor)
    target_list_dict, choice_target_list_dict = split_targets(target, pasted_target, now_frames, target_key)

    # Alpha : region으로 계산된 사각형 부분
    for frame_num in range(now_frames):
        mask_list = []
        W, H = image_list[0].size
        for i in range(1, H+1):
            li = []
            for j in range(1, W+1):
                if i >= region[1] and i <= region[3] and j >= region[0] and j <= region[2]:
                    li.append(True)
                else:
                    li.append(False)
            mask_list.append(li)
        mask_list = [torch.tensor(mask_list)]

        alpha = mask_list[0] > 0
        for mask in mask_list[1:]: # 선택된 모든 객체를 alpha에 합쳐서 alpha는 전체 mask 값을 나타내게 됨
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


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    img = F.to_tensor(img).to(torch.float32).numpy()
    paste_img = F.to_tensor(paste_img).to(torch.float32).numpy()

    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        # alpha = alpha.swapaxes(1, 0)

        # img_dtype = img.dtype
        alpha = alpha[None, ...]

        # alpha = np.stack([alpha, alpha, alpha], axis=0)
        # alpha = alpha.swapaxes(1, 0).swapaxes(2, 1)

        # tmp1 = img * (1 - alpha)
        img = paste_img * alpha + img * (1 - alpha)
        # img = img.astype(img_dtype)

    return img


def masks_copy_paste(masks, paste_masks, alpha):
    if alpha is not None:
        # eliminate pixels that will be pasted over
        masks = [
            np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.uint8) for mask in masks
        ]
        # masks.extend(paste_masks)

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

        # paste_mask_indices = [max_mask_index + ix for ix in range(len(paste_bboxes))]
        # paste_bboxes = [pbox[:-1] + (pmi,) for pbox, pmi in zip(paste_bboxes, paste_mask_indices)]
        adjusted_paste_bboxes = extract_bboxes(paste_masks)
        # adjusted_paste_bboxes = [apbox + tail[4:] for apbox, tail in zip(adjusted_paste_bboxes, paste_bboxes)]

        bboxes = adjusted_bboxes + adjusted_paste_bboxes

    return bboxes

# based on VideoMix
def VideoMix(pasted_image_list, pasted_target, now_frames, alpha=8.0):
    '''
    :param pasted_image_list:
    :param pasted_target_list_dict:
    :param target_key:
    :return: cropped pasted_image_list, cropped pasted_target_list_dict
    '''
    lam = np.random.beta(alpha, alpha)

    bbx1, bby1, bbx2, bby2, H, W = rand_bbox(pasted_target, lam)
    bbx1, bby1, bbx2, bby2 = bbx1.tolist(), bby1.tolist(), bbx2.tolist(), bby2.tolist()
    crop_region = (bby1, bbx1, bby2 - bby1, bbx2 - bbx1)

    # crop : (top, left, height, width)
    pasted_image_list, pasted_target = crop_clip_by_frame(pasted_image_list, pasted_target, crop_region)
    check = Check()
    pasted_image_list, pasted_target = check(pasted_image_list, pasted_target, now_frames)

    # pad : (left, top, right, bottom)
    pad_right = W - bbx2
    pad_bottom = H - bby2
    pad_region = (bbx1, bby1, pad_right.tolist(), pad_bottom.tolist())
    pasted_image_list, pasted_target = pad_clip_by_frame(pasted_image_list, pasted_target, pad_region)

    return pasted_image_list, pasted_target, (bbx1, bby1, bbx2, bby2)


def rand_bbox(target, lam):
    H, W = target['size'] # target_size[0] : height, target_size[1] : width
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2, H, W

# =====
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

# modified from transforms_clip.py
def crop_clip_by_frame(clip, target, region):
    cropped_image = []
    for image in clip:
        cropped_image.append(F.crop(image, *region))

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    return cropped_image, target

def pad_clip_by_frame(clip, target, padding):
    # padding[0]-left, [1]-top, [2]-right, [3]-bottom
    padded_image = []
    for image in clip:
        padded_image.append(F.pad(image, (padding[0], padding[1], padding[2], padding[3])))

    if target is None:
        return padded_image, None

    target = target.copy()

    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[0].size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (padding[0], padding[2], padding[1], padding[3]))

    return padded_image, target

