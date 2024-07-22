import copy
import math
import random
import sys

import PIL
import torch
# import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh, box_iou
from util.misc import interpolate
import numpy as np
from numpy import random as rand
from PIL import Image
import cv2
from skimage.filters import gaussian
import datasets.transforms_clip as T

class CopyPasteCompose(object):
    def __init__(self, flip_transforms, resize_transforms, check, hsv_transforms, cp_transforms, cp_normalize, normalize, scales, size, position, GD, PD, all_frames):
        self.flip_transforms = flip_transforms
        self.resize_transforms = resize_transforms
        self.check = check
        self.hsv_transforms = hsv_transforms
        self.cp_transforms = cp_transforms
        self.cp_normalize = cp_normalize
        self.normalize = normalize
        self.scales = scales
        self.size = size
        self.position = position
        self.GD = GD # Geometric Distortions
        self.PD = PD # Photometric Distortions
        self.all_frames = all_frames # MVDA - all_frames

    def __call__(self, image, target, pasted_image, pasted_target, now_frames):        #
        if self.GD:
            ## Geometric Distortions
            # random flip
            image, target = self.flip_transforms(image, target, now_frames)
            pasted_image, pasted_target = self.flip_transforms(pasted_image, pasted_target, now_frames)

            # random resize
            size = random.choice(self.scales)
            image, target = self.resize_transforms(image, size, target, now_frames)
            image, target = self.check(image, target, now_frames)

            pasted_image, pasted_target = self.resize_transforms(pasted_image, size, pasted_target, now_frames)
            pasted_image, pasted_target = self.check(pasted_image, pasted_target, now_frames)
            ##

        ## Photometric Distortions
        # random contrast, saturation, hue
        if self.PD:
            lower, upper, b_delta, h_delta = 0.5, 1.5, 32, 18.0
            r, con, sat, bri, hue = rand.randint(2), rand.uniform(lower, upper), rand.uniform(lower, upper), rand.uniform(-b_delta, b_delta), rand.uniform(-h_delta, h_delta)
            image, target = self.hsv_transforms(image, target, now_frames, r, con, sat, bri, hue)
            pasted_image, pasted_target = self.hsv_transforms(pasted_image, pasted_target, now_frames, r, con, sat, bri, hue)

        # copy paste
        image, target = self.cp_transforms(image, target, pasted_image, pasted_target, now_frames, self.size, self.position, self.all_frames)

        # normalize
        if isinstance(image[0], np.ndarray):
            image, target = self.cp_normalize(image, target, now_frames)

        else:
            image, target = self.normalize(image, target, now_frames)
        return image, target

#### Image Data Augmentation - Distortions

class RandomResize(object):
    def __init__(self, max_size=None):
        self.max_size = max_size

    def __call__(self, img, size, target=None, now_frames=None):
        return T.resize(img, target, size, self.max_size)


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target, now_frames, r, con, sat, hue):
        if r:
            alpha = con
            image *= alpha
        return image, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target, r, bri):
        if r:
            delta = bri
            image += delta
        return image, target


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target, now_frames, r, con, sat, hue):
        if r:
            image[:, :, 1] *= sat
        return image, target


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target, now_frames, r, con, sat, hue):
        if r:
            image[:, :, 0] += hue
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target, r):
        if r:
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = T.SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target, now_frames, r, con, sat, hue):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, now_frames, r, con, sat, hue):
        for t in self.transforms:
            image, target = t(image, target, now_frames, r, con, sat, hue)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, clip, target, now_frames, r, con, sat, bri, hue):
        imgs = []
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, target = self.rand_brightness(img, target, r, bri)
            if r:
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, target = distort(img, target, now_frames, r, con, sat, hue)
            img, target = self.rand_light_noise(img, target, r)
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, target


#### Video Data Augmentation - MVDA

class CopyAndPaste(object):
    def __init__(self):
        self.target_key = get_target_key()

    def __call__(self, image, target, pasted_image, pasted_target, now_frames, size, position, all_frames):
        cp_image_list, new_target = apply_mask_copy_paste(image, target, pasted_image, pasted_target, now_frames, self.target_key, size, position, all_frames) ##
        return cp_image_list, new_target


# ============== 기존 메소드 =============== #
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
    ## masks: target frame의 mask
    ## paste_masks: 복사할 객체의 mask
    ## alpha: paste_masks의 binary한 tensor
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
            img.append(torch.tensor(im, dtype=torch.float32)) # 수정 코드, 왜 바꾼거지?

        return img, target


# ============== 수정한 메소드 =============== #
def apply_mask_copy_paste(image_list, target, pasted_image_list, pasted_target, now_frames, target_key, size, position, all_frames):
    cp_image_list = []
    cp_mask_list = []
    cp_boxes_list = []

    target_list_dict, pasted_target_list_dict = split_targets(target, pasted_target, now_frames, target_key)  # 2개 같이 split

    # 학습 데이터 중 비어있는 값이 존재
    if pasted_target_list_dict is None:
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

    # Replacement
    image_list, target_list_dict, mask_out_region = mask_out_bbox(image_list, target_list_dict, target['size'], now_frames, target_key) # mask out background instance's bbox
    pasted_image_list, choice_target_list_dict = replace_inst(pasted_image_list, pasted_target_list_dict, now_frames, target['size'], mask_out_region) # replace pasted_target instance

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
        alpha = None

        mask_list = []
        for i in range(len(choice_target_list_dict['masks'][frame_num])):
            mask_list.append(choice_target_list_dict['masks'][frame_num][i])

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
        paste_masks = []  ## 복사할 객체의 mask 값 list
        for i in range(len(choice_target_list_dict['masks'][frame_num])):
            paste_masks.append(choice_target_list_dict['masks'][frame_num][i].numpy())

        if target_list_dict: # background 인물 존재 여부에 따라 달리 CP
            masks = []  ## target frame의 mask 값 list
            for i in range(len(target_list_dict['masks'][frame_num])):
                masks.append(target_list_dict['masks'][frame_num][i].numpy())

            cp_mask = masks_copy_paste(masks, paste_masks, alpha)
            cp_mask_list.append(cp_mask)
        else:
            cp_mask_list.append(paste_masks)

        # Bbox에 copy and pasted 적용
        paste_boxes = []
        for i in range(len(choice_target_list_dict['boxes'][frame_num])):
            paste_boxes.append(choice_target_list_dict['boxes'][frame_num][i].numpy())

        if target_list_dict:
            boxes = []
            for i in range(len(target_list_dict['boxes'][frame_num])):
                boxes.append(target_list_dict['boxes'][frame_num][i].numpy())

            cp_boxes = bboxes_copy_paste(boxes, paste_boxes, masks, paste_masks, alpha)
            cp_boxes_list.append(cp_boxes)
        else:
            cp_boxes_list.append(paste_boxes)

    # label extend
    if target_list_dict:
        cp_labels = torch.cat([target_list_dict['labels'], choice_target_list_dict['labels']], dim=0)
    else:
        cp_labels = choice_target_list_dict['labels']

    # frame format -> instance format
    # valid, area, iscrowd도 계산 필요한 것 같음 (필요 없어도 그래도 맞춰놓기) @@ 수정 필요
    instance_cnt = len(cp_boxes_list[0])
    tmp_cp_boxes_list = frame2instance(cp_boxes_list, instance_cnt, now_frames)
    tmp_cp_masks_list = frame2instance(cp_mask_list, instance_cnt, now_frames)

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



# 기존 split_targets에서 now_frames로 돌아가도록 수정함
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
    pasted_target_empty = False

    for key in target_key:
        '''
        target_list_dict[key] = torch.split(target[key], int(len(target[key]) / now_frames))
        pasted_target_list_dict[key] = torch.split(pasted_target[key], int(len(pasted_target[key]) / now_frames))
        '''
        '''
        target_list_dict[key] = torch.split(target[key], now_frames)
        pasted_target_list_dict[key] = torch.split(pasted_target[key], now_frames)
        '''

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
            if pasted_target_list_dict[key][i]:
                pasted_target_list_dict[key][i] = torch.stack(pasted_target_list_dict[key][i], dim=0)
            else:
                pasted_target_empty = True

    target_list_dict['labels'] = target['labels']
    pasted_target_list_dict['labels'] = pasted_target['labels']
    if pasted_target_empty is True:
        return target_list_dict, None
    else:
        return target_list_dict, pasted_target_list_dict


# frame format -> instance format
def frame2instance(frame_format_list, instance_cnt, now_frames):
    instance_format_list = []

    for i in range(instance_cnt):
        instance_format_list.append([])

        for frame_num in range(now_frames):
            instance_format_list[-1].append(frame_format_list[frame_num][i])

        instance_format_list[-1] = torch.tensor(instance_format_list[-1], dtype=torch.float32)

    tmp_list = torch.cat(instance_format_list, dim=0)

    return tmp_list


# mask out background instance's bbox for replacement
def mask_out_bbox(image_list, target_list_dict, image_size, now_frames, target_key):
    '''
    :param image_list:
    :param target_list_dict:
    :param image_size:
    :param now_frames:
    :param target_key:
    :return:
    '''
    # choose one instance from background instances
    mask_out_target_list_dict = {}
    selected_idx = [i for i, x in enumerate(target_list_dict['labels']) if i != 0] # 일단은 첫 번째 인물을 삭제

    if len(selected_idx) == 0:
        mask_out_target_list_dict = None
    else:
        # mask out target
        # boxes, masks
        for key in target_key:
            mask_out_target_list_dict[key] = []

            for i in range(now_frames):
                mask_out_target_list_dict[key].append([])

                for idx in selected_idx:
                    mask_out_target_list_dict[key][i].append(target_list_dict[key][i][idx])

                if len(mask_out_target_list_dict[key][i]) > 0:
                    if key == 'boxes':
                        mask_out_target_list_dict[key][i] = torch.stack(mask_out_target_list_dict[key][i], dim=0)

                    if key == 'masks':
                        mask_out_target_list_dict[key][i] = torch.stack(mask_out_target_list_dict[key][i], dim=0)

        # labels
        mask_out_target_list_dict['labels'] = [x for i, x in enumerate(target_list_dict['labels']) if i != 0]
        mask_out_target_list_dict['labels'] = torch.stack(mask_out_target_list_dict['labels'], dim=0)

    # mask out image
    # 일단 처음부터 끝까지 같은 크기의 cuboid 형태로 자르기
    mask_out_region = extract_cuboid_region(target_list_dict['boxes'], image_size, now_frames, idx=0) # idx=0 : 첫 번째 개체만 선택
    image_list = mask_out_region_clip_by_frame(image_list, mask_out_region)

    return image_list, mask_out_target_list_dict, mask_out_region


# replace target instance
def replace_inst(pasted_image_list, pasted_target_list_dict, now_frames, image_size, mask_out_region):
    '''
    :param pasted_image_list:
    :param pasted_target_list_dict:
    :param now_frames:
    :param mask_out_region: (i, j, h, w)
    :return:
    '''

    # pasted_target_list_dict로부터 cuboid 생성
    crop_region = extract_cuboid_region(pasted_target_list_dict['boxes'], image_size, now_frames, idx=0)
    pasted_image_list, pasted_target_list_dict = crop_clip_by_frame(pasted_image_list, pasted_target_list_dict, crop_region, now_frames)

    # resize
    ## 수정 필요
    size = pasted_image_list[0].size
    size = math.floor(size[0] * 1.4), math.floor(size[1] * 1.4)
    pasted_image_list, pasted_target_list_dict = resize_clip_by_frame(pasted_image_list, pasted_target_list_dict, now_frames, size, max_size=image_size[0].tolist())

    # 위치 조정 및 원본 크기로 pad
    ## 수정 필요 : 타겟 인물이 더 크다면??, 타겟 인물이 더 작을 때 중앙에 위치
    crop_image_width, crop_image_height = pasted_image_list[0].size
    # mask_out_region : (top, left, height, width)
    pad_left = mask_out_region[1]
    pad_top = mask_out_region[0]
    pad_right = image_size[1] - (pad_left + crop_image_width)
    pad_bottom = image_size[0] - (pad_top + crop_image_height)

    pasted_image_list, pasted_target_list_dict = pad_clip_by_frame(pasted_image_list, pasted_target_list_dict, (pad_left, pad_top, pad_right, pad_bottom), now_frames)

    return pasted_image_list, pasted_target_list_dict

# choice_target_list_dict로부터 전체 프레임에서 cuboid가 움직이는 영역 추출 (for cropping cuboid from source clip)
def extract_cuboid_region(choice_target_list_dict_boxes, image_size, now_frames, idx):
    left = []
    top = []
    right = []
    bottom = []

    for i in range(now_frames):
        l = choice_target_list_dict_boxes[i][idx][0]
        t = choice_target_list_dict_boxes[i][idx][1]
        r = choice_target_list_dict_boxes[i][idx][2]
        b = choice_target_list_dict_boxes[i][idx][3]
        if l == r or t == b: # 인물이 없을 때  -> 안 그러면 crop 시에 크게 잡혀서 blank_space 오류 발생
            left.append(-1)
            right.append(-1)
            top.append(-1)
            bottom.append(-1)
        else:
            left.append(l)
            top.append(t)
            right.append(r)
            bottom.append(b)

    left_ =math.floor(min(i for i in left if i >= 0))
    top_ = math.floor(min(i for i in top if i >= 0))
    right_ = math.ceil(max(i for i in right if i >= 0))
    bottom_ = math.ceil(max(i for i in bottom if i >= 0))

    if right_ > image_size[1]:
        right_ = image_size[1].tolist()
    if bottom_ > image_size[0]:
        bottom_ = image_size[0].tolist()

    width = right_ - left_
    height = bottom_ - top_

    return top_, left_, height, width


# modified from transforms_clip.py : crop, resize, pad
def crop_clip_by_frame(clip, target, region, now_frames):
    cropped_image = []
    for image in clip:
        cropped_image.append(F.crop(image, *region))

    target = target.copy()
    i, j, h, w = region

    # box는 나중에 extract_boxes를 통해서 계산되기 때문에 여기서 굳이 계산 안 해줘도 될 듯
    # if "boxes" in target:
    #     pass

    # 확인용
    # size = cropped_image[0]
    # h_ = (i+h) - i
    # w_ = (j+w) - j
    if "masks" in target:
        for n in range(now_frames):
            target['masks'][n] = target['masks'][n][:, i:i + h, j:j + w] # 개체 1개 기준

    return cropped_image, target


def resize_clip_by_frame(clip, target, now_frames, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(clip[0].size, size, max_size)
    rescaled_image = []
    for image in clip:
        rescaled_image.append(F.resize(image, size))

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image[0].size, clip[0].size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    # box는 나중에 extract_boxes를 통해서 계산되기 때문에 여기서 굳이 계산 안 해줘도 될 듯
    # if "boxes" in target:
    #     boxes = target["boxes"]
    #     scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    #     target["boxes"] = scaled_boxes

    # seqformer에서 area 사용 안 하는 것 같기도 하고 choice_target_list_dict에 area 없음
    # if "area" in target:
    #     area = target["area"]
    #     scaled_area = area * (ratio_width * ratio_height)
    #     target["area"] = scaled_area

    h, w = size
    # target["size"] = torch.tensor([h, w])

    if "masks" in target:
        for n in range(now_frames):
            if target['masks'][n].shape[0] > 0:
                target['masks'][n] = interpolate(
                    target['masks'][n][:, None].float(), size, mode="nearest")[:, 0] > 0.5
            else:
                target['masks'][n] = torch.zeros((target['masks'][n].shape[0], h, w))

    return rescaled_image, target


def pad_clip_by_frame(clip, target, padding, now_frames):
    # padding[0]-left, [1]-top, [2]-right, [3]-bottom
    padded_image = []
    for image in clip:
        padded_image.append(F.pad(image, (padding[0], padding[1], padding[2], padding[3])))

    target = target.copy()
    if "masks" in target:
        for n in range(now_frames):
            target['masks'][n] = torch.nn.functional.pad(target['masks'][n], (padding[0], padding[2], padding[1], padding[3]))
    return padded_image, target


def mask_out_region_clip_by_frame(clip, region):
    masked_out_image = []
    for image in clip:
        tmp_img = F.to_tensor(image)
        tmp_img = F.erase(tmp_img, *region, v=0, inplace=False)
        tmp_img = F.to_pil_image(tmp_img)
        masked_out_image.append(tmp_img)
    return masked_out_image