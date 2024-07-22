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

        # random resize - check 사용 시 bbox가 잘리는 오류 발생해서 뺐음
        image, target = self.resize_transforms(image, size, target, now_frames)
        # image, target = self.check(image, target, now_frames)

        pasted_image, pasted_target = self.resize_transforms(pasted_image, size, pasted_target, now_frames)
        # pasted_image, pasted_target = self.check(pasted_image, pasted_target, now_frames)

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
        return T.resize(img, target, size, self.max_size)


class CopyAndPaste(object):
    def __init__(self):
        self.target_key = get_target_key()

    def __call__(self, image, target, pasted_image, pasted_target, now_frames):
        cp_image_list, new_target = apply_mask_copy_paste(image, target, pasted_image, pasted_target, now_frames, self.target_key) ##
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
def apply_mask_copy_paste(image_list, target, pasted_image_list, pasted_target, now_frames, target_key):
    cp_image_list = []
    cp_mask_list = []
    cp_boxes_list = []

    # 220418 전체 이미지 이용
    # pasted_image_width, pasted_image_height = pasted_image_list[0].size
    # image_width, image_height = image_list[0].size
    # if pasted_image_width >= image_width : # 크기가 커진 경우 - crop - 일단 중앙에 위치 (이동하는 경우 pad를 줘야 함)
    #     crop_top = int(round((pasted_image_height - image_height) / 2.))
    #     crop_left = int(round((pasted_image_width - image_width) / 2.))
    #     pasted_image_list, pasted_target = crop(pasted_image_list, pasted_target,
    #                                             (crop_top, crop_left, image_height, image_width))
    # else : # 크기가 작아진 경우 - pad - 랜덤 위치
    #     pad_left = random.randint(0, (image_width-pasted_image_width))
    #     pad_top = random.randint(0, (image_height-pasted_image_height))
    #     pad_right = image_width - (pad_left + pasted_image_width)
    #     pad_bottom = image_height - (pad_top + pasted_image_height)
    #     pasted_image_list, pasted_target = pad_(pasted_image_list, pasted_target, (pad_left, pad_top, pad_right, pad_bottom))
    #     print()
    # END

    # target과 pasted_target을 now_frames으로 나누기(tensor -> list of tensor)
    target_list_dict, pasted_target_list_dict = split_targets(target, pasted_target, now_frames, target_key) # 2개 같이 split
    # pasted_target_list_dict = split_target(pasted_target, now_frames, target_key)
    # Paste 대상이 되는 객체 선택
    # choice_target_list_dict = select_one_pasted_instance(pasted_target_list_dict, target_key, now_frames) # 장그래와 someone이 아닌 인물만 선택하도록 함 (1명만 선택하는 것은 extract_cuboid에서 하는 것으로 구현해둠)

    # 220602 target clip 속 인물의 크기를 고려하여 source clip의 인물을 선택
    blank_space = cal_blank_space(target['size'], target_list_dict)
    choice_target_list_dict = select_pasted_instance_by_size(target['size'], target_list_dict, pasted_target_list_dict, target_key, now_frames, blank_space)

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

    # 220418 clip 단위로 개체 이미지 이용
    # 첫번째 frame의 첫번째 개체의 box 만큼 이미지 crop - 크기 조정 (△ - 정확히 파악 후 수정 필요), 이동 (o)
    # pasted_image_list, pasted_target = crop_instance_clip(choice_target_list_dict, image_list, pasted_image_list, pasted_target, now_frames)
    # target_list_dict, choice_target_list_dict = split_targets(target, pasted_target, now_frames, target_key)
    # END

    # 220420 clip의 frame을 따로 처리
    # 랜덤 위치에 삽입
    # pasted_image_list, choice_target_list_dict = mask_tracklet_randompos(target['size'], pasted_image_list, choice_target_list_dict, now_frames)

    # 220512 target clip의 인물 고려 - frame #1의 인물 #1의 bbox 고려
    # pasted_image_list, choice_target_list_dict = mask_tracklet_usebbox_one(target['size'], target_list_dict, choice_target_list_dict, pasted_image_list, now_frames)

    # 220518 target clip의 인물 고려 - frame #1의 모든 인물의 bbox 고려
    pasted_image_list, choice_target_list_dict = mask_tracklet_usebbox_multi(target['size'], target_list_dict, choice_target_list_dict, pasted_image_list, now_frames, blank_space)

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

        ## copy and paste 적용하기 이전에 target frame에 너무 많이 가려지면 mask 정보 없애기 (seg, bbox, label 모두)
        ## 고민 필요: 랜덤 위치 + 정보 삭제 or 최대한 적절한 위치 + 정보 삭제 뭐가 적절할까...

        # Image에 copy and paste 적용
        image = image_list[frame_num]
        pasted_image = pasted_image_list[frame_num]

        cp_image = image_copy_paste(image, pasted_image, alpha)
        cp_image_list.append(cp_image)

        # Mask에 copy and paste 적용
        masks = []  ## target frame의 mask 값 list
        for i in range(len(target_list_dict['masks'][frame_num])):
            masks.append(target_list_dict['masks'][frame_num][i].numpy())

        paste_masks = []  ## 복사할 객체의 mask 값 list
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
            pasted_target_list_dict[key][i] = torch.stack(pasted_target_list_dict[key][i], dim=0)

    target_list_dict['labels'] = target['labels']
    pasted_target_list_dict['labels'] = pasted_target['labels']
    return target_list_dict, pasted_target_list_dict


def select_one_pasted_instance(pasted_target_list_dict, target_key, now_frames):
    '''
    :param pasted_target_list_dict:
    :param target_key:
    :return:
    '''
    instance_cnt = len(pasted_target_list_dict['boxes'][0])
    # print('choice target labels:', pasted_target_list_dict['labels'])
    # instance_cnt = 1 # 개체 한 개
    # random_choice = np.random.randint(2, size=instance_cnt).tolist()
    # random_choice = [i for i, x in enumerate(random_choice) if x == 1]
    # print('random_choice:', random_choice)
    random_choice = []
    tmp_choice = []
    if instance_cnt >= 1 :
        for i, idx in enumerate(pasted_target_list_dict['labels']):
            if idx in [2, 3, 4, 5, 6]: ## 수정 필요, 항상 똑같은 것만 선택되니까 random으로 선택되도록
                # random_choice.append(i)
                tmp_choice.append(i)
    if tmp_choice:
        # print('tmp_choice :', tmp_choice)
        random_choice.append(random.choice(tmp_choice))
        # print('random_choice : ', random_choice)

    choice_target_list_dict = {}

    for key in target_key:
        choice_target_list_dict[key] = []

        for i in range(now_frames):
            choice_target_list_dict[key].append([])

            for idx in random_choice:
                choice_target_list_dict[key][i].append(pasted_target_list_dict[key][i][idx])

            if len(choice_target_list_dict[key][i]) > 0:
                if key == 'boxes':
                    choice_target_list_dict[key][i] = torch.stack(choice_target_list_dict[key][i], dim=0)

                if key == 'masks':
                    choice_target_list_dict[key][i] = torch.stack(choice_target_list_dict[key][i], dim=0)

        # choice_target_list_dict[key] = tuple(choice_target_list_dict[key])

    # label
    if len(random_choice) > 0:
        choice_target_list_dict['labels'] = []

        for idx in random_choice:
            choice_target_list_dict['labels'].append(pasted_target_list_dict['labels'][idx])

        choice_target_list_dict['labels'] = torch.stack(choice_target_list_dict['labels'], dim=0)
        # print('choice label:', choice_target_list_dict['labels'])

    else:
        choice_target_list_dict = None

    return choice_target_list_dict


def select_pasted_instance_by_size(size, target_list_dict, pasted_target_list_dict, target_key, now_frames, blank_space):
    instance_cnt = len(pasted_target_list_dict['labels'])
    category = [2, 3, 4, 5, 6]

    instance_choice = []
    if instance_cnt >= 1:
        for i, label in enumerate(pasted_target_list_dict['labels']):
            if label in category:  ## 수정 필요, 항상 똑같은 것만 선택되니까 random으로 선택되도록
                instance_choice.append(i)
    if not instance_choice:
        return None

    target_instance_cnt = len(target_list_dict['boxes'][0])  # 인물이 1명일 때랑 여러 명일 때랑 형식이 달라서 구분해줘야 함
    image_width = size[1]
    selected_idx = -1  # 적당한 인물이 선택되면 해당 index로 변경, for문 이후에도 -1이면 괜찮은 인물이 없다는 의미
    selected = False

    for idx in instance_choice:
        # pasted_target의 가로 추출
        pasted_target = pasted_target_list_dict['boxes'][0][idx]
        p_width, p_height = pasted_target[2] - pasted_target[0], pasted_target[3] - pasted_target[1]
        p_width_percent = (p_width / image_width) * 100

        for i, target in enumerate(target_list_dict['boxes'][0]):
            # target의 가로 세로 추출
            t_width, t_height = target[2] - target[0], target[3] - target[1]
            t_width_percent = (t_width / image_width) * 100

            # 2. 인물(들)이 있으면 target clip 내 인물(들)의 크기와 비교
            # 비율 계산 후 5% 차이 이하인 경우를 선택
            # blank_space - # left : ['left', left_region], right : ['right', right_region], max_gap : ['max_gap', max_gap_region, max_gap(list)]
            if abs(t_width_percent - p_width_percent) <= 10:
                print('크기 10 통과 !')
                # 2. END
                # 3. 크기 비교 후 적당한 게 있을 때, 선택된 인물이 빈 공간보다 과도하게 더 크다면 다음으로 넘기기 (작거나 적당하게 크면 최종 선택)
                # 현재 적당한 크기 : 계산한 비율이 120% 이하인 경우
                if (p_width / blank_space[1]) * 100 <= 120:
                    print('빈 공간 120 통과 !')
                    selected_idx = idx
                    selected = True
                    break
                # 3. END
            # 크기가 5% 차이 이상인 경우 또는 비율 계산 후 120% 이상인 경우는 continue
        if selected:
            break

    if selected_idx == -1:
        return None

    # ===================================================================

    choice_target_list_dict = {}

    for key in target_key:
        choice_target_list_dict[key] = []

        for i in range(now_frames):
            choice_target_list_dict[key].append([])

            choice_target_list_dict[key][i].append(pasted_target_list_dict[key][i][selected_idx])


            if len(choice_target_list_dict[key][i]) > 0:
                if key == 'boxes':
                    choice_target_list_dict[key][i] = torch.stack(choice_target_list_dict[key][i], dim=0)

                if key == 'masks':
                    choice_target_list_dict[key][i] = torch.stack(choice_target_list_dict[key][i], dim=0)

        # choice_target_list_dict[key] = tuple(choice_target_list_dict[key])

    # label
    choice_target_list_dict['labels'] = []

    choice_target_list_dict['labels'].append(pasted_target_list_dict['labels'][selected_idx])

    choice_target_list_dict['labels'] = torch.stack(choice_target_list_dict['labels'], dim=0)
    # print('choice label:', choice_target_list_dict['labels'])


    return choice_target_list_dict

# 랜덤한 위치에 삽입하는 메소드
def mask_tracklet_randompos(image_size, pasted_image_list, choice_target_list_dict, now_frames):
    # 과정 :
    # 1. 인물 bbox만큼 pasted_image_list와 choice_target_list_dict을 crop - ymin, xmin, ymax, xmax
    # 2. crop한 값들을 resize
    # 3. resize한 값들을 원래의 image 크기로 pad
    # 결과물: 크기와 위치가 랜덤하게 설정된 mask tracklet 1개 생성
    left = []
    top = []
    width = []
    height = []
    for i in range(now_frames):
        left.append(math.floor((choice_target_list_dict['boxes'][i][0][0]))) # 개체 하나니까 일단 [0]
        top.append(math.floor((choice_target_list_dict['boxes'][i][0][1])))
        # out of index 때문에 계산 필요
        width_ = math.ceil(choice_target_list_dict['boxes'][i][0][2]) - left[i]
        height_ = math.ceil(choice_target_list_dict['boxes'][i][0][3]) - top[i]
        if width_ + left[i] > image_size[1]:
            width.append(image_size[1].tolist() - left[i])
            height.append(height_)
        elif height_ + top[i] > image_size[0]:
            width.append(width_)
            height.append(image_size[0].tolist() - top[i])
        else:
            width.append(width_)
            height.append(height_)
    # print('left:', left)
    # print('top:', top)
    # print('width:', width)
    # print('height:', height)
    region = (min(top), min(left), max(height), max(width))
    # print('region:', region)
    pasted_image_list, choice_target_list_dict = crop_clip_by_frame(pasted_image_list, choice_target_list_dict, region, now_frames)

    # resize - 줄이는 것만 됨 (크게 늘리려면 resize 후에 다시 pad 해야 함)
    size = min(max(height), max(width))
    # print('원래 mask 크기: ', pasted_image_list[0].size)
    pasted_image_list, choice_target_list_dict = resize_clip_by_frame(pasted_image_list, choice_target_list_dict, now_frames, size, max_size=image_size[0].tolist())
    # print('resize mask 크기: ', pasted_image_list[0].size)
    # pasted_image_list, choice_target_list_dict = check_clip_by_frame(pasted_image_list, choice_target_list_dict, now_frames) # 오류 발생해서 안 함

    # 위치 조정 및 원본 크기로 pad
    crop_image_width, crop_image_height = pasted_image_list[0].size
    pad_left = random.randint(0, (image_size[1] - crop_image_width))
    pad_top = random.randint(0, (image_size[0] - crop_image_height))
    pad_right = image_size[1] - (pad_left + crop_image_width)
    pad_bottom = image_size[0] - (pad_top + crop_image_height)

    pasted_image_list, choice_target_list_dict = pad_clip_by_frame(pasted_image_list, choice_target_list_dict, (pad_left, pad_top, pad_right, pad_bottom), now_frames)

    return pasted_image_list, choice_target_list_dict


# frame #1의 인물 #1의 bbox를 고려한 메소드
def mask_tracklet_usebbox_one(image_size, target_list_dict, choice_target_list_dict, pasted_image_list, now_frames):
    print('** mask_tracklet_usebbox_one **')
    # image_size[0] : height, image_size[1] : width

    # 과정 :
    # 1. 인물 bbox만큼 pasted_image_list와 choice_target_list_dict을 crop - ymin, xmin, ymax, xmax -> cuboid 생성
    # 2. 첫번째 사람의 bbox를 기준으로 왼쪽(left)과 오른쪽(right) 길이를 계산
    # 3. 계산한 영역 중 더 큰 영역을 선택
    # 4. if 선택한 영역 <= cuboid_width then 선택한 영역의 시작점에 위치
    #    else 선택한 영역을 벗어나지 않는 범위에서 랜덤한 위치 선택
    # 5. resize와 pad 진행
    # 결과물 : 위치가 어느정도 고려된 mask tracklet 1개 생성

    # choice_target_list_dict로부터 cuboid 생성
    crop_region = extract_cuboid_region(choice_target_list_dict['boxes'], image_size, now_frames)
    pasted_image_list, choice_target_list_dict = crop_clip_by_frame(pasted_image_list, choice_target_list_dict, crop_region, now_frames)

    # resize - 줄이는 것만 됨 (크게 늘리려면 resize 후에 다시 pad 해야 함)
    size = min(crop_region[2], crop_region[3])
    pasted_image_list, choice_target_list_dict = resize_clip_by_frame(pasted_image_list, choice_target_list_dict, now_frames, size, max_size=image_size[0].tolist())
    # pasted_image_list, choice_target_list_dict = check_clip_by_frame(pasted_image_list, choice_target_list_dict, now_frames) # 오류 발생해서 안 함

    # 위치 조정 및 원본 크기로 pad
    ####### 첫번째 프레임의 첫번째 인물의 bbox 좌우 영역을 계산하여 위치 조정
    region_list = [] # [0] : left, [1] : right
    left_region = math.floor(target_list_dict['boxes'][0][0][0]) # left_region : 0 ~ bbox x1
    right_region = image_size[1] - math.ceil(target_list_dict['boxes'][0][0][2]) # right_region : bbox x2 ~ image_size[1]

    region_list.append(left_region)
    region_list.append(right_region)

    crop_image_width, crop_image_height = pasted_image_list[0].size

    #### 더 넓은 구역 찾기
    index = region_list.index(max(region_list))

    if index == 0: # left region
        if region_list[index] <= crop_image_width:
            pad_left = 0
        else:
            pad_left = random.randint(0, left_region - crop_image_width)
    else: # right region
        if right_region <= crop_image_width: # 우측 구역이 tracklet 너비보다 작은 경우
            pad_left = image_size[1] - crop_image_width
        else:
            tmp = right_region - crop_image_width
            if region_list[index] <= tmp: # randint(최소, 최대) 범위 때문에 조건문으로 경우를 나눴음
                pad_left = random.randint(region_list[index], tmp)
            else:
                pad_left = random.randint(tmp, region_list[index])

    # pad_left = random.randint(0, (image_size[1] - crop_image_width))
    pad_top = random.randint(0, (image_size[0] - crop_image_height))
    pad_right = image_size[1] - (pad_left + crop_image_width)
    pad_bottom = image_size[0] - (pad_top + crop_image_height)

    pasted_image_list, choice_target_list_dict = pad_clip_by_frame(pasted_image_list, choice_target_list_dict, (pad_left, pad_top, pad_right, pad_bottom), now_frames)

    return pasted_image_list, choice_target_list_dict


# frame #1의 모든 인물의 bbox를 고려한 메소드
def mask_tracklet_usebbox_multi(image_size, target_list_dict, choice_target_list_dict, pasted_image_list, now_frames, blank_space):
    print('** mask_tracklet_usebbox_multi **')
    # image_size[0] : height, image_size[1] : width

    # 과정 :
    # 1. 인물 bbox만큼 pasted_image_list와 choice_target_list_dict을 crop - ymin, xmin, ymax, xmax -> cuboid 생성
    # 2. 첫번째 프레임 내 모든 사람의 bbox를 이용하여 좌측/bbox사이값/우측 영역을 계산
    # 3. 계산한 영역 중 제일 큰 영역을 선택
    # 4. [가로 위치] if 선택한 영역 <= cuboid_width then left - 시작점, max_gap - 중간지점, right - 끝점
    #    else 선택한 영역을 벗어나지 않는 범위에서 랜덤한 위치 선택
    # 5. [세로 위치] bbox 최저top, 최고top 사이에 들어갈 수 있도록 (전체를 차지하면 아래쪽에 위치할 수 있도록 추가할 계획..)
    # 6. resize와 pad 진행
    # 결과물 : 위치가 어느정도 고려된 mask tracklet 1개 생성

    # choice_target_list_dict로부터 cuboid 생성
    crop_region = extract_cuboid_region(choice_target_list_dict['boxes'], image_size, now_frames)
    pasted_image_list, choice_target_list_dict = crop_clip_by_frame(pasted_image_list, choice_target_list_dict, crop_region, now_frames)

    # resize - 줄이는 것만 됨 (크게 늘리려면 resize 후에 다시 pad 해야 함)
    size = min(crop_region[2], crop_region[3])
    pasted_image_list, choice_target_list_dict = resize_clip_by_frame(pasted_image_list, choice_target_list_dict, now_frames, size, max_size=image_size[0].tolist())
    # pasted_image_list, choice_target_list_dict = check_clip_by_frame(pasted_image_list, choice_target_list_dict, now_frames) # 오류 발생해서 안 함

    # 위치 조정 및 원본 크기로 pad
    ####### 첫번째 프레임의 모든 인물의 bbox 값을 이용하여 좌측/bbox사이값/우측 영역을 계산하여 위치 조정
    ############## 첫번째 프레임에 나타나는 인물 수 : n, 계산될 bbox 사이값 개수 : n-1

    #### 첫번째 프레임, 모든 인물의 bbox 정보 가져오기

    #### pad 계산
    crop_image_width, crop_image_height = pasted_image_list[0].size


    ''' # 기존에 빈 영역을 계산하던 부분 #
    #### 좌우 영역 계산 - 첫번째 프레임
    inst_box_list = copy.deepcopy(target_list_dict['boxes'][0]).tolist() # 첫번째 프레임에 있는 모든 인물의 bbox 정보 가져옴 : (인물수, 4(left,top,width,height))
    inst_box_list.sort() # 왼쪽->오른쪽 순서로 나오는 bbox 정렬 (가장 넓은 영역을 찾기 위한 작업이므로 img 순서는 안 바꿔도 됨)

    
    region_list = []  # [0] : left, [1] : right, [2] : max_gap[너비, a, b]
    inst_cnt = len(inst_box_list)  # 첫번째 프레임에 있는 인물 수

    left_region = math.floor(inst_box_list[0][0]) # left_region : 0 ~ bbox x1 좌측 구역 너비
    right_region = math.ceil(image_size[1] - inst_box_list[inst_cnt-1][2]) # right_region : bbox x2 ~ image_size[1] 우측 구역 너비
    region_list.append(left_region)
    region_list.append(right_region)

    '''
    #### [확장 220522] 모든 프레임에서 모든 인물의 bbox 정보 가져오기
    # inst_boxes_list = copy.deepcopy(target_list_dict['boxes']) # 모든 프레임에 있는 모든 인물의 bbox 정보 가져옴 : (인물수, 4(left,top,width,height)) * now_frames
    # for i in range(now_frames):
    #     (inst_boxes_list[i].tolist()).sort()

    #### [확장 220522] 좌우 영역 계산 - 모든 프레임 (굳이 해야 할까 싶어서 하다가 말았음)
    # region_list = []
    # inst_cnt = len(max(inst_boxes_list[i].tolist() for i in range(now_frames)))
    # print(inst_cnt)
    '''

    #### bbox 사이 영역 계산
    gap_list = [] # [0] : box 0 ~ box 1, [1] : 1~2, [2] : 2~1, ... [inst_cnt-1] : inst_cnt~inst_cnt-1
    for n in range(inst_cnt - 1):
        gap = inst_box_list[n+1][0] - inst_box_list[n][2]
        gap_list.append(gap)

    if inst_cnt != 1: # 인물이 1명이면 gap_list가 empty라서 max에서 오류 발생하기 때문에 조건 걸어줌
        i = gap_list.index(max(gap_list))
        max_gap = [math.floor(inst_box_list[i][2]), math.ceil(inst_box_list[i+1][0])] # [시작점, 끝점]
        max_gap_region = gap_list[i] # 너비
        region_list.append(max_gap_region)

    #### 가장 넓은 영역 찾기
    if inst_cnt == 1:
        index = region_list.index(max(region_list[0:2])) # gap_list가 비어있기 때문
    else :
        index = region_list.index(max(region_list))

    if index == 0: # left region
        if region_list[index] <= crop_image_width:
            pad_left = 0
        else:
            pad_left = random.randint(0, left_region - crop_image_width)
    elif index == 1: # right region
        if right_region <= crop_image_width: # 우측 영역이 tracklet 너비보다 작은 경우
            pad_left = image_size[1] - crop_image_width # 이 경우 pad_right이 0이 됨
        else:
            tmp = right_region - crop_image_width # 우측 영역에서 crop_image가 움직일 수 있는 범위
            if region_list[index] <= tmp: # randint(최소, 최대) 범위 때문에 조건문으로 경우를 나눴음
                pad_left = random.randint(region_list[index], tmp)
            else:
                pad_left = random.randint(tmp, region_list[index])
    else : # max gap region (max gap region일 때 tracklet이 삐져나가는 경우도 있지 않을까??)
        if max_gap_region <= crop_image_width:
            # pad_left = max_gap[0] # 구간 중 시작점에 위치
            pad_left = round(max_gap[0] - (crop_image_width - max_gap_region)/2) # tracklet이 중앙에 위치할 수 있도록 함
        else :
            pad_left = random.randint(max_gap[0], max_gap[1])
    '''

    # blank_space - # left : ['left', left_region], right : ['right', right_region], max_gap : ['max_gap', max_gap_region, max_gap(list)]
    if blank_space[0] == 'left':
        if blank_space[1] <= crop_image_width:
            pad_left = 0
        else:
            pad_left = random.randint(0, blank_space[1] - crop_image_width)
    elif blank_space[0] == 'right':
        if blank_space[1] <= crop_image_width: # 우측 영역이 tracklet 너비보다 작은 경우
            pad_left = image_size[1] - crop_image_width # 이 경우 pad_right이 0이 됨
        else:
            tmp = blank_space[1] - crop_image_width # 우측 영역에서 crop_image가 움직일 수 있는 범위
            if blank_space[1] <= tmp: # randint(최소, 최대) 범위 때문에 조건문으로 경우를 나눴음
                pad_left = random.randint(blank_space[1], tmp)
            else:
                pad_left = random.randint(tmp, blank_space[1])
    else:
        if blank_space[1] <= crop_image_width:
            # pad_left = max_gap[0] # 구간 중 시작점에 위치
            pad_left = round(blank_space[2][0] - (crop_image_width - blank_space[1])/2) # tracklet이 중앙에 위치할 수 있도록 함
        else :
            pad_left = random.randint(blank_space[2][0], blank_space[2][1])

    # pad_left = random.randint(0, (image_size[1] - crop_image_width))
    pad_top = random.randint(0, (image_size[0] - crop_image_height)) ## pad_top은 고려하지 않아도 충분히 현실성이 있다고 생각함
    pad_right = image_size[1] - (pad_left + crop_image_width)
    pad_bottom = image_size[0] - (pad_top + crop_image_height)

    pasted_image_list, choice_target_list_dict = pad_clip_by_frame(pasted_image_list, choice_target_list_dict, (pad_left, pad_top, pad_right, pad_bottom), now_frames)

    return pasted_image_list, choice_target_list_dict


# target clip의 빈 영역 계산하는 메소드
def cal_blank_space(size, target):
    h, w = size
    blank_space = []  # left : ['left', left_region], right : ['right', right_region], max_gap : ['max_gap', max_gap_region, max_gap(list)]

    # 첫번째 프레임에 있는 모든 인물의 bbox 정보 가져옴 : (인물수, 4(left,top,width,height))
    inst_box_list = copy.deepcopy(target['boxes'][0]).tolist()
    inst_box_list.sort()  # 왼쪽->오른쪽 순서로 나오는 bbox 정렬 (가장 넓은 영역을 찾기 위한 작업이므로 img 순서는 안 바꿔도 됨)

    # 좌우 영역 계산 - 첫번째 프레임
    region_list = []  # [0] : left, [1] : right, [2] : max_gap[너비, a, b]
    inst_cnt = len(inst_box_list)  # 첫번째 프레임에 있는 인물 수

    left_region = math.floor(inst_box_list[0][0])  # left_region : 0 ~ bbox x1 좌측 구역 너비
    right_region = math.ceil(w - inst_box_list[inst_cnt - 1][2])  # right_region : bbox x2 ~ image_size[1] 우측 구역 너비
    region_list.append(left_region)
    region_list.append(right_region)

    #### bbox 사이 영역 계산
    gap_list = []  # [0] : box 0 ~ box 1, [1] : 1~2, [2] : 2~1, ... [inst_cnt-1] : inst_cnt~inst_cnt-1
    for n in range(inst_cnt - 1):
        gap = inst_box_list[n + 1][0] - inst_box_list[n][2]
        gap_list.append(gap)

    if inst_cnt != 1:  # 인물이 1명이면 gap_list가 empty라서 max에서 오류 발생하기 때문에 조건 걸어줌
        i = gap_list.index(max(gap_list))
        max_gap = [math.floor(inst_box_list[i][2]), math.ceil(inst_box_list[i + 1][0])]  # [시작점, 끝점]
        max_gap_region = gap_list[i]  # 너비
        region_list.append(max_gap_region)

    #### 가장 넓은 영역 찾기
    if inst_cnt == 1:
        index = region_list.index(max(region_list[0:2]))  # gap_list가 비어있기 때문
    else:
        index = region_list.index(max(region_list))

    #### 매개변수로 넘길 정보
    if index == 0:
        blank_space.append('left')
        blank_space.append(left_region)
    elif index == 1:
        blank_space.append('right')
        blank_space.append(right_region)
    else:
        blank_space.append('max_gap')
        blank_space.append(max_gap_region)
        blank_space.append(max_gap)

    return blank_space

# choice_target_list_dict로부터 전체 프레임에서 cuboid가 움직이는 영역 추출 (for cropping cuboid from source clip)
def extract_cuboid_region(choice_target_list_dict_boxes, image_size, now_frames):
    left = []
    top = []
    right = []
    bottom = []

    for i in range(now_frames):
        left.append(choice_target_list_dict_boxes[i][0][0]) # 개체 하나만 선택하니까 [0]
        top.append(choice_target_list_dict_boxes[i][0][1])

        right.append(choice_target_list_dict_boxes[i][0][2])
        bottom.append(choice_target_list_dict_boxes[i][0][3])

    left_ = math.floor(min(left))
    top_ = math.floor(min(top))
    right_ = math.ceil(max(right))
    bottom_ = math.ceil(max(bottom))

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

# ============== 사용 안 하는 메소드 =============== #
# clip 단위 (사용 X)
def crop_instance_clip(choice_target_list_dict, image_list, pasted_image_list, pasted_target, now_frames):
    check = T.Check()

    # crop
    top = math.ceil(choice_target_list_dict['boxes'][0][0][1].item())
    left = math.ceil(choice_target_list_dict['boxes'][0][0][0].item())
    width = math.ceil(choice_target_list_dict['boxes'][0][0][2].item())
    height = math.ceil(choice_target_list_dict['boxes'][0][0][3].item())

    pasted_mask_width = width - left # box width: w - x
    pasted_mask_height = height - top # box height: h - y

    pasted_image_list, pasted_target = T.crop(pasted_image_list, pasted_target,
                                            (top, left, pasted_mask_height, pasted_mask_width))

    # resize
    size = min(pasted_mask_height, pasted_mask_width)
    pasted_image_list, pasted_target = T.resize(pasted_image_list, pasted_target, size, max_size=720)
    # print('resized image size:', pasted_image_list[0].size)
    pasted_image_list, pasted_target = check(pasted_image_list, pasted_target, now_frames)

    # pad
    image_width, image_height = image_list[0].size
    crop_image_width, crop_image_height = pasted_image_list[0].size

    # pad_left = random.randint(0, (image_width - crop_image_width))
    # pad_top = random.randint(0, (image_height - crop_image_height))
    # pad_right = image_width - (pad_left + crop_image_width)
    # pad_bottom = image_height - (pad_top + crop_image_height)
    # pasted_image_list, pasted_target = pad_clip(pasted_image_list, pasted_target,
    #                                         (pad_left, pad_top, pad_right, pad_bottom))

    return pasted_image_list, pasted_target

# Check 그대로 복붙한 것 (구현 X 사용 X) - 무슨 용도인지 모르겠음, 애초에 이걸 쓰면 잘림
def check_clip_by_frame(img, target, now_frames):
        fields = ["labels"]  # , "area", "iscrowd"]
        if "boxes" in target:
            fields.append("boxes")
        if "masks" in target:
            fields.append("masks")

        ### check if box or mask still exist after transforms
        if "boxes" in target or "masks" in target:
            if "boxes" in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            num_frames = now_frames

            class_keep = []
            if False in keep:
                for k in range(len(keep)):
                    if not keep[k]:
                        target['boxes'][k] = target['boxes'][k] // 1000

            for inst in range(len(target['labels'])):
                inst_range = [k for k in range(inst * num_frames, inst * num_frames + num_frames)]
                keep_inst = keep[inst_range].any()  # if exist, keep all frames
                keep[inst_range] = keep_inst
                class_keep.append(keep_inst)
            class_keep = torch.tensor(class_keep)

            for field in fields:
                if field == 'labels':
                    target[field] = target[field][class_keep]
                else:
                    target[field] = target[field][keep]

        return img, target

## 시간적으로 중간에 끼워 넣으려고 함... (구현 X 사용 X)
def set_time(clip, target, start_time, end_time):
    # time : (start time, end time)

    return 0

## 추가하는 이미지가 여러 개 (사용 X)
def images_copy_paste(img, paste_imgs, alpha, blend=True, sigma=1):
    # img: Image
    # paste_img: list

    img = F.to_tensor(img).to(torch.float32).numpy() # 한 개

    for paste_img in paste_imgs:
        tmp_img = F.to_tensor(paste_img).to(torch.float32).numpy()
        if alpha is not None:
            if blend:
                alpha = gaussian(alpha, sigma=sigma, preserve_range=True)
            alpha = alpha[None, ...]
            img = tmp_img * alpha + img * (1 - alpha)

    return img

## 클래스 2~6만 여러 개 붙여 넣도록 선별 (사용 X)
def select_pasted_instance_class(pasted_target_list_dict, target_key):
    '''
    :param pasted_target_list_dict:
    :param target_key:
    :return:
    '''
    instance_cnt = len(pasted_target_list_dict['boxes'][0])
    random_choice = np.random.randint(1, 2, size=instance_cnt).tolist()
    random_choice = [i for i, x in enumerate(random_choice) if x == 1]
    choice_target_list_dict = {}

    # class 선별
    random_choice_ = []
    for idx in random_choice:
        if pasted_target_list_dict['labels'][idx] not in [1, 7]:
            random_choice_.append(idx)

    for key in target_key:
        choice_target_list_dict[key] = []

        for i in range(5):
            choice_target_list_dict[key].append([])

            for idx in random_choice_:
                choice_target_list_dict[key][i].append(pasted_target_list_dict[key][i][idx])

            if len(choice_target_list_dict[key][i]) > 0:
                if key == 'boxes':
                    choice_target_list_dict[key][i] = torch.stack(choice_target_list_dict[key][i], dim=0)

                if key == 'masks':
                    choice_target_list_dict[key][i] = torch.stack(choice_target_list_dict[key][i], dim=0)

        # choice_target_list_dict[key] = tuple(choice_target_list_dict[key])

    # label
    if len(random_choice_) > 0:
        choice_target_list_dict['labels'] = []

        for idx in random_choice_:
            choice_target_list_dict['labels'].append(pasted_target_list_dict['labels'][idx])

        choice_target_list_dict['labels'] = torch.stack(choice_target_list_dict['labels'], dim=0)

    else:
        choice_target_list_dict = None

    return choice_target_list_dict

# 구현하다가 말았음
def cal_occluded(target, pasted_target, occluded_inst_threshold):
    # mask area로 계산 -> mask 계산 이후에 bbox를 추출하는 것처럼, mask 계산 이후에 area 추출한 후 여기서 occ 계산
    gt_area = target['area']
    pt_area = pasted_target['area']