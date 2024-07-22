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
from datasets import transforms_copy_paste_using_getitem


class MiseangDataset:
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames, copy_paste_transforms, category, size, position):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.copy_paste_transforms = copy_paste_transforms
        self.category = category
        self.size = size
        self.position = position

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

            # # 적용 확률
            r = random.random()
            p = 0.5

            # if self._transforms is not None and self.copy_paste_transforms is not None and r >= p: # 50% 확률
            if self._transforms is not None and self.copy_paste_transforms is None: # 100% 확률
                img, target = self._transforms(img, target, num_frames)

            # if self._transforms is not None and self.copy_paste_transforms is not None and r < p: # 50% 확률
            if self._transforms is not None and self.copy_paste_transforms is not None: # 100% 확률
                ## 220526 수정 및 추가
                '''
                # 1. target clip 빈 공간 계산
                blank_space = self.cal_blank_space(target)
                '''

                ## 220608 재수정 - blank_space를 여기서 계산하면 resize, flip의 문제가 발생하여 transforms에서 하도록 변경함
                np.random.seed(int(time.time()))
                paste_idx = random.randint(0, self.__len__() - 1)  # source clip 랜덤 선택
                target = self.split_target(target)  # num_frames로 boxes랑 masks 나눴음 -> 나중에 copy_paste_transforms에서 다시 stack함
                '''
                # 2. 계산한 빈 공간에 따른 적절한 source clip 선택하기
                # pasted_img, pasted_target = self.get_paste_img_target_by_blank_space(paste_idx, target, blank_space) # target clip의 빈 공간을 고려한 get_paste_img_target
                '''
                pasted_img, pasted_target = self.get_paste_img_target(paste_idx, target) # 빈 공간을 고려하지 않고 target clip 인물의 크기 정보만을 이용한 get_paste_img_target

                # * target의 boxes와 masks를 unsplit
                target = self.unsplit_target(target)

                # 3. target clip에 source clip의 tracklet 삽입
                img, target = self.copy_paste_transforms(img, target, pasted_img, pasted_target, num_frames)

            if len(target['labels']) == 0: # None instance
                idx = random.randint(0, self.__len__()-1)
            else:
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

    # target clip의 빈 공간을 고려해서 source clip을 선택하는 get_paste_img_target - 현재 사용 X
    def get_paste_img_target_by_blank_space(self, idx, target_list_dict, blank_space):
        instance_check = False
        # 랜덤 idx에 따른 source clip 정보 가져오기
        while not instance_check:
            vid, frame_id = self.img_ids[idx]  ## ex) vid:0, frame_id:32
            vid_id = self.vid_infos[vid]['id']
            img = []
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
                img.append(Image.open(img_path).convert('RGB'))
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)

            target = {'video_id': vid, 'annotations': target}
            target_inds = inds  ## [0,1,2,3,4]
            target = self.prepare(img[0], target, target_inds, sample_inds=sample_indx)

            if len(target['labels']) == 0:  # None instance
                idx = random.randint(0, self.__len__() - 1)
                continue

            ## 220526 수정
            target = self.split_target(target)

            # source clip 내 인물 고려
            # selected_target = self.select_instance_1stbox(target, target_list_dict, blank_space) # selected_target = choice_target_list_dict
            selected_target = self.select_instance_allbox(target, target_list_dict, blank_space)

            # 적절한 인물이 없는 경우 다음으로 넘기기
            if selected_target is None:
                idx = random.randint(0, self.__len__() - 1)
                continue
            else:
                instance_check = True

        return img, selected_target

    # source clip을 선택하는 get_paste_img_target - blank_space 고려 X, target의 인물 크기만 고려 후 선택
    def get_paste_img_target(self, idx, target_list_dict):
        instance_check = False
        # 랜덤 idx에 따른 source clip 정보 가져오기
        while not instance_check:
            vid, frame_id = self.img_ids[idx]  ## ex) vid:0, frame_id:32
            vid_id = self.vid_infos[vid]['id']
            img = []
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
                img.append(Image.open(img_path).convert('RGB'))
            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)

            target = {'video_id': vid, 'annotations': target}
            target_inds = inds  ## [0,1,2,3,4]
            target = self.prepare(img[0], target, target_inds, sample_inds=sample_indx)

            if len(target['labels']) == 0:  # None instance
                idx = random.randint(0, self.__len__() - 1)
                continue

            ## 220526 수정
            target = self.split_target(target) # 여기서의 target은 pasted_target을 의미함

            # source clip 내 인물 고려
            selected_target = self.select_instance_allbox_no_blankspace(target, target_list_dict) # target - pasted_target, target_list_dict - orig target

            # 적절한 인물이 없는 경우 다음으로 넘기기
            if selected_target is None:
                idx = random.randint(0, self.__len__() - 1)
                continue
            else:
                instance_check = True

        return img, selected_target

    # target clip의 첫번째 프레임에서의 빈 공간 계산 - 현재 사용 X
    def cal_blank_space(self, target):
        h, w = target['size']
        blank_space = [] # left : ['left', left_region], right : ['right', right_region], max_gap : ['max_gap', max_gap_region, max_gap(list)]

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

    # source clip 내 인물 고려 - 첫번째 프레임 정보만 이용 - 현재 사용 X
    def select_instance_1stbox(self, pasted_target_list_dict, target_list_dict, blank_space):
        '''
        1. source clip 인물 중 장그래와 someone을 제외한 인물들이 있는지 확인

        2. 인물(들)이 있으면 target clip 내 인물(들)의 크기와 비교
        -> 일단 첫번째 프레임에서만의 크기를 비교
        -> pasted_target의 비율과 target의 비율을 계산했을 때 차이가 5% 이하라면 선택

        3. 크기 비교 후 적당한 게 있을 때, 선택된 인물이 빈 공간보다 과도하게 더 크다면 다음으로 넘기기 (작거나 적당하게 크면 최종 선택)
        -> cal_blank_space에서 첫번째 프레임에서의 빈 공간을 계산하니까 빈 공간의 크기 비교도 첫번째 프레임에서만 이루어지도록 함
        -> (selected_target / blank_space) * 100 계산 시 100보다 작으면 blank_space가 더 큰 거니까 당연히 가능, 따라서 120% 이하라면 최종 선택 (이 비율은 수정 가능)

        단계가 넘어가면서 적당한 인물이 없으면 return None해서 다음 source clip을 선택하도록 함

        if (return choice_target_list_dict) then target 인물과 크기가 비슷하고 blank_space에 들어갈 수 있는 인물 1명 값만 return됨
        '''

        category = [2, 3, 4, 5, 6]
        target_key = ['boxes', 'masks']
        target_key_ = ['image_id', 'orig_size', 'size']

        # 1. source clip 인물 중 장그래와 someone을 제외한 인물들이 있는지 확인
        instance_choice = []

        for i, label in enumerate(pasted_target_list_dict['labels']):
            if label in category:
                instance_choice.append(i) # pasted_target_list_dict 값 중 category에 있는 인물의 id 전부 append

        ## tmp_choice 중 랜덤으로 선택 없이 원하는 카테고리면 전부 선택한 후 greedy search
        if not instance_choice: # 1. source clip 인물 중 장그래와 someone을 제외한 인물이 없어서 return None
            return None
        # 1. END

        # ============================
        # blank_space - left : ['left', left_region], right : ['right', right_region], max_gap : ['max_gap', max_gap_region, max_gap(list)]

        selected_idx = -1 # 적당한 인물이 선택되면 해당 index로 변경, for문 이후에도 -1이면 괜찮은 인물이 없다는 의미
        selected = False
        target_instance_cnt = len(target_list_dict['boxes'][0]) # 인물이 1명일 때랑 여러 명일 때랑 형식이 달라서 구분해줘야 함
        # pasted_target_instance_cnt = len(instance_choice)
        image_width = pasted_target_list_dict['size'][1]

        for idx in instance_choice:
            # pasted_target의 가로 세로 추출
            pasted_target = pasted_target_list_dict['boxes'][0][idx]
            p_width, p_height = pasted_target[2] - pasted_target[0], pasted_target[3] - pasted_target[1]
            p_width_percent = (p_width / image_width) * 100

            for i, target in enumerate(target_list_dict['boxes'][0]):
                ## i는 사용하지 않으며, 반복문을 통해 첫번째 프레임에 있는 모든 인물의 정보를 사용함
                # target의 가로 세로 추출
                t_width, t_height = target[2] - target[0], target[3] - target[1]
                t_width_percent = (t_width / image_width) * 100

                # 2. 인물(들)이 있으면 target clip 내 인물(들)의 크기와 비교
                # 비율 계산 후 5% 차이 이하인 경우를 선택
                # if abs(t_width_percent - p_width_percent) <= 10:
                if abs(t_width_percent - p_width_percent) <= 5:
                    # 2. END
                    # 3. 크기 비교 후 적당한 게 있을 때, 선택된 인물이 빈 공간보다 과도하게 더 크다면 다음으로 넘기기 (작거나 적당하게 크면 최종 선택)
                    # 현재 적당한 크기 : 계산한 비율이 120% 이하인 경우
                    if (p_width / blank_space[1]) * 100 <= 120:
                        selected_idx = idx
                        selected = True
                        break
                    # 3. END
                # 크기가 5% 차이 이상인 경우 또는 비율 계산 후 120% 이상인 경우는 continue
            if selected:
                break

        if selected_idx == -1: # source clip에 적당한 인물이 없는 경우임
            return None

        # 최종 선택된 인물을 choice_target_list_dict로 return
        # =============================

        choice_target_list_dict = {}

        for key in target_key:
            # choice_target_list_dict[key] = []
            tmp_list = []

            for i in range(self.num_frames):
                tmp_list.append(pasted_target_list_dict[key][i][selected_idx])

            # 기존의 target 모습으로 변경
            tmp_list = torch.stack(tmp_list, dim=0)
            choice_target_list_dict[key] = tmp_list

        # label
        choice_target_list_dict['labels'] = []
        choice_target_list_dict['labels'].append(pasted_target_list_dict['labels'][selected_idx])
        choice_target_list_dict['labels'] = torch.stack(choice_target_list_dict['labels'], dim=0)

        # others
        for key_ in target_key_:
            choice_target_list_dict[key_] = pasted_target_list_dict[key_]

        return choice_target_list_dict

    # source clip 내 인물 고려 - 전체 프레임 정보의 평균을 이용 - 현재 사용 X
    def select_instance_allbox(self, pasted_target_list_dict, target_list_dict, blank_space):

        '''
        1. source clip 인물 중 장그래와 someone을 제외한 인물들이 있는지 확인

        2. 인물(들)이 있으면 target clip 내 인물(들)의 크기와 비교
        -> 전체 프레임 bbox 정보의 평균 값을 이용
        -> pasted_target의 비율과 target의 비율을 계산했을 때 차이가 5% 이하라면 선택

        3. 크기 비교 후 적당한 게 있을 때, 선택된 인물이 빈 공간보다 과도하게 더 크다면 다음으로 넘기기 (작거나 적당하게 크면 최종 선택)
        -> cal_blank_space에서 첫번째 프레임에서의 빈 공간을 계산하니까 빈 공간의 크기 비교도 첫번째 프레임에서만 이루어지도록 함
        -> (selected_target / blank_space) * 100 계산 시 100보다 작으면 blank_space가 더 큰 거니까 당연히 가능, 따라서 120% 이하라면 최종 선택 (이 비율은 수정 가능)

        단계가 넘어가면서 적당한 인물이 없으면 return None해서 다음 source clip을 선택하도록 함

        if (return choice_target_list_dict) then target 인물과 크기가 비슷하고 blank_space에 들어갈 수 있는 인물 1명 값만 return됨
        '''

        category = [2, 3, 4, 5, 6]
        target_key = ['boxes', 'masks']
        target_key_ = ['image_id', 'orig_size', 'size']

        # 1. source clip 인물 중 장그래와 someone을 제외한 인물들이 있는지 확인
        instance_choice = []

        for i, label in enumerate(pasted_target_list_dict['labels']):
            if label in category:
                instance_choice.append(i) # pasted_target_list_dict 값 중 category에 있는 인물의 id 전부 append

        ## tmp_choice 중 랜덤으로 선택 없이 원하는 카테고리면 전부 선택한 후 greedy search
        if not instance_choice: # 1. source clip 인물 중 장그래와 someone을 제외한 인물이 없어서 return None
            return None
        # 1. END
        ## blank_space - left : ['left', left_region], right : ['right', right_region], max_gap : ['max_gap', max_gap_region, max_gap(list)]
        selected_idx = -1 # 적당한 인물이 선택되면 해당 index로 변경, for문 이후에도 -1이면 괜찮은 인물이 없다는 의미
        selected = False
        target_instance_cnt = len(target_list_dict['boxes'][0]) # 인물이 1명일 때랑 여러 명일 때랑 형식이 달라서 구분해줘야 함
        image_width = pasted_target_list_dict['size'][1]

        # target clip 전체 프레임 속 모든 인물의 box width 비율 정보를 리스트에 저장
        t_width_list = [0] * target_instance_cnt # 평균 계산 후 image_width로 나눠서 비율 저장
        for frame, target in enumerate(target_list_dict['boxes']): # target[0-4] : tensor([target_instance_cnt, 4])
            for i, box in enumerate(target):
                t_width_list[i] += box[2] - box[0] # target의 가로 추출
                if frame == self.num_frames - 1:
                    t_width_list[i] /= self.num_frames
                    t_width_list[i] = (t_width_list[i] / image_width) * 100

        # source clip 전체 프레임 속 모든 인물의 box width 정보를 리스트에 저장
        p_width_list = [0] * len(instance_choice)
        for i, idx in enumerate(instance_choice):
            for frame in range(self.num_frames):
                p_width_list[i] += pasted_target_list_dict['boxes'][frame][idx][2] - pasted_target_list_dict['boxes'][frame][idx][0]
                if frame == self.num_frames - 1:
                    p_width_list[i] /= self.num_frames

        for idx, p_width in enumerate(p_width_list):
            p_width_percent = (p_width / image_width) * 100
            for t_width_percent in t_width_list: # target 인물 크기 차례로 비교
                # 2. 인물(들)이 있으면 target clip 내 인물(들)의 크기와 비교
                # 비율 계산 후 5% 차이 이하인 경우를 선택
                # if abs(t_width_percent - p_width_percent) <= 10:
                if abs(t_width_percent - p_width_percent) <= 5:
                    # 2. END
                    # 3. 크기 비교 후 적당한 게 있을 때, 선택된 인물이 빈 공간보다 과도하게 더 크다면 다음으로 넘기기 (작거나 적당하게 크면 최종 선택)
                    # 현재 적당한 크기 : 계산한 비율이 120% 이하인 경우
                    if (p_width / blank_space[1]) * 100 <= 120:
                        selected_idx = instance_choice[idx]
                        selected = True
                        break
                    # 3. END
                # 크기가 5% 차이 이상인 경우 또는 비율 계산 후 120% 이상인 경우는 continue
            if selected:
                break

        if selected_idx == -1: # source clip에 적당한 인물이 없는 경우임
            return None

        # 최종 선택된 인물을 choice_target_list_dict로 return
        choice_target_list_dict = {}

        for key in target_key:
            # choice_target_list_dict[key] = []
            tmp_list = []

            for i in range(self.num_frames):
                tmp_list.append(pasted_target_list_dict[key][i][selected_idx])

            # 기존의 target 모습으로 변경
            tmp_list = torch.stack(tmp_list, dim=0)
            choice_target_list_dict[key] = tmp_list

        # label
        choice_target_list_dict['labels'] = []
        choice_target_list_dict['labels'].append(pasted_target_list_dict['labels'][selected_idx])
        choice_target_list_dict['labels'] = torch.stack(choice_target_list_dict['labels'], dim=0)

        # others
        for key_ in target_key_:
            choice_target_list_dict[key_] = pasted_target_list_dict[key_]

        return choice_target_list_dict

    # select_instance_allbox와 동일, but blank_space 계산 X
    def select_instance_allbox_no_blankspace(self, pasted_target_list_dict, target_list_dict):

        '''
        1. source clip 인물 중 장그래와 someone을 제외한 인물들이 있는지 확인

        2. 인물(들)이 있으면 target clip 내 인물(들)의 크기와 비교
        -> 전체 프레임 bbox 정보의 평균 값을 이용
        -> pasted_target의 비율과 target의 비율을 계산했을 때 차이가 5% 이하라면 선택

        3. 크기 비교 후 적당한 게 있을 때, 선택된 인물이 빈 공간보다 과도하게 더 크다면 다음으로 넘기기 (작거나 적당하게 크면 최종 선택)
        -> cal_blank_space에서 첫번째 프레임에서의 빈 공간을 계산하니까 빈 공간의 크기 비교도 첫번째 프레임에서만 이루어지도록 함
        -> (selected_target / blank_space) * 100 계산 시 100보다 작으면 blank_space가 더 큰 거니까 당연히 가능, 따라서 120% 이하라면 최종 선택 (이 비율은 수정 가능)

        단계가 넘어가면서 적당한 인물이 없으면 return None해서 다음 source clip을 선택하도록 함

        if (return choice_target_list_dict) then target 인물과 크기가 비슷하고 blank_space에 들어갈 수 있는 인물 1명 값만 return됨
        '''

        valid_category = [2, 3, 4, 5, 6]
        target_key = ['boxes', 'masks']
        target_key_ = ['image_id', 'orig_size', 'size']

        # 1. source clip 인물 중 장그래와 someone을 제외한 인물들이 있는지 확인
        instance_choice = []
        target_category = target_list_dict['labels'] # target에 있는 인물이면 제외

        for i, label in enumerate(pasted_target_list_dict['labels']):
            if self.category is True:
                if label in valid_category and label not in target_category:
                    instance_choice.append(i) # pasted_target_list_dict 값 중 category에 있는 인물의 id 전부 append
            else:
                instance_choice.append(i)

        ## tmp_choice 중 랜덤으로 선택 없이 원하는 카테고리면 전부 선택한 후 greedy search
        if not instance_choice: # 1. source clip 인물 중 장그래와 someone을 제외한 인물이 없어서 return None
            return None
        # 1. END

        # 2. target clip 인물과 source clip 인물 크기 비교
        selected_idx = -1  # 적당한 인물이 선택되면 해당 index로 변경, for문 이후에도 -1이면 괜찮은 인물이 없다는 의미
        if self.size is True: # 인물 크기 비교
            selected = False
            target_instance_cnt = len(target_list_dict['boxes'][0]) # 인물이 1명일 때랑 여러 명일 때랑 형식이 달라서 구분해줘야 함
            image_width = pasted_target_list_dict['size'][1]

            # target clip 전체 프레임 속 모든 인물의 box width 비율 정보를 리스트에 저장
            t_width_list = [0] * target_instance_cnt # 평균 계산 후 image_width로 나눠서 비율 저장
            for frame, target in enumerate(target_list_dict['boxes']): # target[0-4] : tensor([target_instance_cnt, 4])
                for i, box in enumerate(target):
                    t_width_list[i] += box[2] - box[0] # target의 가로 추출
                    if frame == self.num_frames - 1:
                        t_width_list[i] /= self.num_frames
                        t_width_list[i] = (t_width_list[i] / image_width) * 100

            # source clip 전체 프레임 속 모든 인물의 box width 정보를 리스트에 저장
            p_width_list = [0] * len(instance_choice)
            for i, idx in enumerate(instance_choice):
                for frame in range(self.num_frames):
                    p_width_list[i] += pasted_target_list_dict['boxes'][frame][idx][2] - pasted_target_list_dict['boxes'][frame][idx][0]
                    if frame == self.num_frames - 1:
                        p_width_list[i] /= self.num_frames
                        p_width_list[i] = (p_width_list[i] / image_width) * 100

            for idx, p_width_percent in enumerate(p_width_list):
                for t_width_percent in t_width_list: # target 인물 크기 차례로 비교
                    # 2. 인물(들)이 있으면 target clip 내 인물(들)의 크기와 비교
                    # 비율 계산 후 5% 차이 이하인 경우를 선택
                    # if abs(t_width_percent - p_width_percent) <= 10:
                    # if abs(t_width_percent - p_width_percent) <= 5:
                    if abs(t_width_percent - p_width_percent) <= 3: # 220720
                        # 2. END
                        selected_idx = instance_choice[idx]
                        selected = True
                        break
                    # 크기가 5% 차이 이상인 경우는 continue
                if selected:
                    break
        else: # 인물 크기 비교 X
            selected_idx = random.choice(instance_choice)

        if selected_idx == -1: # source clip에 적당한 인물이 없는 경우임
            return None

        # 최종 선택된 인물을 choice_target_list_dict로 return
        choice_target_list_dict = {}

        for key in target_key:
            # choice_target_list_dict[key] = []
            tmp_list = []

            for i in range(self.num_frames):
                tmp_list.append(pasted_target_list_dict[key][i][selected_idx])

            # 기존의 target 모습으로 변경
            tmp_list = torch.stack(tmp_list, dim=0)
            choice_target_list_dict[key] = tmp_list

        # label
        choice_target_list_dict['labels'] = []
        choice_target_list_dict['labels'].append(pasted_target_list_dict['labels'][selected_idx])
        choice_target_list_dict['labels'] = torch.stack(choice_target_list_dict['labels'], dim=0)

        # others
        for key_ in target_key_:
            choice_target_list_dict[key_] = pasted_target_list_dict[key_]

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


def make_copy_paste_transforms(size, position):
    cp_normalize = T.Compose([
        transforms_copy_paste_using_getitem.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    normalize = normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]
    flip_transforms = T.RandomHorizontalFlip()
    resize_transforms = transforms_copy_paste_using_getitem.RandomResize(max_size=768)
    check = T.Check()
    cp_transforms = transforms_copy_paste_using_getitem.CopyAndPaste()
    cp_compose = transforms_copy_paste_using_getitem.CopyPasteCompose(flip_transforms, resize_transforms, check, cp_transforms,
                                                        cp_normalize, normalize, scales, size, position)

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
        print('use Miseang dataset - miseang_using_getitem.py')

        if args.manual_cp is True and image_set == 'train':
            copy_paste_tfs = make_copy_paste_transforms(size=args.size, position=args.position)
            print('make copy paste transforms - manual cp')
        else:
            copy_paste_tfs = None

        print('category :', args.category, ', size :', args.size, ', position :', args.position)

        dataset = MiseangDataset(img_folder, ann_file, transforms=make_miseang_transforms(image_set), return_masks=args.masks,
                                 num_frames=args.num_frames, copy_paste_transforms=copy_paste_tfs, category=args.category,
                                 size=args.size, position=args.position)

    return dataset