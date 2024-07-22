import json
import os

if __name__ == '__main__':
    sample_frame = 5

    # train
    json_path = '../../miseang_vis_dataset/annotations/instances_train_sub_face.json'
    des_root = '../../miseang_vis_dataset/annotations'
    name = 'instances_train_sub_face_' + str(sample_frame) + 'frames.json'

    with open(json_path, 'r') as json_name:
        json_data = json.load(json_name)

    new_vid_json = {
        'info': json_data['info'],
        'licenses': json_data['licenses'],
        'videos': [],
        'categories': json_data['categories'],
        'annotations': []
    }

    # 1. videos 5 단위로 나누기
    id = 1
    for idx, video in enumerate(json_data['videos']):
        print('video:', idx + 1)
        video_id = 1
        for i in range(0, len(video['file_names']), sample_frame):
            new_video = {}
            new_video["width"] = video['width']
            new_video["length"] = video['length']
            new_video["date_captured"] = video['date_captured']
            new_video["license"] = video["license"]
            new_video["flickr_url"] = video["flickr_url"]
            new_file_names = []
            if (i + sample_frame) <= len(video['file_names']):
                for j in range(i, i + sample_frame):
                    index = video['file_names'][j].find('/')
                    file_name = video['file_names'][j][:index] + '_' + str(video_id).zfill(4) + video['file_names'][j][
                                                                                                index:]
                    new_file_names.append(file_name)
            else:
                for j in range(i, len(video['file_names'])):
                    index = video['file_names'][j].find('/')
                    file_name = video['file_names'][j][:index] + '_' + str(video_id).zfill(4) + video['file_names'][j][
                                                                                                index:]
                    new_file_names.append(file_name)
                    # new_file_names.append(video['file_names'][j])
            video_id += 1
            new_video["file_names"] = new_file_names
            new_video["id"] = id
            id += 1
            # 바뀐 videos의 id와 annotations 값을 일치시키기 위해 original_id와 range_id를 임의로 추가함
            new_video["original_id"] = video["id"]
            new_video["range_id"] = i
            new_video["coco_url"] = video["coco_url"]
            new_video["height"] = video["height"]
            new_vid_json["videos"].append(new_video)

    # 2. annotations 5 단위로 나누기 (bboxes, segmentations, areas)
    instance_id = 1
    for idx, anno in enumerate(json_data['annotations']):
        print('annotation:', idx + 1)
        for i in range(0, len(anno['areas']), sample_frame):
            new_anno = {}
            new_anno["height"] = anno['height']
            new_anno["width"] = anno['width']
            new_anno["length"] = anno['length']
            ##### video id: 5 프레임 단위로 자른 video id와 일치해야 함 #####
            for n_vid in new_vid_json["videos"]:
                if n_vid["original_id"] == anno["video_id"] and n_vid["range_id"] == i:
                    # print('new anno video id', n_vid['id'])
                    new_anno["video_id"] = n_vid['id']
            new_anno["iscrowd"] = anno["iscrowd"]
            new_anno["id"] = instance_id
            instance_id += 1
            new_seg = []
            new_box = []
            new_area = []
            new_face = []
            if (i + sample_frame) <= len(anno['areas']):
                for j in range(i, i + sample_frame):
                    new_seg.append(anno['segmentations'][j])
                    new_box.append(anno['bboxes'][j])
                    new_area.append(anno['areas'][j])
                    try:
                        new_face.append(anno['detected_face'][j])
                    except KeyError:
                        continue
            else:
                for j in range(i, len(anno['areas'])):
                    new_seg.append(anno['segmentations'][j])
                    new_box.append(anno['bboxes'][j])
                    new_area.append(anno['areas'][j])
                    try:
                        new_face.append(anno['detected_face'][j])
                    except KeyError:
                        continue
            new_anno["segmentations"] = new_seg
            new_anno["bboxes"] = new_box
            new_anno["areas"] = new_area
            new_anno["category_id"] = anno['category_id']
            if len(new_face) != 0:
                new_anno["detected_face"] = new_face
            new_vid_json["annotations"].append(new_anno)

    # 3. 임의로 추가한 original_id와 range_id 삭제
    for n_dict in new_vid_json['videos']:
        n_dict.pop('original_id')
        n_dict.pop('range_id')

    outpath = os.path.join(des_root, name)

    with open(outpath, 'w') as save_file:
        json.dump(new_vid_json, save_file)
