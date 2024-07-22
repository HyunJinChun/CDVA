import json
import os


def coco_info2vid(description='Miseang VIS', url='http://ailab.kyonggi.ac.kr/', version='1.0', year=2022,
                  contributor='KGU AI Lab', date_created='2022-02-01 01:01:01:000001'):
    info = {
        'description': description,
        'url': url,
        'version': version,
        'year': year,
        'contributor': contributor,
        'date_created': date_created
    }

    return info


def coco_licenses2vid(url='http://ailab.kyonggi.ac.kr/', id=1, name='KGU AI Lab'):
    license_list = []

    license = {
        'url': url,
        'id': id,
        'name': name
    }
    license_list.append(license)
    return license_list


def coco_videos2vid(json_data, json_name, video_id):
    scene_name = json_name.replace('.json', '_frames')
    coco_image_list = json_data['images']
    vid_video_format = {
    }

    for idx, coco_image in enumerate(coco_image_list):
        if idx == 0:
            vid_video_format["width"] = coco_image["width"]
            vid_video_format["length"] = len(coco_image_list)
            vid_video_format["date_captured"] = '2022-02-01 01:01:01:000001'
            vid_video_format["license"] = 1
            vid_video_format["flickr_url"] = coco_image["flickr_url"]
            vid_video_format["file_names"] = []
            vid_video_format["id"] = video_id
            vid_video_format["coco_url"] = ""
            vid_video_format["height"] = coco_image["height"]

        file_name = '{}/{}'.format(scene_name, coco_image["file_name"])
        vid_video_format["file_names"].append(file_name)

    vid_video_format["file_names"] = sorted(vid_video_format["file_names"])
    return vid_video_format


def get_vid_info(json_data):
    coco_image_list = json_data['images']
    coco_image = coco_image_list[0]

    width = coco_image["width"]
    height = coco_image["height"]
    length = len(coco_image_list)

    return width, height, length


def coco_categories2vid(json_data):
    coco_categories = json_data["categories"]
    return coco_categories


def coco_annotations2vid(json_data, start_instance_id, video_id, height, width, length):
    coco_annotations_list = json_data['annotations']
    new_start_instance_id, instance_anno_list = convert_annotations(coco_annotations_list, start_instance_id,
                                                                    video_id, height, width, length)

    return new_start_instance_id, instance_anno_list


def convert_annotations(annotations_list, start_instance_id, video_id, height, width, length):
    # track_dict['track_id']['image_id'] = anno
    track_dict = {}

    # sort
    for anno in annotations_list:
        track_id = anno['attributes']['track_id']
        image_id = anno['image_id']

        if track_id not in track_dict.keys():
            track_dict[track_id] = {}

        track_dict[track_id][image_id] = anno

    # convert
    instance_anno_list = []

    for t_index, track_id in enumerate(sorted(track_dict.keys())):
        image_dict = track_dict[track_id]
        instance_anno = {
            'height': height,
            'width': width,
            'length': 1,
            'video_id': video_id,
            'iscrowd': 0,
            'id': start_instance_id,
            'segmentations': [],
            'bboxes': [],
            'areas': []
        }

        start_instance_id += 1

        # set segmentation & bbox & area
        for i in range(length):
            if (i + 1) not in image_dict.keys():
                instance_anno['segmentations'].append(None)
                instance_anno['bboxes'].append(None)
                instance_anno['areas'].append(None)

            else:
                '''
                instance_anno['segmentations'].append(
                    {
                        'segmentations': [],
                        'size': []
                    }
                )
                '''
                instance_anno['segmentations'].append([])
                instance_anno['bboxes'].append([])
                instance_anno['areas'].append(0)

        for i_index, image_id in enumerate(sorted(image_dict.keys())):
            if i_index == 0:
                instance_anno['category_id'] = image_dict[image_id]['category_id']

            '''
            instance_anno['segmentations'][image_id - 1]['segmentations'] = image_dict[image_id]['segmentation']
            instance_anno['segmentations'][image_id - 1]['size'] = [height, width]
            '''
            instance_anno['segmentations'][image_id - 1] = image_dict[image_id]['segmentation']
            instance_anno['bboxes'][image_id - 1] = image_dict[image_id]['bbox']
            instance_anno['areas'][image_id - 1] = image_dict[image_id]['area']

        instance_anno_list.append(instance_anno)

    new_start_instance_id = start_instance_id + 1
    return new_start_instance_id, instance_anno_list


if __name__ == '__main__':
    coco_root = '../miseang_vis_data/coco_format'
    vis_root = '../miseang_vis_data/vis_format'
    name = 'train.json'

    coco_file_list = os.listdir(coco_root)
    video_id = 1
    start_instance_id = 1

    vid_json = {
        'info': coco_info2vid(),
        'licenses': coco_licenses2vid(),
        'videos': [],
        'categories': None,
        'annotations': []
    }

    for idx, coco_file in enumerate(coco_file_list):
        print('{}/{}'.format(idx, len(coco_file_list)))

        with open(os.path.join(coco_root, coco_file)) as coco_data:
            coco_json = json.load(coco_data)

        if idx == 0:
            vid_json['categories'] = coco_categories2vid(coco_json)

        vid_json['videos'].append(coco_videos2vid(coco_json, coco_file, video_id))
        width, height, length = get_vid_info(coco_json)
        new_start_instance_id, instance_anno_list = coco_annotations2vid(coco_json, start_instance_id,
                                                                         video_id, height, width, length)
        vid_json['annotations'] += instance_anno_list

        video_id += 1
        start_instance_id = new_start_instance_id

    outpath = os.path.join(vis_root, name)

    with open(outpath, 'w') as save_file:
        json.dump(vid_json, save_file)

