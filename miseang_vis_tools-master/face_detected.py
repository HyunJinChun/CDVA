import json
import os


def txt_read(file_path):
    face_detected_dict = {}
    f = open(file_path, mode='r')

    for line in f:
        class_name = line.split(':')[0]
        frame_range = line.split(':')[1]
        frame_start = int(frame_range.split('~')[0])
        frame_end = int(frame_range.split('~')[1])

        if class_name not in face_detected_dict:
            face_detected_dict[class_name] = []

        face_detected_dict[class_name].append((frame_start, frame_end))

    f.close()
    return face_detected_dict


def check_attribute(face_detected_dict, json_data, frame_name, class_name_dict):
    videos_info_list = json_data["videos"]
    video_id_face_dict = {}
    video_id = None

    for videos_info in videos_info_list:
        if frame_name in videos_info["file_names"][0]:
            video_id = videos_info["id"]

            for class_name in face_detected_dict.keys():
                if class_name not in video_id_face_dict:
                    video_id_face_dict[class_name_dict[class_name]] = [0, ] * len(videos_info["file_names"])

                for idx, file_name in enumerate(videos_info["file_names"]):
                    file_num = int(file_name.split('/')[0].split('.')[0])

                    for range_info in face_detected_dict[class_name]:
                        if range_info[0] <= file_num <= range_info[1]:
                            video_id_face_dict[class_name_dict[class_name]][idx] = 1

            break

    return video_id_face_dict, video_id


def add_attribute(video_id_face_dict, json_data, video_id):
    for anno in json_data["annotations"]:
        if anno["video_id"] == video_id:
            for class_id in video_id_face_dict.keys():
                if class_id == anno["category_id"]:
                    anno["detected_face"] = []

                for idx, seg in enumerate(anno["segmentations"]):
                    if seg is None:
                        anno["detected_face"].append(None)

                    else:
                        if video_id_face_dict[class_id][idx] == 0:
                            video_id_face_dict[class_id].append(False)

                        else:
                            video_id_face_dict[class_id].append(True)

    return json_data


def empty_attributed(json_data):
    annotations = json_data["annotations"]

    for anno in annotations:
        segmentations = anno["segmentations"]
        face_detected = []

        for seg in segmentations:
            if seg is None:
                face_detected.append(None)

            else:
                face_detected.append(False)

        anno["face_detected"] = face_detected

    return annotations


if __name__ == '__main__':
    '''
    1. txt ex (얼굴을 식별 가능한 경우)
    janggeurae:451~476
    ohsangsik:451~476 
    ...
    '''
    class_name_dict = {
        "janggeurae": 1,
        "ohsangsik": 2,
        "kimdongsik": 3,
        "jangbaekki": 4,
        "anyoungyi": 5,
        "hanseokyul": 6,
        "someone": 7
    }

    json_path = '../miseang_vis_dataset/annotations/0070/no_face_detected/instances_train_sub.json'
    face_detected_root = '../miseang_vis_dataset/face_detected'
    face_detected_file_list = os.listdir(face_detected_root)

    with open(json_path, 'r') as json_name:
        json_data = json.load(json_name)

        for face_detected_file in face_detected_file_list:
            face_detected_dict = txt_read(os.path.join(face_detected_root, face_detected_file))
            frame_name = face_detected_file.split('.')[0]

            video_id_face_dict, video_id = check_attribute(face_detected_dict, json_data, frame_name, class_name_dict)
            json_data = add_attribute(video_id_face_dict, json_data, video_id)

    




