import json
import matplotlib.pyplot as plt
import numpy as np


def count_total_frames(json_data):
    videos_info = json_data["videos"]
    frame_cnt = 0

    for vid_info in videos_info:
        frame_cnt += vid_info["length"]

    return frame_cnt


def count_class(json_data):
    class_dict = get_class_dict(json_data)
    annotation_data = json_data["annotations"]
    count_class_dict = {}
    instance_class_dict = {}

    for anno in annotation_data:
        category_id = anno["category_id"]
        category_class = class_dict[category_id]

        if category_class not in instance_class_dict.keys():
            instance_class_dict[category_class] = 0

        instance_class_dict[category_class] += 1

        if category_class not in count_class_dict.keys():
            count_class_dict[category_class] = 0

        for seg in anno["segmentations"]:
            if seg is not None:
                count_class_dict[category_class] += 1

    return count_class_dict, instance_class_dict


# video id마다 사람 count
def get_average_person_exist(json_data):
    annotation_data = json_data["annotations"]
    person_count_dict = {}

    # count person
    for anno in annotation_data:
        video_id = anno["video_id"]

        if video_id not in person_count_dict.keys():
            person_count_dict[video_id] = {}

        for i, seg in enumerate(anno["segmentations"]):
            if i not in person_count_dict[video_id].keys():
                person_count_dict[video_id][i] = 0

            if seg is not None:
                person_count_dict[video_id][i] += 1

    # average person count
    total_frame = 0
    total_person = 0
    max_person = 0
    count_dist_dict = {}

    for video_id in person_count_dict.keys():
        frame_dict = person_count_dict[video_id]

        for idx in frame_dict.keys():
            # check person
            exist_person = frame_dict[idx]

            if exist_person not in count_dist_dict.keys():
                count_dist_dict[exist_person] = 0

            count_dist_dict[exist_person] += 1

            # check max_person
            if exist_person > max_person:
                max_person = exist_person

            # check average
            total_frame += 1
            total_person += exist_person

    # cal average
    average_person = total_person / total_frame

    return max_person, average_person, count_dist_dict


def get_class_dict(json_data):
    categories = json_data["categories"]
    class_dict = {}

    for cat in categories:
        class_dict[cat["id"]] = cat["name"]

    return class_dict


def analysis_data(frame_cnt, count_class_dict, instance_class_dict, max_person, average_person, count_dist_dict, mode='train'):
    print('----------------------------------------')
    print("{} dataset info".format(mode))
    print("전체 frame 개수 : {}".format(frame_cnt))
    print("클래스의 등장 회수 : ", end='')
    print(instance_class_dict)
    print("클래스의 frame 개수 : ", end='')
    print(count_class_dict)
    print("Frame에서 최대 annotation된 클래스 개수 : {}".format(max_person))
    print("Frame에서 평균 annotation된 클래스 개수 : {}".format(average_person))
    print("클래스 개수마다 frame 개수 : ", end='')
    print(count_dist_dict)
    print("그래프 생성 중")
    generate_graph(instance_class_dict, count_class_dict, count_dist_dict, mode, path='./graph_result')
    print()


def generate_graph(instance_class_dict, count_class_dict, count_dist_dict, mode='train', path='./graph_result'):
    # instance_class_dict
    x = np.arange(len(instance_class_dict.keys()))
    x_data = []
    y_data = []

    for ins_class in instance_class_dict.keys():
        x_data.append(ins_class)
        y_data.append(instance_class_dict[ins_class])

    bar = plt.bar(x, y_data)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % height, ha='center', va='bottom', size=12)
    plt.xticks(x, x_data)
    plt.savefig('{}/{}_instance_class.png'.format(path, mode))
    plt.show()

    # count_class_dict
    x = np.arange(len(count_class_dict.keys()))
    x_data = []
    y_data = []

    for count_class in count_class_dict.keys():
        x_data.append(count_class)
        y_data.append(count_class_dict[count_class])

    bar = plt.bar(x, y_data)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % height, ha='center', va='bottom', size=12)
    plt.xticks(x, x_data)
    plt.savefig('{}/{}_count_class.png'.format(path, mode))
    plt.show()

    # count_dist_dict
    x = np.arange(len(count_dist_dict.keys()))
    x_data = []
    y_data = []

    if mode == 'validation':
        for i in range(len(count_dist_dict.keys()) + 1):
            if i == 0:
                continue

            x_data.append(i)
            y_data.append(count_dist_dict[i])

    else:
        for i in range(len(count_dist_dict.keys())):
            x_data.append(i)
            y_data.append(count_dist_dict[i])

    bar = plt.bar(x, y_data)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % height, ha='center', va='bottom', size=12)
    plt.xticks(x, x_data)
    plt.savefig('{}/{}_count_dist.png'.format(path, mode))
    plt.show()


if __name__ == "__main__":
    # train_json_path = '../miseang_vis_data/vis_format/train/json/train.json'
    train_json_path = '../miseang_vis_data/vis_format_down_sample/train.json'
    validation_json_path = '../miseang_vis_data/vis_format/validation/json/validation.json'

    # train json
    with open(train_json_path, 'r') as json_name:
        json_data = json.load(json_name)
        frame_cnt = count_total_frames(json_data)
        count_class_dict, instance_class_dict = count_class(json_data)
        max_person, average_person, count_dist_dict = get_average_person_exist(json_data)

        analysis_data(frame_cnt, count_class_dict, instance_class_dict, max_person, average_person, count_dist_dict, mode='train')

    # validation json
    with open(validation_json_path) as json_name:
        json_data = json.load(json_name)
        frame_cnt = count_total_frames(json_data)
        count_class_dict, instance_class_dict = count_class(json_data)
        max_person, average_person, count_dist_dict = get_average_person_exist(json_data)

        analysis_data(frame_cnt, count_class_dict, instance_class_dict, max_person, average_person, count_dist_dict,
                      mode='validation')

