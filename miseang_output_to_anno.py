import json

if __name__ == '__main__':
    model_result_path = '../vis_test/hyeon/220918_r101_miseang/CDVA_MAX/results_0015.json'
    gt_path = '../miseang_vis_data/vis_format/validation/json/validation.json'
    #gt_path = '../vis_test/hyeon/output_visualize/val.json'
    threshold = 0.25
    out_path = '../vis_test/hyeon/220918_r101_miseang/CDVA_MAX/results_vis_{}.json'.format(str(threshold))

    with open(gt_path, 'r') as gt_result:
        gt_result_file = json.load(gt_result)

    with open(model_result_path, 'r') as model_result:
        model_result_file = json.load(model_result)

    new_json_file = {
        "info": gt_result_file["info"],
        "licenses": gt_result_file["licenses"],
        "videos": gt_result_file["videos"],
        "categories": gt_result_file["categories"],
        "annotations": []
    }
    new_id = 1

    for res_anno in model_result_file:
        video_id = res_anno['video_id']
        score = res_anno['score']
        category_id = res_anno['category_id']
        segmentations = res_anno['segmentations']
        height = segmentations[0]['size'][0]
        width = segmentations[0]['size'][1]

        if score < threshold:
            continue

        new_anno = {
            'height': height,
            'width': width,
            'length': 1,
            'video_id': video_id,
            'iscrowd': 1,
            'id': new_id,
            'segmentations': segmentations,
            'category_id': category_id,
            'score': score
        }
        new_json_file['annotations'].append(new_anno)
        new_id += 1

    print('write visualize json')

    with open(out_path, 'w') as out_path_file:
        json.dump(new_json_file, out_path_file)
