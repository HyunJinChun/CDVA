'''
score 값이 내림차순인 results.json을 오름차순으로 변경
'''

import json

if __name__ == '__main__':
    model_result_path = '../vis_test/hyeon/copy_paste/simple_copy_paste/checkpoint0010/results.json'
    out_path = '../vis_test/hyeon/copy_paste/simple_copy_paste/checkpoint0010/results_asc.json'

    with open(model_result_path, 'r') as model_result:
        model_result_file = json.load(model_result)

    new_json_file = []

    for res_anno in model_result_file:
        video_id = res_anno['video_id']
        video_name = res_anno['video_name']
        score = res_anno['score']
        category_id = res_anno['category_id']
        segmentations = res_anno['segmentations']

        new_anno = {
            'video_id': video_id,
            'video_name': video_name,
            'score': score,
            'category_id': category_id,
            'segmentations': segmentations
        }
        # new_json_file['annotations'].append(new_anno)
        new_json_file.append(new_anno)

    new_json_file.sort(key=lambda k: k['score'])
    new_json_file.sort(key=lambda k: k['video_id'])

    print('write visualize json')

    with open(out_path, 'w') as out_path_file:
        json.dump(new_json_file, out_path_file)
