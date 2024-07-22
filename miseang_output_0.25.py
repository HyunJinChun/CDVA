import json

if __name__ == '__main__':
    threshold = 0.25
    # model_result_path = '../vis_test/hyeon/face_5frame/checkpoint0010/results.json'
    # out_path = '../vis_test/hyeon/face_5frame/checkpoint0010/results_{}.json'.format(str(threshold))

    model_result_path = '../vis_test/hyeon/copy_paste/manual_method_220614/checkpoint0010/results.json'
    out_path = '../vis_test/hyeon/copy_paste/manual_method_220614/checkpoint0010/results_{}.json'.format(str(threshold))

    # model_result_path = '../vis_test/hyeon/copy_paste/mask_tracklet_220503/checkpoint0010/results.json'
    # out_path = '../vis_test/hyeon/copy_paste/mask_tracklet_220503/checkpoint0010/results_{}.json'.format(str(threshold))

    # model_result_path = '../vis_test/hyeon/copy_paste/simple_copy_paste/checkpoint0010/results.json'
    # out_path = '../vis_test/hyeon/copy_paste/simple_copy_paste/checkpoint0010/results_{}.json'.format(str(threshold))

    with open(model_result_path, 'r') as model_result:
        model_result_file = json.load(model_result)

    new_json_results = []

    for res_anno in model_result_file:
        # if res_anno['score'] < threshold:
        #     continue
        print('video_id:', res_anno['video_id'], 'score', res_anno['score'])
        # print(res_anno)
        new_json_results.append(res_anno)

    print('total :', len(new_json_results))
    print()
    print('write visualize json > ', threshold)

    # with open(out_path, 'w') as out_path_file:
    #     json.dump(new_json_results, out_path_file)
