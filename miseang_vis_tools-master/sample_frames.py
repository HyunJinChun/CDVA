import os
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    src_root = '../../itrc_dataset/frame/'
    dst_root = '../../itrc_dataset/sampled_frame/'
    folder_num = '0001'
    sample_interval = 5
    scene_list = sorted(os.listdir(os.path.join(src_root, folder_num)))

    for scene in tqdm(scene_list):
        frame_list = sorted(os.listdir(os.path.join(src_root, folder_num, scene)))

        os.makedirs(os.path.join(dst_root, folder_num, scene), exist_ok=True)

        for i in range(0, len(frame_list), sample_interval):
            src_frame = os.path.join(src_root, folder_num, scene, frame_list[i])
            dst_frame = os.path.join(dst_root, folder_num, scene, frame_list[i])

            shutil.copyfile(src_frame, dst_frame)
