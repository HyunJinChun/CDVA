import os
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    src_root = '../../miseang_vis_dataset/train/frame'
    dst_root = '../../miseang_vis_dataset/train/5frames'
    sample_frame = 5

    src_scene_list = os.listdir(src_root)

    for src_scene in src_scene_list:
        src_scene_path = os.path.join(src_root, src_scene)
        src_img_list = sorted(os.listdir(src_scene_path))

        new_shot_list = []

        for i in range(0, len(src_img_list), sample_frame):
            shot = []

            if (i + sample_frame) <= len(src_img_list):
                for j in range(i, i + sample_frame):
                    src_img = src_img_list[j]
                    shot.append(src_img)

                new_shot_list.append(shot)
            else:
                for j in range(i, len(src_img_list)):
                    src_img = src_img_list[j]
                    shot.append(src_img)

                new_shot_list.append(shot)



        for i, new_shot in enumerate(new_shot_list):
            print(i)
            new_shot_path = '{}/{}_{}'.format(dst_root, src_scene, str(i + 1).zfill(4))
            os.makedirs(new_shot_path, exist_ok=True)

            for src_img in new_shot:
                src_img_path = os.path.join(src_scene_path, src_img)
                dst_img_path = os.path.join(new_shot_path, src_img)

                shutil.copyfile(src_img_path, dst_img_path)




