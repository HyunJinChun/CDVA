import torch
import argparse
import io ,os, json, cv2, tqdm, copy, sys
import numpy as np
import contextlib
import pycocotools.mask as mask_util

from PIL import Image
from collections import defaultdict
##from detectron2.detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from visualize.visualizer import Visualizer
#from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS
from pycocotools.ytvos import YTVOS

## for mmpose
import warnings

from mmpose.apis import (init_pose_model, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def create_instances(predictions, image_size):
    ret = Instances(image_size)
    score = np.asarray([x["score"] for x in predictions])
    chosen = (score >= args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bboxes"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray((predictions[i]["category_id"]) for i in chosen)

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentations"] for i in chosen]
    except KeyError:
        pass
    return ret


def load_miseang_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):

    #timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    #if timer.seconds() > 1:
    #    logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
       # meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        #meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        #meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                # _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                '''
                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue
                '''
                if not (_segm and _segm[frame_idx]):
                    continue

                # bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                # obj["bbox"] = bbox
                # obj["bbox_mode"] = BoxMode.XYWH_ABS
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])

                    else:
                        segm = segm

                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions."
    )
    # parser.add_argument("--input", help="JSON file", default='../visualization/results_vis_0.25.json')
    # parser.add_argument("--output", help="output directory", default='../visualization/output_visualize')
    # parser.add_argument("--frames", help="frames directory", default='../visualization/output')
    parser.add_argument("--input", help="JSON file", default='../vis_test/220418/results_vis_0.25.json')
    parser.add_argument("--output", help="output directory", default='../vis_test/220418/output_visualize')
    parser.add_argument("--frames", help="frames directory", default='../vis_test/220418/output')
    parser.add_argument("--conf-threshold", default=0.15, type=float, help="confidence threshold")
    ## for mmpose
    parser.add_argument(
        '--det_config',
        help='Config file for detection',
        default='mmcv/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py')
    parser.add_argument(
        '--det_checkpoint',
        help='Checkpoint file for detection',
        default='mmcv/mmpose/checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
    parser.add_argument(
        '--pose_config',
        help='Config file for pose',
        default='mmcv/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py')
    parser.add_argument(
        '--pose_checkpoint',
        help='Checkpoint file for pose',
        default='mmcv/mmpose/checkpoint/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')
    # parser.add_argument('--pose-json', type=str, default='../visualization/pose_estimation.json', help='Pose Json root')
    parser.add_argument('--pose-json', type=str, default='../vis_test/220418/pose_estimation.json', help='Pose Json root')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    logger = setup_logger()

    #with PathManager.open(args.input, "r") as f:
    #    predictions = json.load(f)

    #load seg json
    dicts = load_miseang_json(args.input, args.frames)

    # dicts = list(args.frames_image)
    # img = args.frames_image
    ##img_name = os.path.basename(args.frames_image)

    dirname = args.output
    os.makedirs(dirname, exist_ok=True)

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    def extract_frame_dic(dic, frame_idx):
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx] ##frame_idx = 60?

        return frame_dic

    print('Start visualization')

    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2] ##-2 -> -1
        os.makedirs(os.path.join(dirname), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            #img = np.array(Image.open(os.path.join(dirname, file_name)))
            visualizer = Visualizer(img)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx)) ##add text label : instance_ids
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath) # segmented img save

    ## for pose estimation

    with open(args.pose_json, 'r') as result:
        pose_file = json.load(result)

    for pose in pose_file["keypoints"]:
        image = os.path.join(dirname, pose["video_name"], pose["image_name"])
        pose_results = pose["pose_results"]

        # show the results
        vis_pose_result(
            pose_model,
            image,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=image)
        
