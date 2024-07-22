import torch
import argparse
import io ,os, json, cv2, tqdm, copy
import numpy as np
import contextlib
import pycocotools.mask as mask_util

from PIL import Image
from collections import defaultdict
##from detectron2.detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
#from detectron2.utils.visualizer import Visualizer
from visualize.visualizer import Visualizer
#from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS
from pycocotools.ytvos import YTVOS

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
        # and by convention they aref always ignored.
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

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
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
    parser.add_argument("--input", required=True, help="JSON file")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--frames", required=True, help="frames directory")
    parser.add_argument("--conf-threshold", default=0.0, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    #with PathManager.open(args.input, "r") as f:
    #    predictions = json.load(f)

    #load json
    dicts = load_miseang_json(args.input, args.frames)

    # dicts = list(args.frames_image)
    # img = args.frames_image
    ##img_name = os.path.basename(args.frames_image)

    dirname = args.output
    os.makedirs(dirname, exist_ok=True)


    def extract_frame_dic(dic, frame_idx):
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx] ##frame_idx = 60?

        return frame_dic


    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2] ##-2 -> -1
        print(vid_name)
        os.makedirs(os.path.join(dirname), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            #img = np.array(Image.open(os.path.join(dirname, file_name)))
            visualizer = Visualizer(img)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx)) ##add text label : instance_ids
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
    '''
    #pred_by_image = defaultdict(list)
    pred_by_image = []
    ##for p in predictions['annotations']:
    for p in predictions['annotations']:
        #pred_by_image[p["video_id"]].append(p['video_id'])
        pred_by_image[p["video_id"]].append(p['video_id'])
        ##pred_by_image.append(predictions['video_id'])
        # pred_by_image = defaultdict(list)


        dicts = list(args.frames_image)
        #img = args.frames_image
        img_name=os.path.basename(args.frames_image)
        os.makedirs(args.output, exist_ok=True)

    for im in enumerate(predictions['file_names']):
        #if im == img_name:
        print(im)
        img = cv2.imread(im, cv2.IMREAD_COLOR)[:, :, ::-1]
        #basename = os.path.basename(dicts[dic])

        # predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        predictions = create_instances(pred_by_image[im], img.shape[:2])
        vis = Visualizer(img)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img)
        vis_gt = vis.draw_dataset_dict(im).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, img_name), concat[:, :, ::-1])
    '''


