import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import matplotlib.pyplot as plt
from PIL import Image
from datasets.transforms_copy_paste import Denormalize

def make_square(im, x2=1280, y2=720):
    x1, y1 = im.size
    new_im = Image.new('RGB', (x2, y2), "black")
    print()
    # new_im.paste(im, (int((x2 - x1) / 2), int((y2 - y1) / 2))) # center paste
    new_im.paste(im, (0, 0))
    return new_im

if __name__ == '__main__':
    test_image = Image.open('../vis_test/hyeon/test/test_0_0.png')
    new_image = make_square(test_image)
    new_image.show()