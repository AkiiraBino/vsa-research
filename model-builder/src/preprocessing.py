import os
import sys
import glob
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from settings.const import LABELS_DIR, IMAGES_DIR


def preprocessing(
    format: str = ".png",
    shape: tuple = (640, 640)
):
    for ip in tqdm(os.listdir(IMAGES_DIR)):
        default_path = os.path.join(IMAGES_DIR, ip)
        img = cv2.imread(default_path)
        if img.shape[:2] != shape:
            img = cv2.resize(img, shape)
            img_format = ip.split(".")[-1]
            cv2.imwrite(os.path.join(IMAGES_DIR, ip.split(".")[0] + format), img)
            if img_format != "png":
                os.remove(default_path)