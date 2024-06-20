import os
import sys
import glob
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from settings.const import LABELS_DIR, IMAGES_DIR


def reduced_image_quality(
    img: np.array,
    shape: tuple,
    quality: int
) -> np.array:
    """
    Делаем resize изображения, сжимаем webp, далее проводим обратные операции.
    Получаем изображение в худшем качестве, которые теоретически будут приходить с 
    филиалов, можно использовать для аугментации датасета.

    Args
    img: изображение для сжатия
    shape: размерность до которой делаем resize
    quality: сколько сохранить качества от изначального изображения 
    """
    img_resize = cv2.resize(img, shape)

    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    result, img_encode = cv2.imencode(".webp", img_resize, encode_param)

    if result:
        img_result = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
        return (result, cv2.resize(img_result, (640, 640)))
    
    return (result, None)


def run_reduced(
    shape: tuple[int],
    quality: int,
    labels_path: str = LABELS_DIR,
    images_path: str = IMAGES_DIR,
):
    """
    Делаем reduced_image_quality для всех изображений в датасете.
    Получаем изображение в худшем качестве, которые теоретически будут приходить с 
    филиалов, можно использовать для аугментации всего датасета за раз. Также дублирует для
    новых изображений labels с координатами разметки.

    Args
    labels_path: путь до папки с разметкой
    images_path: путь до папки с изображениями
    shape: размерность до которой делаем resize
    quality: сколько сохранить качества от изначального изображения 
    """
    labels_path = list(map(lambda x: os.path.join(LABELS_DIR, x), os.listdir(LABELS_DIR)))
    images_path = list(map(lambda x: os.path.join(IMAGES_DIR, x), os.listdir(IMAGES_DIR)))

    for lp, ip in tqdm(zip(labels_path, images_path)):
        img = cv2.imread(ip)

        status, img_reduced = reduced_image_quality(
            img,
            shape,
            quality
        )
        
        if status:
            cv2.imwrite(ip.replace(".png", f"_{shape[0]}_{quality}.png"), img_reduced)
            shutil.copy(lp, lp.replace(".txt", f"_{shape[0]}_{quality}.txt"))
        else: break


def delete_reduced(
    postfix: str = "_320_60"
):
    """
    При повторной агументации лучше удалить ранее аугментированные изображения,
    так как иначе они тоже повторно аугментируются, чего чаще всего следует избегать.

    Args
    postfix: постфикс названия, складывается из shape и quality, например 
    _320_60, где 320 - shape=(320,320), а 60 - quality
    """
    for lp, ip in tqdm(zip(
            glob.glob(f"*{postfix}*", root_dir=LABELS_DIR),
            glob.glob(f"*{postfix}*", root_dir=IMAGES_DIR))):

        if os.path.exists(lp): os.remove(lp)
        if os.path.exists(ip): os.remove(ip)
