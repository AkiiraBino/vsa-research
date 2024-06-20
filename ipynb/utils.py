import os

import cv2
import pickle
import pandas as pd

from typing import Any


def init_paths(data_folder: str = "/data",) -> dict[str, Any]:
    BASE_DIR = os.getcwd()
    DATA_DIR = BASE_DIR + data_folder

    FOLDERS_WITH_DATA = []
    for item in os.listdir(DATA_DIR):
        current_path = DATA_DIR + f"/{item}"
        
        if os.path.isdir(current_path):
            FOLDERS_WITH_DATA.append(current_path)

    PATH_TO_VIDEOS = []
    PATH_TO_IMGS = []
    for folder in FOLDERS_WITH_DATA:
        for item in os.listdir(folder):
            if item[-4:len(item)] != ".png":
                PATH_TO_VIDEOS.append(f"{folder}" + f"/{item}")
            else:
                PATH_TO_IMGS.append(f"{folder}" + f"/{item}")
    
    return {
        "BASE_DIR": BASE_DIR,
        "DATA_DIR": DATA_DIR,
        "FOLDERS_WITH_DATA": FOLDERS_WITH_DATA,
        "PATH_TO_VIDEOS": PATH_TO_VIDEOS,
        "PATH_TO_IMGS": PATH_TO_IMGS,
    }
    

def init_metadata(
    video_metadata_path: str = "./data/metadata_video.csv",
    img_metadata_path: str = "./data/metadata_imgs.csv"
) -> dict[str, pd.DataFrame]:
    metadata_video = pd.read_csv(video_metadata_path, sep=";", index_col="title")
    metadata_img = pd.read_csv(img_metadata_path, sep=";", index_col="title")
    
    return {
        "metadata_video": metadata_video,
        "metadata_img": metadata_img,
    }


def load_pickle(
    path: str
) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
    

def dump_pickle(
    path: str,
    obj: Any
) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)