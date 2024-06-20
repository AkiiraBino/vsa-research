import torch
from ultralytics import YOLO, settings

import os
import ray

import numpy as np
import yaml

@ray.remote(num_cpus=6, num_gpus=0.5)
class Yolov8Detector:
    def __init__(
            self,
            model_path: str,
    ):
        settings.update({
            'mlflow': True,
            'clearml': False,
            'comet': False,
            'dvc': False,
            'hub': False,
            'neptune': False,
            'raytune': False,
            'tensorboard': False,
            'wandb': False,
            'runs_dir': None,
            "weights_dir": None

        })

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

    def predict(
        self,
        data_path: str = "data/dataset-3/val/images/",
    ):
        results = self.model(data_path)

        preprocess = []
        inference  = []
        postptrocess = []
        
        for result in results:
            preprocess.append(result.speed["preprocess"])
            inference.append(result.speed["inference"])
            postptrocess.append(result.speed["postprocess"])

        preprocess = np.array(preprocess)
        inference  = np.array(inference)
        postptrocess = np.array(postptrocess)

        return {
            "results": results,
            "preprocess": preprocess,
            "inference": inference,
            "postptrocess": postptrocess,
        }

    def train(
        self,
        dataset_path: str = "data/dataset-3/coco.yaml",
        train_config: str = "configs/yolov8l-train.yaml"
    ):
        with open(train_config, "r") as f:
            config = yaml.safe_load(f)

        
        self.model.train(
            dataset_path,
            **config["model_config"]
        )

    def tune(
        self,
        config_path: str = "data/dataset-3/coco.yaml",
        model_config: dict = {
            "batch": 4,
            "epochs": 100,
            "iterations": 30,
            "imgsz": 640,
            "workers": 5,
            "seed": 42,
            "optimizer": "AdamW",
        }
    ):
        self.model.tune(
            data=config_path,
            device=self.device,
            **model_config
        )