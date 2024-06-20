from src.yolov8detector import Yolov8Detector
from src.preprocessing import preprocessing
from src.featuriser import run_reduced

if __name__ == "__main__":
    # ray_address = "ray://10.10.33.74:10001"
    mlflow_address = "http://10.10.33.74:8080/"

    # ray.init(address=ray_address)

    model_remote = Yolov8Detector("yolov8l.pt")

    results = model_remote.train(config_path="data/dataset-5/coco.yaml")
    # results = ray.get(results)

# if __name__ == "__main__":
#     preprocessing()

# if __name__ == "__main__":
#     run_reduced((320,320), 60)