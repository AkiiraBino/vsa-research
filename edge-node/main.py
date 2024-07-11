import argparse

from settings.config import config


from src.creating_video_stream import VideoStream


parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_cam",
    type=int,
    default=2,
    help="Количество источников видеоряда"
)
parser.add_argument(
    "--enable_display",
    type=bool,
    default=False,
    help="Включить/выключить отображение читаемых потоков"
)
parser.add_argument(
    "--enable_filter_keyframes",
    type=bool,
    default=True,
    help="Включить/выключить фильтрацию кадров"
)
parser.add_argument(
    "--imgsz",
    type=int,
    default=640,
    help="Image size for display"
)
parser.add_argument(
    "--enable_zmq",
    type=bool,
    default=False,
    help="Включить/выключить передачу по zmq"
)

args = parser.parse_args()


if __name__ == "__main__":
    camera_addresses = [config[name] for name in list(config.keys())[:args.num_cam]]

    print(f"start {args.num_cam} camers")
    print(f"camera addresses: {camera_addresses}")

    stream = VideoStream(
        addresses=camera_addresses,
        enable_display=args.enable_display,
        imgsz=(args.imgsz,args.imgsz),
        enable_zmq=args.enable_zmq
    )

    stream.run()

