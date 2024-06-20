import argparse
import logging

from settings.config import config


from src.creating_video_stream import VideoStream


parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_cam",
    type=int,
    default=2,
    help="Number of camers for streaming"
)
parser.add_argument(
    "--enable_display",
    type=bool,
    default=False,
    help="Enable/disable display video from streams"
)
parser.add_argument(
    "--enable_filter_keyframes",
    type=bool,
    default=True,
    help="Enable/disable filtering keyframes with absdiff"
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
    help="Enable/disable transport to zmq"
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

