import cv2
import zmq
import numpy as np

import sys
from time import time

import itertools 
from typing import Iterable
from multiprocessing import Pool

from src.extract_keyframes import (
    KeyFrameExtractor,
    ModeExtractor,
    ModeOutlier
)


class VideoStream:
    def __init__(
        self,
        addresses: Iterable[str],
        enable_display: bool = False,
        imgsz: tuple[int, int] = (640, 640),
        enable_zmq: bool = False
    ) -> None:
        self.addresses = addresses
        self.enable_display = enable_display
        self.imgsz = imgsz
        self.num_proc = len(addresses)
        self.enable_zmq = enable_zmq

    def run(
        self
    ) -> None:
        args = zip(
            self.addresses,
            [port for port in range(5550, 5550 + self.num_proc)],
            itertools.repeat(self.enable_display, self.num_proc),
            itertools.repeat(self.imgsz, self.num_proc)
        )

        with Pool(self.num_proc) as p:
            p.starmap(self._create_stream, args)

    def _create_zmq(
            self,
            port
    ):
        context_zmq = zmq.Context()
        dst = context_zmq.socket(zmq.PUSH)
        dst.bind(f"tcp://127.0.0.1:{port}")

        return dst

    def _create_stream(
        self,
        address: str,
        zmq_port: int,
        enable_display: bool = False,
        imgsz: tuple[int, int] = (640, 640)
    ) -> None:
        extractor_absdiff_default = KeyFrameExtractor(
            mode_extractor=ModeExtractor.ABSDIFF_DEFAULT,
            mode_outlier=ModeOutlier.HIGH
        )

        extractor_absdiff_all = KeyFrameExtractor(
            mode_extractor=ModeExtractor.ABSDIFF_OUTLIER,
            mode_outlier=ModeOutlier.ALL,
        )

        extractor_ssim_default = KeyFrameExtractor(
            mode_extractor=ModeExtractor.SSIM_DEFAULT,
            mode_outlier=ModeOutlier.LOW
        )

        extractor_ssim_all = KeyFrameExtractor(
            mode_extractor=ModeExtractor.SSIM_OTLIER,
            mode_outlier=ModeOutlier.ALL
        )

        print("start extract frames for outlier")
        frames = self._extract_frames(
            60,
            address,
            imgsz
        )

        extractor_absdiff_default.calculate_threshold(frames)
        extractor_absdiff_all.calculate_threshold(frames)
        extractor_ssim_default.calculate_threshold(frames)
        extractor_ssim_all.calculate_threshold(frames)

        count_default = 0
        count_absdiff_default = 0
        count_absdiff_all = 0
        count_ssim_default = 0
        count_ssim_all = 0

        # if self.enable_zmq:
        #     print(f"start zmq on port {zmq_port}")
        #     dst = self._create_zmq(zmq_port)

        print("create capture...")
        capture = cv2.VideoCapture(address)
        if capture.isOpened():
            _, curr_frame = capture.read()
            curr_frame = cv2.resize(curr_frame, imgsz)


        time_start = time()
        print("begin stream")
        while(capture.isOpened()):
            prev_frame = curr_frame
            status, curr_frame = capture.read()
            if not status or (cv2.waitKey(1) & 0xFF == ord('q')): break
            curr_frame = cv2.resize(curr_frame, imgsz)

            count_default+=1
            times_diff_calc = {
                "absdiff_default": [],
                "absdiff_all": [],
                "ssim_default": [],
                "ssim_all": []
            }
            
            time_start_absdiff_default = time()
            cv2.imwrite(f"src/keyframes/no_keyframes/{count_default}_{zmq_port}.png", curr_frame)
            if extractor_absdiff_default.check(curr_frame, prev_frame):
                count_absdiff_default+=1
                cv2.imwrite(f"src/keyframes/absdiff_default/{count_default}_{zmq_port}.png", curr_frame)
            time_start_absdiff_all = time()
            if extractor_absdiff_all.check(curr_frame, prev_frame):
                count_absdiff_all+=1
                cv2.imwrite(f"src/keyframes/absdiff_all/{count_default}_{zmq_port}.png", curr_frame)
            time_start_ssim_default = time()
            if extractor_ssim_default.check(curr_frame, prev_frame):
                count_ssim_default += 1
                cv2.imwrite(f"src/keyframes/ssim_default/{count_default}_{zmq_port}.png", curr_frame)
            time_start_ssim_all = time()
            if extractor_ssim_all.check(curr_frame, prev_frame):
                count_ssim_all += 1
                cv2.imwrite(f"src/keyframes/ssim_all/{count_default}_{zmq_port}.png", curr_frame)
            time_stop_diff = time()

            times_diff_calc = self._dump_time(
                times_diff_calc,
                time_start_absdiff_default,
                time_start_absdiff_all,
                time_start_ssim_default,
                time_start_ssim_all,
                time_stop_diff,
            )

            if time_start - time() >= 60:
                break
            # if extractor_absdiff_default.check(curr_frame, prev_frame): 
            #     if self.enable_zmq:
            #         dst.send_pyobj(
            #             {
            #                 "frame": curr_frame,
            #                 "time_begin_stream": ts,
            #             }
            #         )

                # if enable_display: cv2.imshow('frame', curr_frame)
            
        capture.release()
        cv2.destroyAllWindows()

        print(
            f"""
            port {zmq_port}
            count_default {count_default}
            count_absdiff_default/time {count_absdiff_default}/{round(np.mean(times_diff_calc["absdiff_default"]), 5)}
            count_absdiff_all/time {count_absdiff_all}/{round(np.mean(times_diff_calc["absdiff_all"]), 5)}
            count_ssim_default/time {count_ssim_default}/{round(np.mean(times_diff_calc["ssim_default"]), 5)}
            count_ssim/time {count_ssim_all}/{round(np.mean(times_diff_calc["ssim_all"]), 5)}
            """
        )

    def _dump_time(
        self,
        times: dict,
        ts1,ts2,ts3,ts4,ts5
    ):
        times["absdiff_default"].append(
            ts2 - ts1
        )
        times["absdiff_all"].append(
            ts3 - ts2 
        )
        times["ssim_default"].append(
            ts4 - ts3
        )
        times["ssim_all"].append(
            ts5 - ts4
        )

        return times

    def _extract_frames(
        self,
        duration: int,
        address: str,
        imgsz: tuple[int, int]
    ) -> list[np.array]:
        frames = []

        capture = cv2.VideoCapture(address)
        start = time()
        while capture.isOpened():
            _, curr_frame = capture.read()
            frames.append(cv2.resize(curr_frame, imgsz))

            if time() - start >= duration:
                break

        return frames

