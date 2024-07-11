import cv2
import zmq
import numpy as np

from time import time

import itertools 
from typing import Iterable
from multiprocessing import Pool


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



        print("create capture...")
        capture = cv2.VideoCapture(address)
        if capture.isOpened():
            _, curr_frame = capture.read()
            curr_frame = cv2.resize(curr_frame, imgsz)


        print("begin stream")
        while(capture.isOpened()):
            status, curr_frame = capture.read()
            if not status or (cv2.waitKey(50) & 0xFF == ord('q')):
                break
            curr_frame = cv2.resize(curr_frame, imgsz)
            

            if enable_display:
                cv2.imshow('frame', curr_frame)
            
        capture.release()
        cv2.destroyAllWindows()

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

