import cv2
import numpy as np
from skimage.metrics import structural_similarity


from enum import Enum


class ModeExtractor(Enum):
    ABSDIFF_DEFAULT = 1
    ABSDIFF_OUTLIER = 2
    SSIM_DEFAULT = 3
    SSIM_OTLIER = 4


class ModeOutlier(Enum):
    HIGH = 1
    LOW = 2
    ALL = 3


class KeyFrameExtractor:
    def __init__(
        self,
        high_threshold: int | float | None = None,
        low_threshold: int | float | None = None,
        mode_outlier: ModeOutlier | int | None = ModeOutlier.HIGH,
        mode_extractor: ModeExtractor | int = ModeExtractor.ABSDIFF_DEFAULT,
    ):
        self.mode_extractor = mode_extractor
        self.mode_outlier = mode_outlier

        if self.mode_extractor == ModeExtractor.ABSDIFF_DEFAULT:
            self.func = self._absdiff_opencv_default
        elif self.mode_extractor == ModeExtractor.ABSDIFF_OUTLIER:
            self.func = self._absdiff_opencv_outlier
        elif self.mode_extractor == ModeExtractor.SSIM_DEFAULT:
            self.func = self._ssim_skimage_default
        elif self.mode_extractor == ModeExtractor.SSIM_OTLIER:
            self.func = self._ssim_skimage_outlier

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def check(
        self,
        current_frame: np.array,
        preview_frame: np.array,
    ) -> bool | None:
        if self.high_threshold != None or self.low_threshold != None:
            return self.func(current_frame, preview_frame)
        else:
            print("No threshold selected , please, use calculate_threshold")

    def calculate_threshold(
        self,
        frames: list[np.array]
    ):

        if (
            self.mode_extractor == ModeExtractor.SSIM_DEFAULT or
            self.mode_extractor == ModeExtractor.SSIM_OTLIER
        ):
            diffs = self._calculate_threshold_ssim(frames)
        else:
            diffs = self._calculate_threshold_absdiff(frames)

        q_75 = np.quantile(diffs, 0.75)
        q_25 = np.quantile(diffs, 0.25)
        if self.mode_outlier == ModeOutlier.HIGH:
            self.high_threshold = q_75 + 1.5 * (q_75 - q_25) 
        elif self.mode_outlier == ModeOutlier.LOW:
            self.low_threshold = q_25 - 1.5 * (q_75 - q_25)
        elif self.mode_outlier == ModeOutlier.ALL:
            self.high_threshold = q_75 + 1.5 * (q_75 - q_25) 
            self.low_threshold = q_25 - 1.5 * (q_75 - q_25)

    def _calculate_threshold_absdiff(
        self,
        frames: list[np.array]
    ) -> list[float]:
        prev_frame = frames[0]
        diffs = []

        for i in range(1, len(frames)):
            diffs.append(cv2.absdiff(frames[i], prev_frame).sum())
            prev_frame = frames[i]
        return diffs
    
    def _calculate_threshold_ssim(
        self,
        frames
    ):
        prev_frame = frames[0]
        diffs = []

        for i in range(1, len(frames)):
            diffs.append(
                structural_similarity(frames[i], prev_frame, channel_axis=2)
            )
            prev_frame = frames[i]

        return diffs
        
    def _absdiff_opencv_default(
        self,
        current_frame: np.array,
        preview_frame: np.array,
    ) -> bool:
        diff = cv2.absdiff(current_frame, preview_frame).sum()
        return diff > self.high_threshold

    def _absdiff_opencv_outlier(
        self,
        current_frame: np.array,
        preview_frame: np.array,
    ) -> bool:
        diff = cv2.absdiff(current_frame, preview_frame).sum()
        return (
            diff > self.high_threshold
            or diff < self.low_threshold
        )

    def _ssim_skimage_default(
        self,
        current_frame: np.array,
        preview_frame: np.array,
    ) -> bool:
        diff = structural_similarity(current_frame, preview_frame, channel_axis=2)
        return diff < self.low_threshold
    
    def _ssim_skimage_outlier(
        self,
        current_frame: np.array,
        preview_frame: np.array,
    ) -> bool:
        diff = structural_similarity(current_frame, preview_frame, channel_axis=2)
        return (
            diff > self.high_threshold
            or diff < self.low_threshold
        )
 