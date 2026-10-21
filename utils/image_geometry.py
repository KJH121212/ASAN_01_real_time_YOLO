"""Aspect-preserving preview geometry; measurement coordinates remain untouched."""

import cv2
import numpy as np


def resize_within(frame, width, height):
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    return cv2.resize(frame, size)


def letterbox_preview(frame, keypoints, width=640, height=480):
    resized = resize_within(frame, width, height)
    resized_height, resized_width = resized.shape[:2]
    left, top = (width - resized_width) // 2, (height - resized_height) // 2
    canvas = np.zeros((height, width, 3), dtype=frame.dtype)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    points = [{**point, 'x': (point['x'] * resized_width + left) / width,
               'y': (point['y'] * resized_height + top) / height} for point in keypoints]
    return canvas, points
