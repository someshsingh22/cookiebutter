import argparse
import csv
import io
import logging
import math
import os
import queue
import sys
import threading
from enum import Enum
from string import Template
from typing import Callable, Dict, Iterable, List, Optional, TextIO, Tuple, Union

import boto3
import cv2
import numpy as np
import pandas as pd
import tqdm
from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    SceneManager,
    detect,
    open_video,
    save_images,
    scene_manager,
)
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.platform import get_and_create_path, get_cv2_imwrite_params, tqdm
from scenedetect.scene_detector import SceneDetector, SparseSceneDetector
from scenedetect.stats_manager import FrameMetricRegistered, StatsManager
from scenedetect.video_stream import VideoStream


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter JSONL.ZST files in an AWS S3 bucket based on a substring."
    )
    parser.add_argument(
        "--process_idx",
        type=int,
        default=0,
        help="Index of the current process (default: 0)",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=16,
        help="Total number of processes (default: 16)",
    )
    return parser.parse_args()


args = parse_arguments()


_suffix = os.listdir('videos')[args.process_idx :: args.num_process]

if __name__ == "__main__":
    for suffix in tqdm(_suffix):
        try:
            suffix = "videos/" + suffix
            video = open_video(suffix)
            scene_manager = SceneManager()
            scene_manager.add_detector(AdaptiveDetector())
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
            video_id = suffix.split('_')[0]
            save_images(
                scene_list=scene_list,
                video=video,
                image_name_template=f"{video_id}-$SCENE_NUMBER",
                output_dir="video_sccenes",
                num_images=1,
            )
        except Exception as e:
            with open(f"error_csv_{args.process_idx}.text", "a") as f:
                f.write(suffix + "\n")
