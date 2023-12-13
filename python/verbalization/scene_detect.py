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

s3 = boto3.client("s3")
bucket = "crawldatafromgcp"
prefix = "somesh/adsofworld"

suffix = pd.read_json("yt_ads_sharingan_nocap_f.jsonl", lines=True)["Video"].tolist()
suffix = suffix[args.process_idx :: args.num_process]


def write_scene_list(
    video_id,
    output_csv_file: TextIO,
    scene_list: Iterable[Tuple[FrameTimecode, FrameTimecode]],
    include_cut_list: bool = True,
    cut_list: Optional[Iterable[FrameTimecode]] = None,
) -> None:
    """Writes the given list of scenes to an output file handle in CSV format.

    Arguments:
        output_csv_file: Handle to open file in write mode.
        scene_list: List of pairs of FrameTimecodes denoting each scene's start/end FrameTimecode.
        include_cut_list: Bool indicating if the first row should include the timecodes where
            each scene starts. Should be set to False if RFC 4180 compliant CSV output is required.
        cut_list: Optional list of FrameTimecode objects denoting the cut list (i.e. the frames
            in the video that need to be split to generate individual scenes). If not specified,
            the cut list is generated using the start times of each scene following the first one.
    """
    csv_writer = csv.writer(output_csv_file, lineterminator="\n")
    # If required, output the cutting list as the first row (i.e. before the header row).
    if include_cut_list:
        csv_writer.writerow(
            ["Timecode List:"] + cut_list
            if cut_list
            else [start.get_timecode() for start, _ in scene_list[1:]]
        )

    for i, (start, end) in enumerate(scene_list):
        duration = end - start
        csv_writer.writerow(
            [
                "%d" % video_id,
                "%d" % (i + 1),
                "%d" % (start.get_frames() + 1),
                start.get_timecode(),
                "%.3f" % start.get_seconds(),
                "%d" % end.get_frames(),
                end.get_timecode(),
                "%.3f" % end.get_seconds(),
                "%d" % duration.get_frames(),
                duration.get_timecode(),
                "%.3f" % duration.get_seconds(),
            ]
        )


if __name__ == "__main__":
    with open(f"scene_duration_{args.process_idx}.csv", "a") as f:
        csv_writer = csv.writer(f, lineterminator="\n")

        csv_writer.writerow(
            [
                "Video id",
                "Scene Number",
                "Start Frame",
                "Start Timecode",
                "Start Time (seconds)",
                "End Frame",
                "End Timecode",
                "End Time (seconds)",
                "Length (frames)",
                "Length (timecode)",
                "Length (seconds)",
            ]
        )

    for _suffix in tqdm(suffix):
        key = prefix + "/" + _suffix
        try:
            obj_data = s3.get_object(Bucket=bucket, Key=key)
            video_id = _suffix.split("/")[1].split(".")[0]
            if os.path.exists(_suffix):
                continue
            s3.download_file(bucket, key, _suffix)
            video = open_video(_suffix)
            scene_manager = SceneManager()
            scene_manager.add_detector(AdaptiveDetector())
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
            video_id = _suffix.split("/")[1].split(".")[0]
            save_images(
                scene_list=scene_list,
                video=video,
                image_name_template=f"{video_id}-$SCENE_NUMBER",
                output_dir="yt_scenes",
                num_images=1,
            )
        except Exception as e:
            with open(f"error_csv_{args.process_idx}.text", "a") as f:
                f.write(key + "\n")
