import sys

# aws s3 sync s3://crawldatafromgcp/somesh/color_service color_service
sys.path.append("color_service")
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from ccai_logical_services.color_extractor_logical_service import (
    ColorExtractorLogicalService,
)
from PIL import Image
from tqdm import tqdm


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
        default=64,
        help="Total number of processes (default: 64)",
    )
    return parser.parse_args()


args = parse_arguments()

if __name__ == "__main__":
    IMAGE_ROOT = "reddit_scrape"
    df = pd.read_json("llava/rpics_verb.jsonl", lines=True)[["post_id", "path"]][
        args.process_idx :: args.num_process
    ]
    df["path"] = df["path"].apply(
        lambda x: x.replace("/mnt/localssd/", "/home/someshs/")
    )
    image_paths = df["path"].tolist()
    old_json_file = f"rpics_color_results.json"
    new_json_file = f"colors/rpics_color_results_{args.process_idx}.jsonl"
    if os.path.exists(old_json_file):
        done = set(json.load(open(old_json_file, "r")).keys())
    else:
        done = {}
    image_paths = [image_path for image_path in image_paths if image_path not in done]
    color_extractor = ColorExtractorLogicalService()
    color_results = {}
    batch_size = 100
    counter = 0
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
            colors = color_extractor(
                [image], {"retrieve_tone": True, "resize_image": True}
            )
            color_results[image_path] = colors[0]
        except:
            color_results[image_path] = {"error": "Image could not be loaded"}
        # print(image_path)
        # image = np.array(Image.open(image_path).convert('RGB'))
        # colors = color_extractor([image], {"retrieve_tone":True, "resize_image":True})
        # color_results[image_path] = colors[0]
        if (i + 1) % batch_size == 0 or i + 1 == len(image_paths):
            counter += 1
            with open(new_json_file, "w") as outfile:
                outfile.write(json.dumps(color_results) + "\n")
