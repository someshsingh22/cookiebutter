# !pip install lxml

import math
import time
import xml.etree.ElementTree as ElementTree
from html import unescape

import pandas as pd


def read_from_jsonl(path):
    """Read a jsonl file into a pandas dataframe."""
    df = pd.read_json(path, lines=True)
    df = df[df["Video"].notna()]
    df = df[df["caption"].notna()]
    return df


def float_to_srt_time_format(d: float) -> str:
    """Convert decimal durations into proper srt format.

    :rtype: str
    :returns:
        SubRip Subtitle (str) formatted time duration.

    float_to_srt_time_format(3.89) -> '00:00:03,890'
    """
    fraction, whole = math.modf(d)
    time_fmt = time.strftime("%H:%M:%S,", time.gmtime(whole))
    ms = f"{fraction:.3f}".replace("0.", "")
    return time_fmt + ms


def xml_caption_to_srt(xml_captions: str, text_only=True) -> str:
    """Convert xml caption tracks to "SubRip Subtitle (srt)".

    :param str xml_captions:
        XML formatted caption tracks.
    """
    segments = []
    root = ElementTree.fromstring(xml_captions)
    for i, child in enumerate(list(root.findall("body/p"))):
        text = "".join(child.itertext()).strip()
        if not text:
            continue
        caption = unescape(
            text.replace("\n", " ").replace("  ", " "),
        )
        try:
            duration = float(child.attrib["d"])
        except KeyError:
            duration = 0.0
        start = float(child.attrib["t"])
        end = start + duration
        sequence_number = i + 1  # convert from 0-indexed to 1.
        if text_only:
            segments.append(caption)
        else:
            line = "{seq}\n{start} --> {end}\n{text}\n".format(
                seq=sequence_number,
                start=float_to_srt_time_format(start),
                end=float_to_srt_time_format(end),
                text=caption,
            )
            segments.append(line)
    join = " " if text_only else "\n"
    return join.join(segments)
