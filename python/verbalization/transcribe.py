import json
import logging
import os

import pandas as pd
from tqdm import trange
from transformers import pipeline

logging.basicConfig(filename="asr.log", filemode="w")

chunk_size = 16
batch_size = 64

ASR = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    chunk_length_s=30,
    device=f"cuda:{os.getenv('DEVICE')}",
)

out_path = f"asr_{os.getenv('DEVICE')}.jsonl"
audio_db_path = "asr.jsonl"

if __name__ == "__main__":
    if os.path.exists(audio_db_path):
        transcribed = pd.read_json(audio_db_path, lines=True)
    else:
        transcribed = pd.DataFrame(columns=["text", "chunks", "audio"])
    audio_db = open(out_path, "a")
    channel_db = json.load(open("data/agadmator/agadmator_channel_db.json"))
    audios = [
        a["audio"]
        for a in sorted(list(channel_db.values()), key=lambda x: x["length"])
        if not (a["audio"] in set(transcribed["audio"]))
    ]

    for start in trange(0, len(audios), chunk_size):
        try:
            chunk = audios[start : start + chunk_size]
            asrs = ASR(chunk, batch_size=batch_size, return_timestamps=True)
            for audio, asr in zip(chunk, asrs):
                asr["audio"] = audio
                audio_db.write(json.dumps(asr) + "\n")
        except:
            logging.error(json.dumps({"STATUS": "FAILED", "paths": chunk}))
