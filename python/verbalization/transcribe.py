import argparse
import json
import logging
import os

import pandas as pd
import torch
from tqdm import trange
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logging.basicConfig(filename=f"yt_asr_{os.getenv('DEVICE')}.log", filemode="w")

chunk_size = 200
batch_size = 200

device = f"cuda:{os.getenv('DEVICE')}"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    use_flash_attention_2=True,
)

ASR = pipeline(
    "automatic-speech-recognition",
    model=model,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=batch_size,
)

num_gpus = 8
out_path = f"yt_asr_{os.getenv('DEVICE')}.jsonl"
audio_db_path = f"yt_asr_{os.getenv('DEVICE')}.jsonl"

if __name__ == "__main__":
    if os.path.exists(audio_db_path):
        transcribed = pd.read_json(audio_db_path, lines=True)
    else:
        transcribed = pd.DataFrame(columns=["text", "chunks", "audio"])
    audio_db = open(out_path, "a")

    cb = pd.read_json("yt_ads_sharingan_nocap_f.jsonl", lines=True).sort_values(
        "length"
    )
    cb = cb[cb["length"] < 120]
    audios = cb["audio"].apply(lambda x: "/".join(x.split("/")[-2:])).tolist()

    failed = []

    for start in trange(
        int(os.getenv("DEVICE")) * chunk_size, len(audios), chunk_size * num_gpus
    ):
        # print(f"Start: {start} End: {start+chunk_size}")
        # chunk = audios[start : start + chunk_size]
        # asrs = ASR(chunk, batch_size=batch_size, return_timestamps=True)
        # for audio, asr in zip(chunk, asrs):
        #     asr["audio"] = audio
        #     audio_db.write(json.dumps(asr) + "\n")
        try:
            chunk = audios[start : start + chunk_size]
            asrs = ASR(chunk, batch_size=batch_size, return_timestamps=True)
            for audio, asr in zip(chunk, asrs):
                asr["audio"] = audio
                audio_db.write(json.dumps(asr) + "\n")
        except:
            logging.error(json.dumps({"STATUS": "FAILED", "paths": chunk}))
            failed.extend(chunk)
    with open("failed.jsonl", "a") as f:
        f.write(json.dumps(chunk) + "\n")
