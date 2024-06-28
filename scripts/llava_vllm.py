from PIL import Image
import pandas as pd
from tqdm import trange
from vllm import LLM, SamplingParams
from vllm.multimodal.image import ImagePixelData
import warnings
import os
import subprocess

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
df = pd.read_json("stock_annots.jsonl", lines=True)
df["path"] = "EmotionNet_dataset/images-256/" + df["path"]
if not os.path.exists("out"):
    os.makedirs("out")

if os.path.exists("out/completed.jsonl"):
    completed = pd.read_json("out/completed.jsonl", lines=True)
    df = df[~df["path"].isin(completed["path"])]

llm = LLM(
    model="llava-hf/llava-v1.6-mistral-7b-hf",
    image_input_type="pixel_values",
    image_token_id=32000,
    image_input_shape="1,3,336,336",
    image_feature_size=1176,
    disable_image_processor=False,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.97,
)
warnings.filterwarnings("ignore")
prompt = f"[INST] \nYou are an image captioner. Your task is to analyze an image and describe the image as a human would interpret them in one sentence only not more than that {'<image>' * 1176} What is shown in this image? [/INST]"
batch_size = 1000

for batch in trange(0, len(df), batch_size):
    batch_df = df.iloc[batch : batch + batch_size]
    batch_df["image"] = batch_df["path"].apply(lambda x: ImagePixelData(Image.open(x)))
    image_data = batch_df["image"].tolist()
    outputs = llm.generate(
        [
            {"prompt": prompt, "multi_modal_data": image_data_i}
            for image_data_i in image_data
        ],
        sampling_params=sampling_params,
    )
    batch_df["caption"] = [o.outputs[0].text for o in outputs]
    batch_df.to_json(f"out/completed_{batch}.jsonl", lines=True, orient="records")
    subprocess.Popen(
        [
            "aws",
            "s3",
            "cp",
            f"out/completed_{batch}.jsonl",
            "s3://crawldatafromgcp/somesh/emotion/captions/completed_{batch}.jsonl",
        ]
    )
    os.system(
        f"aws s3 cp out/completed_{batch}.jsonl s3://crawldatafromgcp/somesh/emotion/captions/completed_{batch}.jsonl"
    )
