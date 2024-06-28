from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import os

import pandas as pd
from PIL import Image
from tqdm import trange
import torch

gpu_id = int(os.getenv("DEVICE", 0))
device = f"cuda:{gpu_id}"
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) .to(device)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
num_gpus = torch.cuda.device_count()
batch_size = 22
df = pd.read_json('stock_annots.jsonl', lines=True)
df['path']='EmotionNet_dataset/images-256/' + df['path']
df['prompt'] = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"

# query = """For the given image, write a one line caption and maximum 20 descriptive keywords, no more than that.
# For example:
# {"caption": "This is a sample caption", "keywords": "keyword_1, keyword_2, keyword_3"}
# Answer in JSON format only. Do not include any other information in your answer."""

# prompt = f"USER: {query}\n<image>\nASSISTANT:"

# images = glob.glob('/mnt/localssd/*/*.jpg')
# df = pd.DataFrame(images, columns=['path'])
# df['id'] = df['path'].apply(lambda x: x.split('/')[-1].split('.')[0].split('-')[0])
# df['scene'] = df['path'].apply(lambda x: x.split('/')[-1].split('.')[0].split('-')[1])

# videos = pd.read_json('asr.jsonl', lines=True)
# videos = videos[videos['filter']]

# df = df[df['id'].isin(videos['id'])]
# df['prompt'] = prompt

df = pd.read_json("llava.jsonl", lines=True)[gpu_id::num_gpus]
batches = []
for i in trange(0, len(df), batch_size):
    try:
        batch = df.iloc[i : i + batch_size]
        batch_prompts = batch["prompt"].tolist()
        images_batch = batch["path"].apply(Image.open).tolist()
        inputs = processor(
            batch_prompts, images=images_batch, return_tensors="pt", padding=True
        ).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=90,
            use_cache=True,
            do_sample=True,
            temperature=0.4,
            top_p=0.8,
            repetition_penalty=1.0,
        )
        batch["output"] = processor.batch_decode(output, skip_special_tokens=True)
    except:
        batch["output"] = None
    batches.append(batch)

pd.concat(batches).to_json(f"output-{gpu_id}.jsonl", orient="records", lines=True)
