from vllm import LLM
import pandas as pd
import re
import numpy as np
import os
from tqdm import trange

def clean_tweet(tweet):
    masked_tweet = re.sub(r'https?://\S+|www\.\S+', '<HYPERLINK>', tweet)
    masked_tweet = re.sub(r'\s+', ' ', masked_tweet)
    return masked_tweet

folder = 'emb/e5-mistral-7b-instruct'
if not os.path.exists(folder):
    os.makedirs(folder)

df = pd.read_parquet('all_tweets_transsuasion.parquet')
model = LLM(model="intfloat/e5-mistral-7b-instruct", enforce_eager=True, tensor_parallel_size=8)

task_description = "Retrieve tweets that are marketing or talking about the same event, objective, idea, intent, product, or cause"

df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)
df['len'] = df['cleaned_tweet'].apply(len)
df.sort_values('len', inplace=True)

prompts = [f'Instruct: {task_description}\nQuery: {query}' for query in df['cleaned_tweet'].tolist()]
batch_size = int(1e5)

for i in trange(len(prompts)//batch_size + 1):
    _prompts = prompts[i*batch_size:(i+1)*batch_size]
    outputs = model.encode(_prompts)
    outputs = np.array([o.outputs.embedding for o in outputs])
    np.save(f'{folder}/embeddings_{i}.npy', outputs)
    os.system(f'aws s3 sync {folder} s3://crawldatafromgcp/somesh/transsuasion/')