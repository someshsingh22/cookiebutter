import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from collections import defaultdict
from enum import Enum
from itertools import zip_longest
from multiprocessing import Process
from pathlib import Path
from random import random, shuffle
from typing import Any, Literal, Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from pydantic import BaseModel, validator
from pydantic.fields import ModelField
from torch import Generator
from torch.utils.data import Dataset, Subset, random_split
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM_GPUS = 3
QA_SPECIAL_TOKENS = {
    "Question": "<|prompter|>",
    "Answer": "<|assistant|>",
    "System": "<|system|>",
    "StartPrefix": "<|prefix_begin|>",
    "EndPrefix": "<|prefix_end|>",
}


def format_system_prefix(prefix, eos_token):
    return "{}{}{}".format(
        QA_SPECIAL_TOKENS["System"],
        prefix,
        eos_token,
    )


def compute_length(s: str) -> int:
    return len(re.findall(r"\w+", s)) // 5 + 1


class Mode(str, Enum):
    sft = "sft"
    rm = "rm"
    rl = "rl"


class Role(str, Enum):
    prompter = "prompter"
    assistant = "assistant"


class Utterance(BaseModel):
    text: str
    role: Role
    lang: str | None = None
    quality: float | None = None
    humor: float | None = None
    creativity: float | None = None
    context: str | None = None

    @validator("quality", "humor", "creativity")
    def between_0_1(cls, v, field: ModelField) -> float:
        if v is not None and not (0 <= v <= 1):
            raise ValueError(
                f"Field {field.name} must be between 0 and 1. Received: {v}"
            )
        return v

    def system_tag(
        self,
        eos_token: str,
        enabled: bool = True,
        property_dropout: float = 0.0,
        add_length: bool = True,
    ) -> str:
        if not enabled:
            return ""

        properties: list[tuple[float | str]] = []
        for k, v in self.dict().items():
            if v is not None and k in ["lang", "quality", "humor", "creativity"]:
                properties.append((k, v))

        if add_length:
            properties.append(("length", compute_length(self.text)))

        shuffle(properties)

        # ensure that potentially multi-line conext field comes last
        if self.context:
            properties.append(("context", self.context))

        fragments: list[str] = []
        for k, v in properties:
            if random() < property_dropout:
                continue

            if isinstance(v, float):
                fragments.append(f"{k}: {v:0.1f}")
            elif isinstance(v, str):
                if not v.isspace():  # ignore whitespace-only values
                    fragments.append(f"{k}: {v}")
            else:
                fragments.append(f"{k}: {v}")

        if len(fragments) == 0:
            return ""

        content = "\n".join(fragments)
        return f"{QA_SPECIAL_TOKENS['System']}{content}\n{eos_token}"


class DatasetEntry(BaseModel):
    pass


class DatasetEntryLm(DatasetEntry):
    """Language modelling dataset entry"""

    text: str | None = None


class DatasetEntrySft(DatasetEntry):
    """Supervised fine-tuning conversation dataset entry"""

    conversation: list[Utterance]
    system_message: Optional[str]

    def get_formatted(
        self,
        eos_token: str,
        use_system_tag: bool = False,
        system_property_dropout: float = 0.5,
        system_add_length: bool = False,
    ) -> list[str]:
        output: list[str] = []

        for i, m in enumerate(self.conversation):
            if m.role == Role.prompter:
                if use_system_tag and i + 1 < len(self.conversation):
                    a = self.conversation[i + 1]
                    assert a.role == Role.assistant
                    system_tag = a.system_tag(
                        eos_token=eos_token,
                        property_dropout=system_property_dropout,
                        add_length=system_add_length,
                    )
                else:
                    system_tag = ""
                if i == 0 and self.system_message:
                    output.append(
                        f"{QA_SPECIAL_TOKENS['System']}{self.system_message}{eos_token}{QA_SPECIAL_TOKENS['Question']}{m.text}{eos_token}{system_tag}"
                    )
                else:
                    output.append(
                        f"{QA_SPECIAL_TOKENS['Question']}{m.text}{eos_token}{system_tag}"
                    )
            else:
                output.append(f"{QA_SPECIAL_TOKENS['Answer']}{m.text}{eos_token}")

        return output


def create_dataset_entry_qa(
    mode: Mode | Literal["sft", "rm", "rl"],
    questions: list[str],
    answers: list[str] | list[list[str]],
    context: Optional[str] = None,
    lang: Optional[str] = None,
) -> DatasetEntry:
    """Helper function to create DatasetEntry objects (DatasetEntrySft or DatasetEntryRm) for simple
    Q&A datasets."""
    if mode == Mode.sft:
        messages: list[Utterance] = []

        for q, a in zip_longest(questions, answers):
            messages.append(Utterance(text=q, role=Role.prompter, lang=lang))
            if isinstance(a, list):
                a = a[0]
            messages.append(
                Utterance(text=a, role=Role.assistant, lang=lang, context=context)
            )

        return DatasetEntrySft(conversation=messages)

    # elif mode == Mode.rl:
    else:
        raise RuntimeError(f"Unsupported mode ({mode=})")


def format_pairs(
    pairs: list[str],
    eos_token: str,
    add_initial_reply_token: bool = False,
) -> list[str]:
    assert isinstance(pairs, list)
    conversations = [
        "{}{}{}".format(
            QA_SPECIAL_TOKENS["Question" if i % 2 == 0 else "Answer"],
            pairs[i],
            eos_token,
        )
        for i in range(len(pairs))
    ]
    if add_initial_reply_token:
        conversations.append(QA_SPECIAL_TOKENS["Answer"])
    return conversations


def format_reply(text: str, eos_token: str) -> str:
    return "{}{}{}".format(QA_SPECIAL_TOKENS["Answer"], text, eos_token)


# class emotion(Dataset):
#     def __init__(self, cache_dir, mode="sft"):
#         super().__init__()
#         self.rows = []
#         self.mode = mode
#         data = pd.read_csv("comments_test.csv")
#         self.rows = [
#             create_dataset_entry_qa(
#                 mode=self.mode,
#                 questions=[
#                     row["instruction"].replace("[INST]", "").replace("[/INST]", "")
#                 ],
#                 answers=[row["combined"]],
#             )
#             for _, row in data.iterrows()
#         ]

#     def __len__(self):
#         return len(self.rows)

#     def __getitem__(self, index):
#         return self.rows[index]


class sharingan_pft(Dataset):
    def __init__(self, cache_dir, mode="sft", train=True, pft=True):
        super().__init__()
        self.rows = []
        self.mode = mode
        if pft:
            if train:
                data = pd.read_csv("./pft_train_ads.csv")
            else:
                data = pd.read_csv("./pft_test_ads.csv")
        else:
            if train:
                data = pd.read_csv("./hq_train.csv")
            else:
                data = pd.read_csv("./hq_test.csv")

        self.rows = [
            create_dataset_entry_qa(
                mode=self.mode,
                questions=[row["prompt"]],
                answers=[row["asr"]],
            )
            for _, row in data.iterrows()
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def batch_infer(
    data,
    df,
    tokenizer,
    model,
    min_len,
    max_len,
    len_penalty,
    repetition_penalty,
    top_p,
    ignore_text,
    gpu,
):
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    torch.manual_seed(42)

    batch_size = 16

    for i in trange(0, len(data), batch_size):
        batch_texts = [
            data[j].get_formatted(eos_token="</s>")[0] for j in range(i, i + batch_size)
        ]
        batch_input_ids = [
            tokenizer.encode(text, return_tensors="pt") for text in batch_texts
        ]

        max_size_dim1 = max(len(tensor[0]) for tensor in batch_input_ids)

        # Pad tensors to the same size along dimension 1
        padded_batch_input_ids = [
            F.pad(
                tensor,
                (0, max_size_dim1 - len(tensor[0])),
                value=tokenizer.pad_token_id,
            )
            for tensor in batch_input_ids
        ]

        model.eval()
        batch_output = model.generate(
            input_ids=torch.cat(padded_batch_input_ids, dim=0),
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=1,
            max_new_tokens=max_len,
            top_p=top_p,
            temperature=0.7,
        )

        # Split the batch_output into individual outputs
        individual_outputs = [
            batch_output[j][batch_input_ids[j].size()[1] :]
            for j in range(len(batch_texts))
        ]

        for j, output in enumerate(individual_outputs):
            output_text = tokenizer.decode(output, skip_special_tokens=True)

            outputs = {
                "url": df.iloc[i + j]["url"],
                "instruction": df.iloc[i + j]["instruction"],
                "pred": output_text,
                "gt": df.iloc[i + j]["combined"],
            }
            with open(f"asr_vicuna_{gpu}.jsonl", "a") as file:
                file.write(json.dumps(outputs) + "\n")


def process_images(
    data, df, min_len, max_len, len_penalty, repetition_penalty, top_p, ignore_text, gpu
):
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/harini/comments_7B")
    model = AutoModelForCausalLM.from_pretrained("/mnt/data/harini/comments_7B")
    with torch.inference_mode():
        batch_infer(
            data,
            df,
            tokenizer,
            model,
            min_len,
            max_len,
            len_penalty,
            repetition_penalty,
            top_p,
            ignore_text,
            gpu,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch image inference with hyperparameters"
    )
    parser.add_argument(
        "--min_len", type=int, default=3, help="Minimum length for generated text"
    )
    parser.add_argument(
        "--max_len", type=int, default=10, help="Maximum length for generated text"
    )
    parser.add_argument(
        "--len_penalty",
        type=float,
        default=1.0,
        help="Length penalty for text generation",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for text generation",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling value"
    )

    parser.add_argument(
        "--ignore_text", type=str, default="", help="Ignore text on the image"
    )

    args = parser.parse_args()
    data = emotion(".")

    df = pd.read_csv("/mnt/data/harini/comments_test.csv")
    multiprocessing.set_start_method("spawn")
    NUM_GPUS = 1
    processes = []

    for gpu in range(NUM_GPUS):
        p = Process(
            target=process_images,
            args=(
                data,
                df,
                args.min_len,
                args.max_len,
                args.len_penalty,
                args.repetition_penalty,
                args.top_p,
                args.ignore_text,
                gpu,
            ),
        )
        processes.append(p)

    # Start the processes
    for p in processes:
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()
