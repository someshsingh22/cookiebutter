import yaml
from datasets import load_dataset
from transformers import pipeline
import pathlib
from typing import Dict, List
import trlx
from trlx.data.default_configs import TRLConfig, default_ilql_config
import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ilql import ILQLConfig

default_config = TRLConfig(
    train=TrainConfig(
            seq_length=512,
            epochs=10,
            total_steps=40,
            batch_size=4,
            checkpoint_interval=2,
            eval_interval=1,
            pipeline="PromptPipeline",
            trainer="AccelerateILQLTrainer",
            save_best=False,
        ),
    model=ModelConfig(model_path="lmsys/vicuna-7b-v1.5",num_layers_unfrozen=1,),
    tokenizer=TokenizerConfig(tokenizer_path="lmsys/vicuna-7b-v1.5",truncation_side="right", padding_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)),
    method=ILQLConfig(
        name="ILQLConfig",
        tau=0.7,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.001,
        beta=0,
        steps_for_target_q_sync=5,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=100, beta=4, temperature=0.7),
    ),
)
config = TRLConfig.update(default_config, {})
print(config)

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

imdb = load_dataset("imdb", split="train+test")

trainer = trlx.train(
    samples=imdb["text"], 
    rewards=imdb["label"],
    eval_prompts=[
        "I don't know much about Hungarian underground",
        "What made this movie so distinctly",
        "Like the sandwich I just bought at the grocery store,",
        "I cannot believe how much this movie made me want to"
    ],
    config=config,
)