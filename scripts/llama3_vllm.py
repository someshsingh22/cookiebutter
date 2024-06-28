from vllm import LLM, SamplingParams
import pandas as pd

llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=8)
# df = pd.read_parquet('filtered_cs_links.parquet')
# df=df[~(df['Final URL'].apply(lambda x: x.split('/')[2])=='bit.ly')]

tokenizer = llm.get_tokenizer()
SYS = """You are an AI trained to infer and describe the possible contents of a webpage based solely on the URL provided. Given a URL, you will analyze its structure, keywords, and any available context to provide a concise description of what the page is likely about. Your description should be speculative but grounded in logical assumptions based on the URL's components.

When describing the possible contents of the URL, consider:

The main subject or topic indicated by the URL.
Try to utilize information from the domain, subdomain and language codes. 
Potential key details or points that might be covered.
Likely subtopics or sections based on URL path and parameters.
The probable purpose or goal of the page (e.g., informative, promotional, instructional).
Any other relevant context or patterns recognized from similar URLs.
Make sure your description is logical and plausible, providing an educated guess about the webpage's content.
If the URL is ambiguous or unclear, like a shortened link or link with just a hash, you can just reply with "The URL is ambiguous or unclear."
"""

convert = lambda x: tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYS},
        {
            "role": "user",
            "content": f"Describe in 2-3 sentences the possible contents of {x}",
        },
    ],
    tokenize=False,
)

import json

prompts = json.load(open("prompts.json"))

outputs = llm.generate(
    prompts,
    SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        frequency_penalty=1.5,
        stop_token_ids=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],  # KEYPOINT HERE
    ),
)
