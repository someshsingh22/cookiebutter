import io
import json
import logging

import boto3
import tqdm
import zstandard as zstd

s3 = boto3.client("s3")
logging.basicConfig(filemode="a", filename="filter.log", level=logging.INFO)


def get_all_keys(bucket, prefix):
    """Get a list of all keys in an S3 bucket."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in page_iterator:
        if "Contents" in page:
            objects = page["Contents"]
            for obj in objects:
                if obj["Key"].endswith(".jsonl.zst"):
                    keys.append(obj["Key"])
    if len(keys) == 0:
        logging.log(f"No files found in {bucket}/{prefix}")
    return keys


def filter_and_dump_jsonl(input_bucket, key, substring):
    try:
        obj_data = s3.get_object(Bucket=input_bucket, Key=key)["Body"]
        decompressor = zstd.ZstdDecompressor()
        stream_reader = decompressor.stream_reader(obj_data)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        filtered_rows = []
        for line in text_stream:
            if substring in line:
                filtered_rows.append(line)
        return filtered_rows
    except Exception as e:
        logging.error(f"An error occurred: {e} in {input_bucket}/{key}")


def filter_all_jsonl(
    input_bucket, folder_path, output_bucket, substring, suffix="_filtered_chess"
):
    keys = get_all_keys(input_bucket, folder_path)
    for key in tqdm.tqdm(keys):
        new_key = key.replace(folder_path, folder_path + suffix).replace(".zst", "")
        if s3.list_objects_v2(Bucket=output_bucket, Prefix=new_key)["KeyCount"] > 0:
            logging.debug(
                f"Skipping {input_bucket}/{key} because {output_bucket}/{new_key} already exists"
            )
            continue
        filtered_rows = filter_and_dump_jsonl(input_bucket, key, substring)
        if len(filtered_rows) == 0:
            logging.warning(
                f"No data containing '{substring}' found in {input_bucket}/{key}"
            )
            continue
        s3.put_object(Bucket=output_bucket, Key=new_key, Body="".join(filtered_rows))
        logging.info(
            f"Filtered data containing '{substring}' saved to s3://{output_bucket}/{new_key}"
        )


# if __name__ == '__main__':
#     input_bucket = output_bucket = 'crawldatafromgcp'
#     folder_path = "somesh/reddit_hf/submissions"
#     substring = '"subreddit":"chess"'
#     filter_all_jsonl(input_bucket, folder_path, output_bucket, substring)
