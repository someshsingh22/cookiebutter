import multiprocessing as mp

from paddleocr import PaddleOCR
from tqdm import tqdm
import json
import os


def perform_ocr(process_id, img_paths, output_file):
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    ocr_results = []
    for i, img_path in enumerate(img_paths, start=1):
        try:
            result = ocr.ocr(img_path, cls=True)
        except Exception as e:
            ocr_results.append({"img_path": img_path, "ocr_text": {"Error": str(e)}})
            continue
        if result:
            ocr_results.append({"img_path": img_path, "ocr_text": result})
        else:
            ocr_results.append({"img_path": img_path, "ocr_text": []})
        if i % 500 == 0:
            with open(f"ocr/ocr_output_{process_id}_{i}.jsonl", "w") as f:
                for res in ocr_results:
                    f.write(json.dumps(res) + "\n")
            ocr_results = []

    with open(output_file, "w") as f:
        for res in ocr_results:
            f.write(json.dumps(res) + "\n")


def divide_and_process_images(processes=180):
    img_paths = sorted(["frames_marketing/"+f for f in os.listdir("frames_marketing")])
    chunk_size = len(img_paths) // processes
    chunks = [
        img_paths[i : i + chunk_size] for i in range(0, len(img_paths), chunk_size)
    ]
    jobs = []
    for i, chunk in enumerate(chunks):
        output_file = f"ocr/ocr_output_{i}.jsonl"
        p = mp.Process(target=perform_ocr, args=(i, chunk, output_file))
        jobs.append(p)
        p.start()

    for job in tqdm(jobs, desc="Processing Images", total=len(jobs)):
        job.join()


if __name__ == "__main__":
    divide_and_process_images(processes=180)
