import os
from multiprocessing import Pool
from subprocess import call


# Function to extract audio from video
def extract_audio(file):
    output_file = os.path.splitext(file)[0] + ".mp3"
    call(["ffmpeg", "-i", file, "-vn", "-acodec", "libmp3lame", output_file])


if __name__ == "__main__":
    directory = "YOUR DIRECTORY HERE"
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith((".mp4"))
    ]

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=64) as pool:  # Adjust the number of processes as needed
        pool.map(extract_audio, files)
