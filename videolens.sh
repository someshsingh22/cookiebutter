#!/bin/bash

# Directory containing your files
directory="YOUR_DIRECTORY_HERE"

# Create a JSONL file and loop through the files in the directory
echo "[" > video_duration.jsonl  # Start the JSON array

for file in "$directory"/*.{mp4,octet-stream}; do
    if [ -f "$file" ]; then  # Check if it's a file
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null)
        if [ -n "$duration" ]; then  # Check if duration was retrieved
            filename=$(basename "$file")
            echo "{\"video\":\"$filename\",\"length\":$duration}," >> video_duration.jsonl
        fi
    fi
done

# Remove the last comma and close the JSON array
sed -i '$s/,$//' video_duration.jsonl
echo "]" >> video_duration.jsonl