#!/bin/bash

# Bucket path and file name pattern
bucket_path="s3://crawldatafromgcp/somesh/KPITranslation/ckpt/composer/"
file_pattern="/sensei-fs/users/someshs/transuasion/v2/ep*.pt"

# Find the latest file matching the pattern
latest_file=$(ls -t $file_pattern | head -1)

# Check if the latest file exists in the bucket
if aws s3 ls "$bucket_path$latest_file" 2>/dev/null; then
    echo "File $latest_file already exists in bucket"
else
    # File doesn't exist, upload it
    echo "Uploading $latest_file to bucket $bucket_path"
    aws s3 cp "$latest_file" "$bucket_path"
    echo "Upload completed"
fi