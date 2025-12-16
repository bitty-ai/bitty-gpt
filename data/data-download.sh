
#!/bin/bash

mkdir -p data
cd data

TRAIN_FILE="TinyStoriesV2-GPT4-train.txt"
VALID_FILE="TinyStoriesV2-GPT4-valid.txt"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "$TRAIN_FILE not found. Downloading..."
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
else
    echo "$TRAIN_FILE already exists. Skipping download."
fi

if [ ! -f "$VALID_FILE" ]; then
    echo "$VALID_FILE not found. Downloading..."
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
else
    echo "$VALID_FILE already exists. Skipping download."
fi


