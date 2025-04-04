#!/bin/sh

export CONTAINER_NAME=llm-benchmark

export IMG_NAME="nvcr.io/nim/${1}"

export LOCAL_NIM_CACHE=/ephemeral/.cache
mkdir -p "${LOCAL_NIM_CACHE}"

# Start the LLM NIM
docker run -it --rm --name=${CONTAINER_NAME} \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -e NIM_LOW_MEMORY_MODE=1 \
  -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  ${IMG_NAME}