#!/bin/bash
# Script to build and run the simulation container

IMAGE_NAME="driver-fatigue:simulation"
DOCKERFILE="Dockerfile.simulation_py39"

echo "Building simulation image ($IMAGE_NAME)..."
docker build -t $IMAGE_NAME -f $DOCKERFILE .

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Running simulation..."
docker run --rm \
    --net=host \
    --privileged \
    --device=/dev/video0 \
    --device=/dev/video1 \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE_NAME "$@"
