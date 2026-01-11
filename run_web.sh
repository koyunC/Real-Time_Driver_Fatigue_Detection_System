#!/bin/bash
# Script to build and run the web simulation container

IMAGE_NAME="driver-fatigue:simulation"
DOCKERFILE="Dockerfile.simulation_py39"

echo "Building simulation image ($IMAGE_NAME)..."
docker build -t $IMAGE_NAME -f $DOCKERFILE .

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Running web simulation..."
echo "Open http://localhost:5000 in your browser."

docker run --rm \
    --net=host \
    --privileged \
    --device=/dev/video0 \
    --device=/dev/video1 \
    -e PYTHONUNBUFFERED=1 \
    --entrypoint python \
    $IMAGE_NAME web_main.py --device cpu --camera 0 "$@"
