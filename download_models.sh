#!/bin/bash

# There also a resnet34 model available, which is trained on same dataset in experiments term.
# However, it's performance is not as good as resnet18 on jetson nano.
# You can still download it for fun if you want.

MODEL_DIR="models"
mkdir -p $MODEL_DIR

# Expected SHA256 checksums
CHECKSUM_RESNET18="d4bf16c210ba116d43ebae7ebd19a2bfa3bc80dbe305746c066a1e552dd5b536"
CHECKSUM_DETECTOR="b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"

download_and_verify() {
    local url=$1
    local dest=$2
    local expected_checksum=$3
    local name=$4

    if [ -f "$dest" ]; then
        echo "Checking existing $name..."
        current_checksum=$(sha256sum "$dest" | awk '{print $1}')
        if [ "$current_checksum" == "$expected_checksum" ]; then
            echo "$name already exists and is valid. Skipping."
            return 0
        else
            echo "$name exists but checksum mismatch. Deleting and re-downloading."
            rm "$dest"
        fi
    fi

    echo "Downloading $name..."
    wget -L -O "$dest" "$url"
    
    if [ $? -ne 0 ]; then
        echo "Error: Download failed for $name"
        return 1
    fi

    echo "Verifying $name..."
    current_checksum=$(sha256sum "$dest" | awk '{print $1}')
    if [ "$current_checksum" == "$expected_checksum" ]; then
        echo "$name verified successfully."
    else
        echo "Error: Checksum mismatch for $name."
        echo "Expected: $expected_checksum"
        echo "Got:      $current_checksum"
        echo "Deleting corrupted file."
        rm "$dest"
        return 1
    fi
}

echo "-------------------------------------------------------"
echo "Starting to download models for Fatigue Detection..."
echo "-------------------------------------------------------"

URL_RESNET18="https://huggingface.co/koyunC/Driver_Fatigue_Detection/resolve/main/resnet18-e_20-d_10k.pth"
URL_DETECTOR="https://huggingface.co/koyunC/Driver_Fatigue_Detection/resolve/main/detector.tflite"

download_and_verify "$URL_RESNET18" "$MODEL_DIR/resnet18-e_20-d_10k.pth" "$CHECKSUM_RESNET18" "ResNet18 weights" || exit 1
download_and_verify "$URL_DETECTOR" "$MODEL_DIR/detector.tflite" "$CHECKSUM_DETECTOR" "Face Detector (TFLite)" || exit 1

echo "-------------------------------------------------------"
echo "All models are ready in the /$MODEL_DIR directory!"
echo "-------------------------------------------------------"
