#!/bin/bash

# Make directories
mkdir -p wavlm
mkdir -p test
mkdir -p hifigan
mkdir -p checkpoints

# Create symlink
ln -s /content/FreeVC/test DUMMY

# Install requirements
pip install -r requirements.txt
pip install webrtcvad
pip install -qq pyannote.audio

# Download WavLM-Large.pt
wget https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt -P wavlm/

# Download vocoder
gdown 1RkZ8reW0WjR9lE_ztTnN1qFVx24JZhhy
mv generator_v1 hifigan/

# Download models
gdown 1-GHhi-p-ms2aTPGUECzRQwMwQZw2MJ70
mv freevc.pth checkpoints/

# Uncomment the following lines if you want to download freevc-s.pth
# gdown 1-HP6xrPPMp9a5CphywZmRq40sftKMq5d
# mv freevc-s.pth checkpoints/

echo "FreeVC setup completed successfully!"