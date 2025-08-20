#!/usr/bin/env bash
# Simple, reliable ST-Tahoe model download using wget
# No Python dependencies, just standard Unix tools

set -e

MODEL_DIR="./models/ST-Tahoe"
BASE_URL="https://huggingface.co/arcinstitute/ST-Tahoe/resolve/main"

# Check if model already exists
if [[ -f "$MODEL_DIR/final.ckpt" ]] && [[ $(stat -c%s "$MODEL_DIR/final.ckpt" 2>/dev/null || stat -f%z "$MODEL_DIR/final.ckpt" 2>/dev/null) -gt 1000000000 ]]; then
    echo "✅ ST-Tahoe model already downloaded at $MODEL_DIR"
    exit 0
fi

echo "📥 Downloading ST-Tahoe model (~3GB)..."
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# Download essential files with progress
echo "⬇️  Downloading model checkpoint (2.8GB)..."
wget -c "$BASE_URL/final.ckpt" -O final.ckpt

echo "⬇️  Downloading config files..."
wget -c "$BASE_URL/config.yaml" -O config.yaml
wget -c "$BASE_URL/var_dims.pkl" -O var_dims.pkl

# Optional: Download other files if needed
echo "⬇️  Downloading additional files..."
wget -c "$BASE_URL/data_module.torch" -O data_module.torch || echo "Warning: Could not download data_module.torch"
wget -c "$BASE_URL/pert_onehot_map.pt" -O pert_onehot_map.pt || echo "Warning: Could not download pert_onehot_map.pt"

echo "✅ ST-Tahoe model downloaded successfully!"
echo "   Model size: $(du -h final.ckpt | cut -f1)"
echo "   Location: $(pwd)"
