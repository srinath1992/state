#!/bin/bash
# Alternative: Download ST-Tahoe model using git with LFS on host
# This avoids Docker container download issues

set -e

MODEL_DIR="./models/ST-Tahoe"
REPO_URL="https://huggingface.co/arcinstitute/ST-Tahoe"

echo "🚀 Downloading ST-Tahoe model using git clone (alternative method)"

# Check if model already exists
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/final.ckpt" ] && [ -s "$MODEL_DIR/final.ckpt" ]; then
    echo "✅ ST-Tahoe model already exists: $MODEL_DIR"
    ls -lh "$MODEL_DIR/final.ckpt"
    exit 0
fi

# Create models directory
mkdir -p models
cd models

echo "📥 Cloning repository (this downloads ~3GB)..."

# Remove existing directory if it exists but is incomplete
if [ -d "ST-Tahoe" ]; then
    echo "🧹 Removing incomplete download..."
    rm -rf ST-Tahoe
fi

# Clone with git (handles LFS automatically if git-lfs is installed)
if command -v git-lfs >/dev/null 2>&1; then
    echo "✅ Using git-lfs for large file support"
    git clone "$REPO_URL"
    cd ST-Tahoe
    git lfs pull
else
    echo "⚠️  git-lfs not found - downloading without LFS (may not work)"
    git clone "$REPO_URL"
fi

echo "✅ Model download completed!"
echo "📁 Model location: $(pwd)/ST-Tahoe"
ls -lh ST-Tahoe/final.ckpt 2>/dev/null || echo "⚠️  final.ckpt not found - may be LFS pointer"

