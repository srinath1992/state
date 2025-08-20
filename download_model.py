#!/usr/bin/env python3
"""
Fast ST-Tahoe model download on host system
Avoids Docker container download slowdowns
"""

import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_st_tahoe():
    """Download ST-Tahoe model with progress tracking"""
    model_path = Path("./models/ST-Tahoe")
    
    if model_path.exists():
        final_ckpt = model_path / "final.ckpt"
        if final_ckpt.exists() and final_ckpt.stat().st_size > 1000000:  # > 1MB
            logger.info(f"✅ ST-Tahoe model already downloaded: {model_path}")
            return model_path
    
    logger.info("📥 Downloading ST-Tahoe model (~3GB) - this will take a few minutes...")
    
    try:
        downloaded_path = snapshot_download(
            repo_id="arcinstitute/ST-Tahoe",
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True  # Resume if interrupted
        )
        
        # Verify download
        final_ckpt = model_path / "final.ckpt"
        var_dims = model_path / "var_dims.pkl"
        config_yaml = model_path / "config.yaml"
        
        for file_path in [final_ckpt, var_dims, config_yaml]:
            if not file_path.exists():
                raise FileNotFoundError(f"Missing: {file_path}")
            if file_path.stat().st_size < 10:
                raise ValueError(f"File too small (LFS pointer?): {file_path}")
        
        logger.info(f"✅ ST-Tahoe model downloaded successfully!")
        logger.info(f"   Model size: {final_ckpt.stat().st_size / (1024*1024*1024):.1f} GB")
        logger.info(f"   Location: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise

if __name__ == "__main__":
    download_st_tahoe()
