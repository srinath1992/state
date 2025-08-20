#!/usr/bin/env python3
"""
Download Tahoe-100M gene metadata to map token IDs to gene symbols
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_gene_metadata():
    """Download gene metadata from Tahoe-100M"""
    
    logger.info("📥 Downloading Tahoe-100M gene metadata...")
    
    # Direct URL to gene metadata parquet
    url = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/gene_metadata.parquet"
    
    # Download and read
    gene_metadata = pd.read_parquet(url)
    
    logger.info(f"✅ Downloaded {len(gene_metadata)} gene mappings")
    logger.info(f"Columns: {list(gene_metadata.columns)}")
    logger.info(f"First 5 entries:\n{gene_metadata.head()}")
    
    # Save locally
    gene_metadata.to_parquet("/workspace/data/gene_metadata.parquet")
    logger.info("💾 Saved to /workspace/data/gene_metadata.parquet")
    
    # Create token_id to gene_symbol mapping
    token_to_symbol = dict(zip(gene_metadata['token_id'].astype(str), gene_metadata['gene_symbol']))
    
    logger.info(f"Sample mappings: {dict(list(token_to_symbol.items())[:5])}")
    
    return gene_metadata, token_to_symbol

if __name__ == "__main__":
    download_gene_metadata()
