#!/usr/bin/env python3
"""
Fast Tahoe-100M downloader - downloads ONE parquet file directly
No overcomplicated dataset libraries - just wget + pandas
"""

import subprocess
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import anndata as ad
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_tahoe_fast(output_file="/workspace/data/tahoe_real.h5ad"):
    """Download real Tahoe-100M data efficiently"""
    
    # Direct parquet URL - just download ONE file
    parquet_url = "https://huggingface.co/api/datasets/tahoebio/Tahoe-100M/parquet/expression_data/train/0.parquet"
    temp_parquet = "/workspace/data/tahoe_sample.parquet"
    
    Path("/workspace/data").mkdir(exist_ok=True)
    
    # Download with urllib (works in Docker)
    logger.info("📥 Downloading ONE parquet file from Tahoe-100M...")
    import urllib.request
    from tqdm import tqdm
    
    # Download with progress bar
    with urllib.request.urlopen(parquet_url) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        block_size = 8192
        
        with open(temp_parquet, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    f.write(block)
                    pbar.update(len(block))
    
    # Read parquet file
    logger.info("📖 Reading parquet file...")
    df = pd.read_parquet(temp_parquet)
    
    logger.info(f"✅ Loaded {len(df)} cells")
    logger.info(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # Convert to AnnData format
    logger.info("🔄 Converting to AnnData...")
    
    # Tahoe-100M has 'genes' and 'expressions' as list columns
    if 'genes' in df.columns and 'expressions' in df.columns:
        # Get unique genes across all cells to create var index
        logger.info("Building gene vocabulary...")
        all_genes = set()
        for genes_list in df['genes']:
            all_genes.update(genes_list)
        
        sorted_genes = sorted(list(all_genes))
        gene_to_idx = {gene: idx for idx, gene in enumerate(sorted_genes)}
        
        # Build sparse expression matrix
        logger.info(f"Creating sparse matrix for {len(sorted_genes)} genes...")
        data, row_indices, col_indices = [], [], []
        
        for row_idx, (genes, expressions) in enumerate(zip(df['genes'], df['expressions'])):
            for gene, expr in zip(genes, expressions):
                if gene in gene_to_idx:
                    row_indices.append(row_idx)
                    col_indices.append(gene_to_idx[gene])
                    data.append(expr)
        
        X = csr_matrix((data, (row_indices, col_indices)), 
                       shape=(len(df), len(sorted_genes)))
        
        # Extract metadata (non-gene/expression columns)
        metadata_columns = [col for col in df.columns if col not in ['genes', 'expressions']]
        obs_df = df[metadata_columns].copy() if metadata_columns else pd.DataFrame(index=df.index)
        
        # If there's a drugname or perturbation column, format it for ST-Tahoe
        for col in obs_df.columns:
            if 'drug' in col.lower() or 'pert' in col.lower() or 'treatment' in col.lower():
                logger.info(f"Found perturbation column: {col}")
                # Ensure it's in the format ST-Tahoe expects
                if not obs_df[col].astype(str).str.contains(r"\[.*\]").any():
                    # Add brackets if not already formatted
                    obs_df['drugname_drugconc'] = obs_df[col].apply(
                        lambda x: f"[('{x}', 1.0, 'uM')]" if pd.notna(x) else "[('DMSO_TF', 0.0, 'uM')]"
                    )
        
        # Create var dataframe
        var_df = pd.DataFrame(index=sorted_genes)
        var_df['gene_name'] = sorted_genes
        
        # Create AnnData
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        
        # Save
        adata.write_h5ad(output_file)
        
        logger.info(f"✅ Success! Saved {adata.shape[0]} cells x {adata.shape[1]} genes")
        logger.info(f"📁 Output: {output_file}")
        logger.info(f"💾 File size: {Path(output_file).stat().st_size / 1e6:.1f} MB")
        
        # Show sample of perturbations if available
        if 'drugname_drugconc' in adata.obs.columns:
            logger.info(f"Sample perturbations: {adata.obs['drugname_drugconc'].value_counts().head()}")
        
        # Clean up
        Path(temp_parquet).unlink()
        
        return adata
    else:
        logger.error("Could not find gene expression columns in the data")
        logger.info(f"Available columns: {list(df.columns)}")
        return None

if __name__ == "__main__":
    download_tahoe_fast()
