#!/usr/bin/env python3
"""
Preprocess data to match ST-Tahoe's expected 2000 genes
"""

import pickle
import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_for_st_tahoe(
    input_file="/workspace/data/tahoe_real.h5ad",
    output_file="/workspace/data/tahoe_preprocessed.h5ad",
    var_dims_file="/workspace/models/ST-Tahoe/var_dims.pkl",
    gene_metadata_file="/workspace/data/gene_metadata.parquet"
):
    """Filter data to match ST-Tahoe's expected 2000 genes"""
    
    logger.info("📖 Loading data...")
    adata = ad.read_h5ad(input_file)
    logger.info(f"Original shape: {adata.shape}")
    
    # Load gene metadata to map token IDs to gene symbols
    logger.info("📋 Loading gene metadata...")
    gene_metadata = pd.read_parquet(gene_metadata_file)
    token_to_symbol = dict(zip(gene_metadata['token_id'].astype(str), gene_metadata['gene_symbol']))
    
    # Map our numeric gene IDs to gene symbols
    our_gene_ids = adata.var.index.tolist()
    our_gene_symbols = []
    for gene_id in our_gene_ids:
        symbol = token_to_symbol.get(str(gene_id), None)
        our_gene_symbols.append(symbol)
    
    logger.info(f"Mapped {sum(s is not None for s in our_gene_symbols)} genes to symbols")
    
    # Load expected genes from model
    logger.info("📋 Loading model gene list...")
    with open(var_dims_file, 'rb') as f:
        var_dims = pickle.load(f)
    
    model_genes = var_dims['gene_names']
    logger.info(f"Model expects {len(model_genes)} genes")
    
    # Find intersection between our gene symbols and model genes
    gene_mapping = {}
    for i, (gene_id, symbol) in enumerate(zip(our_gene_ids, our_gene_symbols)):
        if symbol in model_genes:
            gene_mapping[gene_id] = symbol
    
    logger.info(f"Found {len(gene_mapping)} matching genes")
    
    # Filter to matching genes
    matching_genes = list(gene_mapping.keys())
    adata_filtered = adata[:, matching_genes].copy()
    
    # Create expression matrix with all 2000 genes (fill missing with zeros)
    logger.info("🔧 Creating full 2000-gene matrix...")
    n_cells = adata_filtered.shape[0]
    X_new = np.zeros((n_cells, len(model_genes)))
    
    # Map our genes to model gene positions
    model_gene_to_idx = {gene: i for i, gene in enumerate(model_genes)}
    
    for i, gene in enumerate(matching_genes):
        model_gene = gene_mapping[gene]
        if model_gene in model_gene_to_idx:
            model_idx = model_gene_to_idx[model_gene]
            X_new[:, model_idx] = adata_filtered.X[:, i].toarray().flatten()
    
    # Create new AnnData with correct gene order
    adata_new = ad.AnnData(
        X=csr_matrix(X_new),
        obs=adata_filtered.obs.copy(),
        var=pd.DataFrame(index=model_genes)
    )
    
    # Add gene names
    adata_new.var['gene_name'] = model_genes
    
    logger.info(f"✅ Final shape: {adata_new.shape}")
    logger.info(f"   Non-zero genes: {(adata_new.X.sum(axis=0) > 0).sum()}")
    
    # Save
    adata_new.write_h5ad(output_file)
    logger.info(f"💾 Saved to {output_file}")
    
    return adata_new

if __name__ == "__main__":
    preprocess_for_st_tahoe()
