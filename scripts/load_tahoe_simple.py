#!/usr/bin/env python3
"""
Simple Tahoe-100M loader following official HuggingFace tutorial
Based on: https://huggingface.co/datasets/tahoebio/Tahoe-100M/blob/main/tutorials/loading_data.ipynb

This follows the official approach exactly - no reinventing the wheel!
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata
import pandas as pd
from datasets import load_dataset
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_anndata_from_generator(generator, gene_vocab, sample_size=None):
    """
    Official function from Tahoe-100M tutorial
    Converts HuggingFace dataset records to AnnData format
    """
    sorted_vocab_items = sorted(gene_vocab.items())
    token_ids, gene_names = zip(*sorted_vocab_items)
    token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}

    data, indices, indptr = [], [], [0]
    obs_data = []

    for i, cell in enumerate(generator):
        if sample_size is not None and i >= sample_size:
            break
            
        genes = cell['genes']
        expressions = cell['expressions']
        
        # Handle negative expressions (from tutorial)
        if expressions[0] < 0: 
            genes = genes[1:]
            expressions = expressions[1:]

        col_indices = [token_id_to_col_idx[gene] for gene in genes if gene in token_id_to_col_idx]
        valid_expressions = [expr for gene, expr in zip(genes, expressions) if gene in token_id_to_col_idx]

        data.extend(valid_expressions)
        indices.extend(col_indices)
        indptr.append(len(data))

        # Store observation metadata
        obs_entry = {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
        obs_data.append(obs_entry)
        
        if i % 1000 == 0:
            logger.info(f"Processed {i} cells...")

    # Create sparse matrix (State's preferred format)
    expr_matrix = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(gene_names)))
    obs_df = pd.DataFrame(obs_data)
    var_df = pd.DataFrame(index=gene_names)
    
    # Create AnnData object
    adata = anndata.AnnData(X=expr_matrix, obs=obs_df, var=var_df)
    
    logger.info(f"Created AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
    return adata


def main():
    """Simple CLI for loading Tahoe-100M subsets"""
    parser = argparse.ArgumentParser(description="Load Tahoe-100M dataset following official tutorial")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of cells to load")
    parser.add_argument("--output", type=str, default="tahoe_sample.h5ad", help="Output H5AD file")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Loading Tahoe-100M dataset (sample size: {args.sample_size})")
    
    try:
        # Load dataset using official HuggingFace approach
        logger.info("📥 Loading dataset from HuggingFace...")
        dataset = load_dataset("tahoebio/Tahoe-100M", streaming=args.streaming, trust_remote_code=True)
        
        # Get gene vocabulary
        logger.info("📋 Loading gene vocabulary...")
        gene_vocab = dataset["train"].info.features["genes"].feature._vocab
        logger.info(f"Gene vocabulary size: {len(gene_vocab)}")
        
        # Convert to AnnData using official function
        logger.info("🔄 Converting to AnnData format...")
        train_data = dataset["train"]
        adata = create_anndata_from_generator(train_data, gene_vocab, sample_size=args.sample_size)
        
        # Save in State-compatible format
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving to {output_path}...")
        adata.write_h5ad(output_path, compression='gzip')
        
        # Success summary
        logger.info("\n" + "="*50)
        logger.info("✅ SUCCESS! Tahoe-100M subset loaded")
        logger.info(f"📊 Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
        logger.info(f"📁 Output: {output_path}")
        logger.info(f"💾 File size: {output_path.stat().st_size / 1e6:.1f} MB")
        logger.info("🚀 Ready for State training!")
        logger.info("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
