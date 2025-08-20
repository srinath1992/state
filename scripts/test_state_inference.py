#!/usr/bin/env python3
"""
Simple State inference test using pre-trained ST-Tahoe model
Following official HuggingFace documentation approach
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess

import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_st_tahoe_model(output_dir: str = "./models/ST-Tahoe") -> Path:
    """Download ST-Tahoe model from HuggingFace using hub library (handles LFS properly)"""
    from huggingface_hub import snapshot_download
    
    model_path = Path(output_dir)
    
    # Check if model files exist and are not LFS pointers
    config_path = model_path / "config.yaml"
    final_ckpt = model_path / "final.ckpt"
    
    if config_path.exists() and final_ckpt.exists():
        # Check if final.ckpt is actually downloaded (not an LFS pointer)
        if final_ckpt.stat().st_size > 1000:  # Real model file should be > 1KB
            logger.info(f"✅ ST-Tahoe model already exists at {model_path}")
            return model_path
        else:
            logger.info("🔄 Model exists but files are LFS pointers, re-downloading...")
    
    logger.info("📥 Downloading ST-Tahoe model from HuggingFace (~3GB - this may take 5-10 minutes)...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use HuggingFace Hub to download with proper LFS handling and progress
        from tqdm import tqdm
        import time
        
        logger.info("🔄 Starting download with resumption support...")
        start_time = time.time()
        
        downloaded_path = snapshot_download(
            repo_id="arcinstitute/ST-Tahoe",
            local_dir=str(model_path),
            resume_download=True,  # Resume interrupted downloads
            force_download=False,  # Don't redownload existing files
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ ST-Tahoe model downloaded in {elapsed/60:.1f} minutes to {downloaded_path}")
        
        # Verify key files exist and have reasonable sizes
        key_files = ["config.yaml", "final.ckpt", "var_dims.pkl"]
        for filename in key_files:
            file_path = model_path / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required file missing: {filename}")
            if file_path.stat().st_size < 10:  # Should be more than 10 bytes
                raise ValueError(f"File {filename} appears to be corrupted or empty")
        
        logger.info("✅ Model files verified successfully")
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        raise


def create_test_data(output_file: str = "./data/test_drug_data.h5ad", n_cells: int = 1000) -> Path:
    """
    Create test AnnData with drug perturbation format expected by ST-Tahoe
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🧪 Creating test drug perturbation data ({n_cells} cells)...")
    
    # Create realistic test data structure
    n_genes = 2000  # Reasonable number for testing
    
    # Generate synthetic expression data (sparse matrix)
    np.random.seed(42)  # Reproducible results
    
    # Create sparse expression matrix (most genes have zero expression)
    density = 0.1  # 10% non-zero values (typical for single-cell)
    data = np.random.negative_binomial(5, 0.3, size=int(n_cells * n_genes * density))
    
    # Create sparse matrix
    row_indices = np.random.choice(n_cells, size=len(data), replace=True)
    col_indices = np.random.choice(n_genes, size=len(data), replace=True)
    
    expr_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_genes))
    
    # Create gene names (using standard format)
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    
    # Create drug perturbation annotations (ST-Tahoe expects 'drugname_drugconc')
    drug_names = ["Doxorubicin", "Paclitaxel", "5-Fluorouracil", "Cisplatin", "Methotrexate", "DMSO"]
    concentrations = ["0.1uM", "1uM", "10uM", "100uM", "Control"]
    
    # Create drugname_drugconc column
    drugname_drugconc = []
    for i in range(n_cells):
        if i % 10 == 0:  # 10% control cells
            drugname_drugconc.append("DMSO_Control")
        else:
            drug = np.random.choice(drug_names[:-1])  # Exclude DMSO from random selection
            conc = np.random.choice(concentrations[:-1])  # Exclude Control
            drugname_drugconc.append(f"{drug}_{conc}")
    
    # Create additional metadata
    obs_data = pd.DataFrame({
        'drugname_drugconc': drugname_drugconc,
        'cell_type': np.random.choice(['HeLa', 'A549', 'MCF7'], size=n_cells),
        'batch': np.random.choice(['batch_1', 'batch_2', 'batch_3'], size=n_cells),
        'cell_id': [f"cell_{i:06d}" for i in range(n_cells)]
    })
    
    var_data = pd.DataFrame({
        'gene_name': gene_names,
        'gene_id': gene_names,  # Sometimes needed
    }, index=gene_names)
    
    # Create AnnData object
    adata = ad.AnnData(
        X=expr_matrix,
        obs=obs_data,
        var=var_data
    )
    
    # Add some standard single-cell metadata
    adata.uns['created_by'] = 'test_state_inference.py'
    adata.uns['n_drugs'] = len(set(drugname_drugconc))
    
    # Save the data
    logger.info(f"💾 Saving test data to {output_path}...")
    adata.write_h5ad(output_path, compression='gzip')
    
    logger.info(f"✅ Test data created: {adata.shape[0]} cells x {adata.shape[1]} genes")
    logger.info(f"   Drug conditions: {len(set(drugname_drugconc))}")
    logger.info(f"   Cell types: {list(set(obs_data['cell_type']))}")
    
    return output_path


def run_inference_test(model_dir: Path, test_data: Path, output_dir: str = "./results") -> bool:
    """
    Run State inference test using pre-trained ST-Tahoe model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Running State inference test...")
    logger.info(f"   Model: {model_dir}")
    logger.info(f"   Data: {test_data}")
    logger.info(f"   Output: {output_path}")
    
    try:
        # Build the inference command
        cmd = [
            "state", "tx", "infer",
            "--model_dir", str(model_dir),
            "--pert_col", "drugname_drugconc",  # ST-Tahoe expects this column
            "--adata", str(test_data),
            "--output", str(output_path)
        ]
        
        logger.info(f"🔧 Running command: {' '.join(cmd)}")
        
        # Run the inference
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("✅ State inference completed successfully!")
        logger.info("📋 Output summary:")
        if result.stdout:
            logger.info(f"   stdout: {result.stdout[-500:]}")  # Last 500 chars
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ State inference failed: {e}")
        logger.error(f"   stdout: {e.stdout}")
        logger.error(f"   stderr: {e.stderr}")
        return False


def main():
    """Main test workflow"""
    parser = argparse.ArgumentParser(description="Test State inference with ST-Tahoe")
    parser.add_argument("--model-dir", type=str, default="./models/ST-Tahoe", help="ST-Tahoe model directory")
    parser.add_argument("--test-data", type=str, default="./data/test_drug_data.h5ad", help="Test data file")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--n-cells", type=int, default=1000, help="Number of test cells")
    
    args = parser.parse_args()
    
    try:
        logger.info("🎯 Starting State inference test workflow...")
        
        # Step 1: Download ST-Tahoe model
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Download ST-Tahoe Model")
        logger.info("="*60)
        model_path = download_st_tahoe_model(args.model_dir)
        
        # Step 2: Create test data
        logger.info("\n" + "="*60) 
        logger.info("STEP 2: Create Test Drug Perturbation Data")
        logger.info("="*60)
        test_data_path = create_test_data(args.test_data, args.n_cells)
        
        # Step 3: Run inference
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Run State Inference")
        logger.info("="*60)
        success = run_inference_test(model_path, test_data_path, args.output)
        
        # Summary
        logger.info("\n" + "="*60)
        if success:
            logger.info("🎉 STATE INFERENCE TEST COMPLETED SUCCESSFULLY!")
            logger.info(f"📁 Model: {model_path}")
            logger.info(f"📊 Test data: {test_data_path}")
            logger.info(f"📈 Results: {args.output}")
            logger.info("✅ Pre-trained State model is working correctly!")
        else:
            logger.error("❌ STATE INFERENCE TEST FAILED!")
            return 1
            
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test workflow failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
