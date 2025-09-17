#!/usr/bin/env python3
"""
Modular Pipeline Runner

This script demonstrates how to use the modular pipeline approach:
1. Download GW events (run once)
2. Train model
3. Evaluate model

Usage:
    python run_modular_pipeline.py --step all
    python run_modular_pipeline.py --step download
    python run_modular_pipeline.py --step train
    python run_modular_pipeline.py --step evaluate
"""

import argparse
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_download_step():
    """Step 1: Download GW events."""
    logger.info("🌌 Step 1: Downloading GW events...")
    
    try:
        from download_gw_events import download_gw_events
        successful_downloads, failed_downloads = download_gw_events()
        
        if len(successful_downloads) > 0:
            logger.info(f"✅ Download step completed! {len(successful_downloads)} events downloaded.")
            return True
        else:
            logger.error("❌ No events were downloaded successfully.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Download step failed: {e}")
        return False

def run_training_step():
    """Step 2: Train model."""
    logger.info("🚀 Step 2: Training model...")
    
    try:
        # Add the project root to the sys.path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'gravitational_wave_hunter', 'data'))
        from modular_training_pipeline import ModularTrainingPipeline
        
        # Initialize pipeline
        pipeline = ModularTrainingPipeline(random_seed=7)
        
        # Run training
        lstm_results, transformer_results = pipeline.run_training_pipeline(
            num_training_samples=100, 
            num_test_samples=25
        )
        
        logger.info(f"✅ Training step completed!")
        logger.info(f"📊 Results: AUC={lstm_results['auc']:.3f}, AP={lstm_results['avg_precision']:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        return False

def run_evaluation_step():
    """Step 3: Evaluate model."""
    logger.info("🔍 Step 3: Evaluating model...")
    
    try:
        from evaluate_model import evaluate_model
        
        # Run evaluation
        results = evaluate_model(num_test_samples=25, random_seed=7)
        
        logger.info(f"✅ Evaluation step completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   AUC: {results['auc']:.3f}")
        logger.info(f"   Average Precision: {results['avg_precision']:.3f}")
        logger.info(f"   Precision: {results['precision']:.3f}")
        logger.info(f"   Recall: {results['recall']:.3f}")
        logger.info(f"   F1-Score: {results['f1']:.3f}")
        logger.info(f"   Detected Signals: {results['detected_signals']}")
        logger.info(f"   Missed Signals: {results['missed_signals']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation step failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run modular gravitational wave detection pipeline')
    parser.add_argument('--step', choices=['all', 'download', 'train', 'evaluate'], 
                       default='all', help='Which step to run')
    parser.add_argument('--training-samples', type=int, default=100, 
                       help='Number of training samples')
    parser.add_argument('--test-samples', type=int, default=25, 
                       help='Number of test samples')
    parser.add_argument('--random-seed', type=int, default=7, 
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting modular gravitational wave detection pipeline...")
    logger.info(f"📊 Configuration: {args.training_samples} training samples, {args.test_samples} test samples")
    
    success = True
    
    if args.step in ['all', 'download']:
        success &= run_download_step()
        
    if args.step in ['all', 'train'] and success:
        success &= run_training_step()
        
    if args.step in ['all', 'evaluate'] and success:
        success &= run_evaluation_step()
    
    if success:
        logger.info("🎉 Modular pipeline completed successfully!")
    else:
        logger.error("❌ Modular pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
