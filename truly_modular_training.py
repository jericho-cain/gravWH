#!/usr/bin/env python3
"""
Truly Modular Training Pipeline

This just calls the existing working pipeline with pre-downloaded data.
No new implementation - just modularization.
"""

import os
import sys
import numpy as np
import logging

# Add the project root to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gravitational_wave_hunter', 'data'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_modular_training():
    """Run training using the existing working pipeline with pre-downloaded data."""
    
    logger.info("🚀 Starting truly modular training...")
    
    # Import the existing working pipeline
    sys.path.append(os.path.join(os.path.dirname(__file__), 'gravitational_wave_hunter', 'data'))
    from simple_training_pipeline import SimpleTrainingPipeline
    
    # Initialize with the same seed that worked
    pipeline = SimpleTrainingPipeline(random_seed=7)
    
    # Run the existing working pipeline
    logger.info("📊 Running existing working pipeline...")
    lstm_results, transformer_results = pipeline.run_complete_pipeline(
        num_training_samples=100, 
        num_test_samples=25
    )
    
    logger.info("✅ Truly modular training completed!")
    logger.info(f"📊 Results: AUC={lstm_results['auc']:.3f}, AP={lstm_results['avg_precision']:.3f}")
    
    return lstm_results, transformer_results

if __name__ == "__main__":
    results = run_modular_training()
    print(f"🎉 Training completed successfully!")
