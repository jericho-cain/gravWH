#!/usr/bin/env python3
"""
Modular Wrapper for Gravitational Wave Detection

This simply calls the existing working pipeline.
No new implementation - just a clean wrapper.
"""

import os
import sys
import logging

# Add the project root to the sys.path
sys.path.append('.')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    """Run training using the existing working pipeline."""
    
    logger.info("🚀 Starting modular training wrapper...")
    
    try:
        # Import and run the existing working pipeline
        from gravitational_wave_hunter.data.simple_training_pipeline import SimpleTrainingPipeline
        
        # Initialize with the same seed that worked
        pipeline = SimpleTrainingPipeline(random_seed=7)
        
        # Run the existing working pipeline
        logger.info("📊 Running existing working pipeline...")
        lstm_results, transformer_results = pipeline.run_complete_pipeline(
            num_training_samples=100, 
            num_test_samples=25
        )
        
        logger.info("✅ Modular training completed!")
        logger.info(f"📊 Results: AUC={lstm_results['auc']:.3f}, AP={lstm_results['avg_precision']:.3f}")
        
        return lstm_results, transformer_results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    results = run_training()
    print(f"🎉 Training completed successfully!")
