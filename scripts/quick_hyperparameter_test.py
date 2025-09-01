#!/usr/bin/env python3
"""
Quick Hyperparameter Test for CWT-LSTM Autoencoder
Smaller grid search for initial testing and validation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.hyperparameter_grid_search import HyperparameterGridSearch

def main():
    """Run a quick hyperparameter test with fewer combinations."""
    print("⚡ Quick Hyperparameter Test for CWT-LSTM Autoencoder")
    print("=" * 60)
    
    # Create a smaller grid search for testing
    grid_search = HyperparameterGridSearch(results_dir="quick_test_results")
    
    # Override with smaller grid for quick testing
    grid_search.hyperparameter_grid = {
        'latent_dim': [16, 32],           # 2 options
        'lstm_hidden': [32, 64],          # 2 options  
        'learning_rate': [0.001, 0.01],   # 2 options
        'batch_size': [8, 16],            # 2 options
        'epochs': [20, 30]                # 2 options
    }
    
    print(f"🧪 Quick test with {grid_search._get_total_combinations()} combinations")
    print("This should take about 10-15 minutes...")
    
    # Run the test
    grid_search.run_grid_search(num_samples=100)  # Smaller dataset too
    
    # Quick analysis
    print(f"\n📊 QUICK RESULTS SUMMARY")
    print("-" * 40)
    
    if grid_search.results:
        # Find best by avg_precision
        best_result = max(grid_search.results, key=lambda x: x['metrics']['avg_precision'])
        
        print(f"🏆 Best Configuration:")
        print(f"   latent_dim: {best_result['config']['latent_dim']}")
        print(f"   lstm_hidden: {best_result['config']['lstm_hidden']}")
        print(f"   learning_rate: {best_result['config']['learning_rate']}")
        print(f"   batch_size: {best_result['config']['batch_size']}")
        print(f"   epochs: {best_result['config']['epochs']}")
        
        print(f"\n📈 Best Metrics:")
        for metric, value in best_result['metrics'].items():
            print(f"   {metric}: {value:.4f}")
    
    print(f"\n✅ Quick test complete! Check quick_test_results/ for details.")

if __name__ == "__main__":
    main()
