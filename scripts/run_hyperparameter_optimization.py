#!/usr/bin/env python3
"""
Run Hyperparameter Optimization for CWT-LSTM Autoencoder
Simple script to execute the complete hyperparameter optimization pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.hyperparameter_grid_search import HyperparameterGridSearch
from scripts.analyze_hyperparameter_results import HyperparameterAnalyzer

def main():
    """Run the complete hyperparameter optimization pipeline."""
    print("🚀 CWT-LSTM Autoencoder Hyperparameter Optimization")
    print("=" * 60)
    
    # Step 1: Run Grid Search
    print("\n🔍 STEP 1: Running Hyperparameter Grid Search")
    print("-" * 50)
    
    grid_search = HyperparameterGridSearch()
    grid_search.run_grid_search(num_samples=200)
    
    # Step 2: Analyze Results
    print("\n📊 STEP 2: Analyzing Results")
    print("-" * 50)
    
    analyzer = HyperparameterAnalyzer()
    analyzer.generate_comprehensive_report()
    
    # Step 3: Summary
    print("\n✅ OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print("📁 Check the following files:")
    print("   • hyperparameter_results/results_summary.csv - Complete results table")
    print("   • hyperparameter_results/best_config.json - Best configuration")
    print("   • hyperparameter_results/analysis/ - Visualization plots")
    print("\n🎯 To find the best configuration:")
    print("   1. Open hyperparameter_results/results_summary.csv")
    print("   2. Sort by 'avg_precision' column (descending)")
    print("   3. The top row contains your optimal hyperparameters!")
    
    if analyzer.best_config:
        print(f"\n🏆 RECOMMENDED CONFIGURATION:")
        print(f"   latent_dim: {analyzer.best_config['latent_dim']}")
        print(f"   lstm_hidden: {analyzer.best_config['lstm_hidden']}")
        print(f"   learning_rate: {analyzer.best_config['learning_rate']}")
        print(f"   batch_size: {analyzer.best_config['batch_size']}")
        print(f"   epochs: {analyzer.best_config['epochs']}")

if __name__ == "__main__":
    main()
