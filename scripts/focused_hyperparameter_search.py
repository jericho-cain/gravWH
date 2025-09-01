#!/usr/bin/env python3
"""
Focused Hyperparameter Search for CWT-LSTM Autoencoder
Targeted search around the current best configuration with more granular values.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.hyperparameter_grid_search import HyperparameterGridSearch

def main():
    """Run a focused hyperparameter search around the current best configuration."""
    print("🎯 Focused Hyperparameter Search for CWT-LSTM Autoencoder")
    print("=" * 70)
    
    # Create a focused grid search
    grid_search = HyperparameterGridSearch(results_dir="focused_search_results")
    
    # Focused grid around current best: latent_dim=16, lstm_hidden=32, lr=0.001
    grid_search.hyperparameter_grid = {
        'latent_dim': [12, 16, 20, 24],           # Around 16
        'lstm_hidden': [24, 32, 40, 48],          # Around 32
        'learning_rate': [0.0005, 0.001, 0.002, 0.005],  # Around 0.001
        'batch_size': [6, 8, 10, 12],             # Around 8
        'epochs': [15, 20, 25, 30]                # Around 20
    }
    
    total_combinations = grid_search._get_total_combinations()
    print(f"🎯 Focused search with {total_combinations} combinations")
    print(f"⏱️  Estimated time: {total_combinations * 0.5:.1f} minutes")
    print(f"📊 Current best: latent_dim=16, lstm_hidden=32, lr=0.001, batch=8, epochs=20")
    print(f"🎯 Target: avg_precision > 0.780 (current main model)")
    
    # Run the focused search
    grid_search.run_grid_search(num_samples=200)  # Use full dataset
    
    # Quick analysis
    print(f"\n📊 FOCUSED SEARCH RESULTS SUMMARY")
    print("-" * 50)
    
    if grid_search.results:
        # Find best by avg_precision
        best_result = max(grid_search.results, key=lambda x: x['metrics']['avg_precision'])
        
        print(f"🏆 Best Configuration Found:")
        print(f"   latent_dim: {best_result['config']['latent_dim']}")
        print(f"   lstm_hidden: {best_result['config']['lstm_hidden']}")
        print(f"   learning_rate: {best_result['config']['learning_rate']}")
        print(f"   batch_size: {best_result['config']['batch_size']}")
        print(f"   epochs: {best_result['config']['epochs']}")
        
        print(f"\n📈 Best Metrics:")
        for metric, value in best_result['metrics'].items():
            print(f"   {metric}: {value:.4f}")
        
        # Compare to current best
        current_best = 0.780  # From main model
        improvement = best_result['metrics']['avg_precision'] - current_best
        
        if improvement > 0:
            print(f"\n🎉 IMPROVEMENT FOUND!")
            print(f"   Previous best: {current_best:.4f}")
            print(f"   New best: {best_result['metrics']['avg_precision']:.4f}")
            print(f"   Improvement: +{improvement:.4f} ({improvement/current_best*100:.2f}%)")
        else:
            print(f"\n📊 No improvement over current best ({current_best:.4f})")
            print(f"   Best found: {best_result['metrics']['avg_precision']:.4f}")
            print(f"   Difference: {improvement:.4f}")
    
    print(f"\n✅ Focused search complete! Check focused_search_results/ for details.")
    
    # Show top 5 configurations
    if grid_search.results:
        print(f"\n🥇 TOP 5 CONFIGURATIONS:")
        print("-" * 80)
        
        # Sort by avg_precision
        sorted_results = sorted(grid_search.results, 
                               key=lambda x: x['metrics']['avg_precision'], 
                               reverse=True)
        
        print(f'{"Rank":<4} {"latent_dim":<10} {"lstm_hidden":<11} {"lr":<8} {"batch":<6} {"epochs":<7} {"avg_prec":<10}')
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:5], 1):
            config = result['config']
            metrics = result['metrics']
            print(f'{i:<4} {config["latent_dim"]:<10} {config["lstm_hidden"]:<11} '
                  f'{config["learning_rate"]:<8.3f} {config["batch_size"]:<6} {config["epochs"]:<7} '
                  f'{metrics["avg_precision"]:<10.4f}')

if __name__ == "__main__":
    main()
