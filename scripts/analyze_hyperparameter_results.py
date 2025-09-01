#!/usr/bin/env python3
"""
Analyze Hyperparameter Grid Search Results
Provides easy-to-read analysis and visualization of grid search results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

class HyperparameterAnalyzer:
    """
    Analyze and visualize hyperparameter grid search results.
    """
    
    def __init__(self, results_dir: str = "hyperparameter_results"):
        """
        Initialize analyzer with results directory.
        
        Parameters
        ----------
        results_dir : str
            Directory containing grid search results.
        """
        self.results_dir = Path(results_dir)
        self.df = None
        self.best_config = None
        
    def load_results(self) -> None:
        """Load results from CSV file."""
        csv_path = self.results_dir / "results_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(self.df)} hyperparameter combinations")
        
        # Load best config
        best_config_path = self.results_dir / "best_config.json"
        if best_config_path.exists():
            with open(best_config_path, 'r') as f:
                best_data = json.load(f)
                self.best_config = best_data['config']
                print(f"🏆 Best config loaded: avg_precision = {best_data['score']:.4f}")
    
    def get_top_configs(self, metric: str = 'avg_precision', top_k: int = 10) -> pd.DataFrame:
        """
        Get top K configurations for a given metric.
        
        Parameters
        ----------
        metric : str
            Metric to rank by.
        top_k : int
            Number of top configurations to return.
            
        Returns
        -------
        pd.DataFrame
            Top K configurations sorted by metric.
        """
        if self.df is None:
            self.load_results()
        
        return self.df.nlargest(top_k, metric)[
            ['latent_dim', 'lstm_hidden', 'learning_rate', 'batch_size', 'epochs', metric]
        ]
    
    def print_best_configs(self) -> None:
        """Print best configurations for each key metric."""
        if self.df is None:
            self.load_results()
        
        metrics = ['opt_precision', 'opt_recall', 'max_precision', 'auc', 'avg_precision']
        
        print(f"\n🏆 BEST CONFIGURATIONS BY METRIC")
        print(f"{'='*80}")
        
        for metric in metrics:
            best_idx = self.df[metric].idxmax()
            best_row = self.df.loc[best_idx]
            
            print(f"\n📈 {metric.upper()}: {best_row[metric]:.4f}")
            print(f"   latent_dim: {best_row['latent_dim']}")
            print(f"   lstm_hidden: {best_row['lstm_hidden']}")
            print(f"   learning_rate: {best_row['learning_rate']}")
            print(f"   batch_size: {best_row['batch_size']}")
            print(f"   epochs: {best_row['epochs']}")
    
    def print_top_10_overall(self) -> None:
        """Print top 10 configurations by average precision."""
        if self.df is None:
            self.load_results()
        
        top_10 = self.get_top_configs('avg_precision', 10)
        
        print(f"\n🥇 TOP 10 CONFIGURATIONS (by avg_precision)")
        print(f"{'='*100}")
        print(f"{'Rank':<4} {'latent_dim':<10} {'lstm_hidden':<11} {'lr':<8} {'batch':<6} {'epochs':<7} {'avg_prec':<10} {'opt_prec':<10} {'auc':<8}")
        print(f"{'-'*100}")
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:<4} {row['latent_dim']:<10} {row['lstm_hidden']:<11} "
                  f"{row['learning_rate']:<8} {row['batch_size']:<6} {row['epochs']:<7} "
                  f"{row['avg_precision']:<10.4f} {row['opt_precision']:<10.4f} {row['auc']:<8.4f}")
    
    def create_heatmap(self, metric: str = 'avg_precision', save_path: str = None) -> None:
        """
        Create heatmap showing metric performance across hyperparameters.
        
        Parameters
        ----------
        metric : str
            Metric to visualize.
        save_path : str
            Path to save the heatmap.
        """
        if self.df is None:
            self.load_results()
        
        # Create pivot table for latent_dim vs lstm_hidden
        pivot = self.df.pivot_table(
            values=metric, 
            index='latent_dim', 
            columns='lstm_hidden', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis', cbar_kws={'label': metric})
        plt.title(f'{metric.upper()} Heatmap: Latent Dim vs LSTM Hidden')
        plt.xlabel('LSTM Hidden Size')
        plt.ylabel('Latent Dimension')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Heatmap saved to: {save_path}")
        
        plt.show()
    
    def create_learning_rate_analysis(self, save_path: str = None) -> None:
        """
        Analyze the effect of learning rate on performance.
        
        Parameters
        ----------
        save_path : str
            Path to save the analysis plot.
        """
        if self.df is None:
            self.load_results()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['avg_precision', 'opt_precision', 'auc', 'opt_recall']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Group by learning rate and calculate mean/std
            lr_stats = self.df.groupby('learning_rate')[metric].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(lr_stats['learning_rate'], lr_stats['mean'], 
                       yerr=lr_stats['std'], marker='o', capsize=5, capthick=2)
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Learning Rate')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Learning rate analysis saved to: {save_path}")
        
        plt.show()
    
    def create_batch_size_analysis(self, save_path: str = None) -> None:
        """
        Analyze the effect of batch size on performance.
        
        Parameters
        ----------
        save_path : str
            Path to save the analysis plot.
        """
        if self.df is None:
            self.load_results()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['avg_precision', 'opt_precision', 'auc', 'opt_recall']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Group by batch size and calculate mean/std
            batch_stats = self.df.groupby('batch_size')[metric].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(batch_stats['batch_size'], batch_stats['mean'], 
                       yerr=batch_stats['std'], marker='s', capsize=5, capthick=2)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Batch Size')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Batch size analysis saved to: {save_path}")
        
        plt.show()
    
    def create_epochs_analysis(self, save_path: str = None) -> None:
        """
        Analyze the effect of training epochs on performance.
        
        Parameters
        ----------
        save_path : str
            Path to save the analysis plot.
        """
        if self.df is None:
            self.load_results()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['avg_precision', 'opt_precision', 'auc', 'opt_recall']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Group by epochs and calculate mean/std
            epochs_stats = self.df.groupby('epochs')[metric].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(epochs_stats['epochs'], epochs_stats['mean'], 
                       yerr=epochs_stats['std'], marker='^', capsize=5, capthick=2)
            ax.set_xlabel('Training Epochs')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Training Epochs')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Epochs analysis saved to: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self) -> None:
        """Generate a comprehensive analysis report."""
        if self.df is None:
            self.load_results()
        
        print(f"\n📋 COMPREHENSIVE HYPERPARAMETER ANALYSIS REPORT")
        print(f"{'='*80}")
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS")
        print(f"Total experiments: {len(self.df)}")
        print(f"Best avg_precision: {self.df['avg_precision'].max():.4f}")
        print(f"Best opt_precision: {self.df['opt_precision'].max():.4f}")
        print(f"Best auc: {self.df['auc'].max():.4f}")
        
        # Best configurations
        self.print_best_configs()
        
        # Top 10 overall
        self.print_top_10_overall()
        
        # Create visualizations
        print(f"\n📊 GENERATING VISUALIZATIONS...")
        
        # Create analysis directory
        analysis_dir = self.results_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Generate plots
        self.create_heatmap('avg_precision', str(analysis_dir / 'avg_precision_heatmap.png'))
        self.create_learning_rate_analysis(str(analysis_dir / 'learning_rate_analysis.png'))
        self.create_batch_size_analysis(str(analysis_dir / 'batch_size_analysis.png'))
        self.create_epochs_analysis(str(analysis_dir / 'epochs_analysis.png'))
        
        print(f"\n✅ Analysis complete! Check {analysis_dir}/ for visualizations.")

def main():
    """Main function to analyze hyperparameter results."""
    print("📊 Hyperparameter Grid Search Results Analyzer")
    print("=" * 50)
    
    analyzer = HyperparameterAnalyzer()
    
    try:
        analyzer.generate_comprehensive_report()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure to run the grid search first!")

if __name__ == "__main__":
    main()
