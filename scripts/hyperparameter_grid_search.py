#!/usr/bin/env python3
"""
Hyperparameter Grid Search for CWT-LSTM Autoencoder
Systematically tests different hyperparameter combinations and tracks performance metrics.
"""

import os
import json
import time
import itertools
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import our model and functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gravitational_wave_hunter.models.cwt_lstm_autoencoder import (
    CWT_LSTM_Autoencoder, 
    generate_realistic_chirp, 
    generate_colored_noise, 
    preprocess_with_cwt,
    train_autoencoder,
    detect_anomalies
)
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score

class HyperparameterGridSearch:
    """
    Comprehensive hyperparameter grid search for CWT-LSTM Autoencoder.
    
    Tests different combinations of hyperparameters and tracks all key metrics
    to identify optimal configuration for gravitational wave detection.
    """
    
    def __init__(self, results_dir: str = "hyperparameter_results"):
        """
        Initialize grid search with results directory.
        
        Parameters
        ----------
        results_dir : str
            Directory to save all results, figures, and metrics.
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/figures", exist_ok=True)
        os.makedirs(f"{results_dir}/models", exist_ok=True)
        
        # Define hyperparameter grid
        self.hyperparameter_grid = {
            'latent_dim': [8, 16, 32, 64],
            'lstm_hidden': [16, 32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [4, 8, 16],
            'epochs': [20, 30, 50]
        }
        
        # Results storage
        self.results = []
        self.best_config = None
        self.best_score = 0
        
        print(f"🔍 Hyperparameter Grid Search Initialized")
        print(f"📊 Testing {self._get_total_combinations()} combinations")
        print(f"💾 Results will be saved to: {results_dir}")
    
    def _get_total_combinations(self) -> int:
        """Calculate total number of hyperparameter combinations."""
        total = 1
        for values in self.hyperparameter_grid.values():
            total *= len(values)
        return total
    
    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible hyperparameter combinations."""
        keys = list(self.hyperparameter_grid.keys())
        values = list(self.hyperparameter_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _generate_data(self, num_samples: int = 200, signal_prob: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic gravitational wave data for testing.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        signal_prob : float
            Probability of each sample containing a signal.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            CWT data, labels, and SNR values.
        """
        print(f"🌊 Generating {num_samples} samples for hyperparameter testing...")
        
        SAMPLE_RATE = 512
        DURATION = 4
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
        
        strain_data = []
        labels = []
        snr_values = []
        
        for i in range(num_samples):
            # Generate colored noise
            noise = generate_colored_noise(len(t), SAMPLE_RATE, seed=42+i)
            
            if np.random.random() < signal_prob:
                # Random parameters for diversity
                m1 = np.random.uniform(25, 50)
                m2 = np.random.uniform(25, 50)
                distance = np.random.uniform(300, 800)
                
                # Generate GW signal
                gw_signal = generate_realistic_chirp(t, m1, m2, distance)
                
                # Calculate SNR and scale
                signal_power = np.std(gw_signal)
                noise_power = np.std(noise)
                target_snr = np.random.uniform(8, 20)
                
                if signal_power > 0:
                    scaling = target_snr * noise_power / signal_power
                    gw_signal = gw_signal * scaling
                
                combined = noise + gw_signal
                label = 1
                snr_values.append(target_snr)
            else:
                combined = noise
                label = 0
                snr_values.append(0)
            
            strain_data.append(combined)
            labels.append(label)
        
        # Convert to arrays
        strain_data = np.array(strain_data)
        labels = np.array(labels)
        snr_values = np.array(snr_values)
        
        # Compute CWT representations
        print(f"🌊 Computing CWT for {len(strain_data)} samples...")
        cwt_data = preprocess_with_cwt(strain_data, SAMPLE_RATE, target_height=32)
        
        return cwt_data, labels, snr_values
    
    def _calculate_metrics(self, labels: np.ndarray, reconstruction_errors: np.ndarray) -> Dict[str, float]:
        """
        Calculate all key performance metrics.
        
        Parameters
        ----------
        labels : np.ndarray
            True labels (0=noise, 1=signal).
        reconstruction_errors : np.ndarray
            Reconstruction errors for each sample.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing all 8 key metrics.
        """
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(labels, reconstruction_errors)
        avg_precision = auc(recall, precision)
        
        # Find key operating points
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        
        # Find highest precision point
        max_precision_idx = np.argmax(precision)
        max_precision = precision[max_precision_idx]
        max_precision_recall = recall[max_precision_idx]
        
        # Find highest precision >= 90%
        precision_90_plus = precision >= 0.90
        if np.any(precision_90_plus):
            valid_indices = np.where(precision_90_plus)[0]
            best_90_idx = valid_indices[np.argmax(recall[valid_indices])]
            precision_90 = precision[best_90_idx]
            recall_90 = recall[best_90_idx]
        else:
            precision_90 = precision[optimal_idx]
            recall_90 = recall[optimal_idx]
        
        # Calculate AUC-ROC
        try:
            auc_score = roc_auc_score(labels, reconstruction_errors)
        except ValueError:
            auc_score = 0.5
        
        return {
            'opt_precision': float(precision_90),
            'opt_recall': float(recall_90),
            'max_precision': float(max_precision),
            'max_recall': float(max_precision_recall),
            'f1_precision': float(precision[optimal_idx]),
            'f1_recall': float(recall[optimal_idx]),
            'auc': float(auc_score),
            'avg_precision': float(avg_precision)
        }
    
    def _run_single_experiment(self, config: Dict[str, Any], cwt_data: np.ndarray, 
                             labels: np.ndarray, snr_values: np.ndarray) -> Dict[str, Any]:
        """
        Run a single experiment with given hyperparameters.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Hyperparameter configuration.
        cwt_data : np.ndarray
            CWT data for training and testing.
        labels : np.ndarray
            True labels.
        snr_values : np.ndarray
            SNR values for each sample.
            
        Returns
        -------
        Dict[str, Any]
            Results including metrics and configuration.
        """
        print(f"🧪 Testing config: {config}")
        
        # Split data: Train autoencoder ONLY on noise
        noise_indices = np.where(labels == 0)[0]
        noise_cwt = cwt_data[noise_indices]
        
        # Create datasets
        noise_dataset = TensorDataset(torch.FloatTensor(noise_cwt).unsqueeze(1))
        test_dataset = TensorDataset(torch.FloatTensor(cwt_data).unsqueeze(1))
        
        noise_loader = DataLoader(noise_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Create model
        height, width = cwt_data.shape[1], cwt_data.shape[2]
        model = CWT_LSTM_Autoencoder(
            input_height=height, 
            input_width=width, 
            latent_dim=config['latent_dim'], 
            lstm_hidden=config['lstm_hidden']
        )
        
        # Train model
        train_losses = train_autoencoder(
            model, 
            noise_loader, 
            num_epochs=config['epochs'], 
            lr=config['learning_rate']
        )
        
        # Detect anomalies
        results = detect_anomalies(model, test_loader, noise_threshold_percentile=90)
        
        # Calculate metrics
        metrics = self._calculate_metrics(labels, results['reconstruction_errors'])
        
        # Combine results
        experiment_result = {
            'config': config.copy(),
            'metrics': metrics,
            'train_losses': train_losses,
            'reconstruction_errors': results['reconstruction_errors'].tolist(),
            'threshold': float(results['threshold']),
            'timestamp': datetime.now().isoformat()
        }
        
        return experiment_result
    
    def run_grid_search(self, num_samples: int = 200) -> None:
        """
        Run the complete hyperparameter grid search.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate for testing.
        """
        print(f"🚀 Starting Hyperparameter Grid Search")
        print(f"📊 Total combinations: {self._get_total_combinations()}")
        
        # Generate data once for all experiments
        cwt_data, labels, snr_values = self._generate_data(num_samples)
        
        # Get all combinations
        combinations = self._generate_combinations()
        
        # Run experiments
        start_time = time.time()
        
        for i, config in enumerate(combinations):
            print(f"\n{'='*60}")
            print(f"🧪 Experiment {i+1}/{len(combinations)}")
            print(f"⏱️  Elapsed: {time.time() - start_time:.1f}s")
            
            try:
                result = self._run_single_experiment(config, cwt_data, labels, snr_values)
                self.results.append(result)
                
                # Track best result (using avg_precision as primary metric)
                if result['metrics']['avg_precision'] > self.best_score:
                    self.best_score = result['metrics']['avg_precision']
                    self.best_config = config.copy()
                    print(f"🏆 NEW BEST! Avg Precision: {self.best_score:.4f}")
                
                # Save intermediate results
                if (i + 1) % 10 == 0:
                    self._save_results()
                    
            except Exception as e:
                print(f"❌ Error in experiment {i+1}: {e}")
                continue
        
        # Final save
        self._save_results()
        
        print(f"\n🎉 Grid Search Complete!")
        print(f"⏱️  Total time: {time.time() - start_time:.1f}s")
        print(f"🏆 Best config: {self.best_config}")
        print(f"📈 Best avg_precision: {self.best_score:.4f}")
    
    def _save_results(self) -> None:
        """Save results to JSON and CSV files."""
        # Save detailed results as JSON
        with open(f"{self.results_dir}/detailed_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary table
        summary_data = []
        for result in self.results:
            row = result['config'].copy()
            row.update(result['metrics'])
            summary_data.append(row)
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{self.results_dir}/results_summary.csv", index=False)
        
        # Save best configuration
        if self.best_config:
            with open(f"{self.results_dir}/best_config.json", 'w') as f:
                json.dump({
                    'config': self.best_config,
                    'score': self.best_score,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        
        print(f"💾 Results saved to {self.results_dir}/")
    
    def analyze_results(self) -> None:
        """Analyze and display results."""
        if not self.results:
            print("❌ No results to analyze. Run grid search first.")
            return
        
        df = pd.DataFrame([{**r['config'], **r['metrics']} for r in self.results])
        
        print(f"\n📊 RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        # Best configurations for each metric
        metrics = ['opt_precision', 'opt_recall', 'max_precision', 'auc', 'avg_precision']
        
        for metric in metrics:
            best_idx = df[metric].idxmax()
            best_config = df.loc[best_idx]
            print(f"\n🏆 Best {metric}: {best_config[metric]:.4f}")
            print(f"   Config: latent_dim={best_config['latent_dim']}, "
                  f"lstm_hidden={best_config['lstm_hidden']}, "
                  f"lr={best_config['learning_rate']}, "
                  f"batch_size={best_config['batch_size']}, "
                  f"epochs={best_config['epochs']}")
        
        # Overall best
        if self.best_config:
            print(f"\n🥇 OVERALL BEST (avg_precision): {self.best_score:.4f}")
            print(f"   Config: {self.best_config}")

def main():
    """Main function to run hyperparameter grid search."""
    print("🔍 CWT-LSTM Autoencoder Hyperparameter Grid Search")
    print("=" * 60)
    
    # Initialize grid search
    grid_search = HyperparameterGridSearch()
    
    # Run grid search
    grid_search.run_grid_search(num_samples=200)
    
    # Analyze results
    grid_search.analyze_results()
    
    print(f"\n✅ Grid search complete! Check {grid_search.results_dir}/ for results.")

if __name__ == "__main__":
    main()
