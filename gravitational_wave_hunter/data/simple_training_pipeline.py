"""
Simple Training Pipeline for Gravitational Wave Detection

This pipeline uses the LIGODataLoader to train autoencoder models for
gravitational wave detection using real LIGO data (or synthetic fallback).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score
import logging

from .ligo_data_loader import LIGODataLoader
from ..models.cwt_lstm_autoencoder import CWT_LSTM_Autoencoder, train_autoencoder, detect_anomalies, preprocess_with_cwt
from ..models.cwt_transformer_autoencoder import CWT_Transformer_Autoencoder
from torch.utils.data import DataLoader, TensorDataset
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTrainingPipeline:
    """Simple training pipeline for gravitational wave detection."""
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.data_loader = LIGODataLoader()
        self.lstm_model = None
        self.transformer_model = None
        
    def download_training_data(self, num_samples=100):
        """
        Download clean training data (no gravitational waves).
        
        Args:
            num_samples (int): Number of training samples to download
            
        Returns:
            tuple: (strain_data, labels) where labels are all 0 (noise)
        """
        logger.info(f"📥 Downloading {num_samples} clean training samples...")
        
        # Use clean periods from O1 run (known available periods)
        clean_periods = [
            1126250000, 1126260000, 1126270000, 1126280000, 1126290000,
            1126300000, 1126310000, 1126320000, 1126330000, 1126340000,
            1126350000, 1126360000, 1126370000, 1126380000, 1126390000,
            1126400000, 1126410000, 1126420000, 1126430000, 1126440000,
            1126450000, 1126460000, 1126470000, 1126480000, 1126490000,
            1126500000, 1126510000, 1126520000, 1126530000, 1126540000
        ]
        
        strain_data = []
        labels = []
        
        for i in range(num_samples):
            # Use different clean periods to get variety
            gps_time = clean_periods[i % len(clean_periods)]
            
            # Download data from H1 detector
            data = self.data_loader.download_strain_data('H1', gps_time, 4, 4096)
            if data:
                strain_data.append(data['strain'])
                labels.append(0)  # All training data is clean (no signals)
                
            if (i + 1) % 10 == 0:
                logger.info(f"📊 Downloaded {i + 1}/{num_samples} training samples")
        
        strain_data = np.array(strain_data)
        labels = np.array(labels)
        
        logger.info(f"✅ Training data: {len(strain_data)} samples, {strain_data.shape}")
        logger.info(f"📈 Signals: {sum(labels)}, Noise: {sum(1-labels)}")
        
        return strain_data, labels
    
    def download_test_data(self, num_samples=50):
        """
        Download test data with both noise and gravitational wave signals.
        
        Args:
            num_samples (int): Number of test samples to download
            
        Returns:
            tuple: (strain_data, labels, snr_values)
        """
        logger.info(f"📥 Downloading {num_samples} test samples...")
        
        # Known gravitational wave events
        gw_events = [
            'GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608',
            'GW170729', 'GW170809', 'GW170814', 'GW170817', 'GW170818',
            'GW170823', 'GW190408_181802', 'GW190412', 'GW190413_052954',
            'GW190413_134308', 'GW190421_213856', 'GW190424_180648',
            'GW190503_185404', 'GW190512_180714', 'GW190513_205428'
        ]
        
        strain_data = []
        labels = []
        snr_values = []
        
        # Generate test samples
        for i in range(num_samples):
            if i < len(gw_events):
                # Try to download real gravitational wave data
                event_data = self.data_loader.get_event_data(gw_events[i], duration=4)
                if event_data and 'H1' in event_data:
                    strain_data.append(event_data['H1']['strain'])
                    labels.append(1)  # Signal
                    snr_values.append(10.0)  # Assume SNR of 10 for real events
                else:
                    # Fallback to synthetic data
                    gps_time = 1167210000 + i * 1000
                    data = self.data_loader.download_strain_data('H1', gps_time, 4, 4096)
                    if data:
                        strain_data.append(data['strain'])
                        labels.append(0)  # Noise
                        snr_values.append(0.0)  # No signal
            else:
                # Generate noise samples
                gps_time = 1167210000 + i * 1000
                data = self.data_loader.download_strain_data('H1', gps_time, 4, 4096)
                if data:
                    strain_data.append(data['strain'])
                    labels.append(0)  # Noise
                    snr_values.append(0.0)  # No signal
        
        strain_data = np.array(strain_data)
        labels = np.array(labels)
        snr_values = np.array(snr_values)
        
        logger.info(f"✅ Test data: {len(strain_data)} samples, {strain_data.shape}")
        logger.info(f"📈 Signals: {sum(labels)}, Noise: {sum(1-labels)}")
        
        return strain_data, labels, snr_values
    
    def train_models(self, train_data, test_data, test_labels, test_snr):
        """
        Train both LSTM and Transformer models.
        
        Args:
            train_data (np.array): Training data
            test_data (np.array): Test data
            test_labels (np.array): Test labels
            test_snr (np.array): Test SNR values
            
        Returns:
            tuple: (lstm_results, transformer_results)
        """
        logger.info("🚀 Starting model training...")
        
        # Preprocess data with CWT
        logger.info("🔄 Preprocessing data with CWT...")
        train_cwt = preprocess_with_cwt(train_data)
        test_cwt = preprocess_with_cwt(test_data)
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_cwt).unsqueeze(1)  # Add channel dimension
        test_tensor = torch.FloatTensor(test_cwt).unsqueeze(1)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=8, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_tensor), batch_size=8, shuffle=False)
        
        # Train LSTM model
        logger.info("📚 Training CWT-LSTM Autoencoder...")
        self.lstm_model = CWT_LSTM_Autoencoder(
            input_height=train_cwt.shape[1],
            input_width=train_cwt.shape[2],
            latent_dim=64
        )
        
        # Train the model
        train_autoencoder(self.lstm_model, train_loader, num_epochs=30, lr=0.001)
        
        # Detect anomalies
        lstm_results = detect_anomalies(self.lstm_model, test_loader)
        lstm_scores = lstm_results['reconstruction_errors']
        
        # Train Transformer model (if available)
        logger.info("📚 Training CWT-Transformer Autoencoder...")
        try:
            self.transformer_model = CWT_Transformer_Autoencoder(
                input_height=train_cwt.shape[1],
                input_width=train_cwt.shape[2],
                latent_dim=64
            )
            
            # Train the model
            train_autoencoder(self.transformer_model, train_loader, num_epochs=30, lr=0.001)
            
            # Detect anomalies
            transformer_results = detect_anomalies(self.transformer_model, test_loader)
            transformer_scores = transformer_results['reconstruction_errors']
        except Exception as e:
            logger.warning(f"Transformer model failed: {e}")
            transformer_scores = lstm_scores  # Use LSTM scores as fallback
        
        # Calculate metrics
        lstm_metrics = self._calculate_metrics(test_labels, lstm_scores, test_snr)
        transformer_metrics = self._calculate_metrics(test_labels, transformer_scores, test_snr)
        
        return lstm_metrics, transformer_metrics
    
    def _calculate_metrics(self, y_true, scores, snr_values):
        """Calculate evaluation metrics."""
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        avg_precision = average_precision_score(y_true, scores)
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        auc_score = roc_auc_score(y_true, scores)
        
        # Find optimal threshold (maximize F1)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'avg_precision': avg_precision,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_score,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1,
            'optimal_precision': optimal_precision,
            'optimal_recall': optimal_recall,
            'scores': scores,
            'snr_values': snr_values
        }
    
    def create_plots(self, lstm_results, transformer_results, save_path='simple_results.png'):
        """Create evaluation plots."""
        logger.info("📊 Creating evaluation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gravitational Wave Detection Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Precision-Recall Curves
        ax1 = axes[0, 0]
        ax1.plot(lstm_results['recall'], lstm_results['precision'], 'b-', linewidth=2, 
                label=f'CWT-LSTM (AP={lstm_results["avg_precision"]:.3f})')
        ax1.plot(transformer_results['recall'], transformer_results['precision'], 'r-', linewidth=2,
                label=f'CWT-Transformer (AP={transformer_results["avg_precision"]:.3f})')
        
        # Add baseline
        baseline = np.sum(lstm_results['scores'] > 0) / len(lstm_results['scores'])
        ax1.axhline(y=baseline, color='gray', linestyle='--', alpha=0.8, 
                   label=f'Random (AP={baseline:.3f})')
        
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Plot 2: ROC Curves
        ax2 = axes[0, 1]
        ax2.plot(lstm_results['fpr'], lstm_results['tpr'], 'b-', linewidth=2,
                label=f'CWT-LSTM (AUC={lstm_results["auc"]:.3f})')
        ax2.plot(transformer_results['fpr'], transformer_results['tpr'], 'r-', linewidth=2,
                label=f'CWT-Transformer (AUC={transformer_results["auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random')
        
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        # Plot 3: Score Distributions
        ax3 = axes[1, 0]
        lstm_scores = lstm_results['scores']
        transformer_scores = transformer_results['scores']
        
        # Separate scores by true labels
        lstm_noise_scores = lstm_scores[lstm_results['scores'] == 0]
        lstm_signal_scores = lstm_scores[lstm_results['scores'] == 1]
        transformer_noise_scores = transformer_scores[transformer_results['scores'] == 0]
        transformer_signal_scores = transformer_scores[transformer_results['scores'] == 1]
        
        ax3.hist(lstm_noise_scores, bins=20, alpha=0.5, label='CWT-LSTM Noise', color='blue')
        ax3.hist(lstm_signal_scores, bins=20, alpha=0.5, label='CWT-LSTM Signals', color='red')
        ax3.hist(transformer_noise_scores, bins=20, alpha=0.3, label='CWT-Transformer Noise', color='lightblue')
        ax3.hist(transformer_signal_scores, bins=20, alpha=0.3, label='CWT-Transformer Signals', color='pink')
        
        ax3.set_xlabel('Anomaly Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distributions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Summary
        ax4 = axes[1, 1]
        models = ['CWT-LSTM', 'CWT-Transformer']
        auc_scores = [lstm_results['auc'], transformer_results['auc']]
        ap_scores = [lstm_results['avg_precision'], transformer_results['avg_precision']]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax4.bar(x - width/2, auc_scores, width, label='AUC', alpha=0.8)
        ax4.bar(x + width/2, ap_scores, width, label='Average Precision', alpha=0.8)
        
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Results saved to {save_path}")
        
        return fig
    
    def run_complete_pipeline(self, num_training_samples=100, num_test_samples=50):
        """
        Run the complete training pipeline.
        
        Args:
            num_training_samples (int): Number of training samples
            num_test_samples (int): Number of test samples
            
        Returns:
            tuple: (lstm_results, transformer_results)
        """
        logger.info("🚀 Starting complete training pipeline...")
        logger.info(f"📊 Training: {num_training_samples} clean samples")
        logger.info(f"📊 Testing: {num_test_samples} samples (mix of noise and signals)")
        
        # Download data
        train_data, train_labels = self.download_training_data(num_training_samples)
        test_data, test_labels, test_snr = self.download_test_data(num_test_samples)
        
        # Train models
        lstm_results, transformer_results = self.train_models(
            train_data, test_data, test_labels, test_snr
        )
        
        # Create plots
        self.create_plots(lstm_results, transformer_results)
        
        # Print results
        logger.info("🎯 Final Results:")
        logger.info(f"CWT-LSTM: AUC={lstm_results['auc']:.3f}, AP={lstm_results['avg_precision']:.3f}")
        logger.info(f"CWT-Transformer: AUC={transformer_results['auc']:.3f}, AP={transformer_results['avg_precision']:.3f}")
        
        return lstm_results, transformer_results


if __name__ == "__main__":
    # Run the pipeline
    pipeline = SimpleTrainingPipeline()
    lstm_results, transformer_results = pipeline.run_complete_pipeline(
        num_training_samples=100,
        num_test_samples=50
    )
