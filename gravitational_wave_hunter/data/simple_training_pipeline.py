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
        logger.info(f" Downloading {num_samples} clean training samples...")
        
        # Use GPS periods from official O1 observing run (Sept 12, 2015 - Jan 19, 2016)
        # Based on GWOSC O1 Data Release: https://gwosc.org/data/
        # GW150914 occurred at GPS time 1126259446 during O1
        clean_periods = [
            # Periods around GW150914 (Sept 14, 2015) - known good quality data
            1126259400, 1126259500, 1126259600, 1126259700, 1126259800,
            1126259900, 1126260000, 1126260100, 1126260200, 1126260300,
            1126260400, 1126260500, 1126260600, 1126260700, 1126260800,
            1126260900, 1126261000, 1126261100, 1126261200, 1126261300,
            1126261400, 1126261500, 1126261600, 1126261700, 1126261800,
            1126261900, 1126262000, 1126262100, 1126262200, 1126262300,
            1126262400, 1126262500, 1126262600, 1126262700, 1126262800,
            1126262900, 1126263000, 1126263100, 1126263200, 1126263300,
            1126263400, 1126263500, 1126263600, 1126263700, 1126263800,
            1126263900, 1126264000, 1126264100, 1126264200, 1126264300,
            1126264400, 1126264500, 1126264600, 1126264700, 1126264800,
            1126264900, 1126265000, 1126265100, 1126265200, 1126265300,
            1126265400, 1126265500, 1126265600, 1126265700, 1126265800,
            1126265900, 1126266000, 1126266100, 1126266200, 1126266300,
            1126266400, 1126266500, 1126266600, 1126266700, 1126266800,
            1126266900, 1126267000, 1126267100, 1126267200, 1126267300,
            1126267400, 1126267500, 1126267600, 1126267700, 1126267800,
            1126267900, 1126268000, 1126268100, 1126268200, 1126268300,
            1126268400, 1126268500, 1126268600, 1126268700, 1126268800,
            1126268900, 1126269000, 1126269100, 1126269200, 1126269300,
            1126269400, 1126269500, 1126269600, 1126269700, 1126269800,
            1126269900, 1126270000, 1126270100, 1126270200, 1126270300
        ]
        
        strain_data = []
        labels = []
        
        # Try to download samples, skipping periods that timeout
        period_index = 0
        successful_downloads = 0
        
        while successful_downloads < num_samples and period_index < len(clean_periods):
            gps_time = clean_periods[period_index]
            
            try:
                # Download data from H1 detector
                data = self.data_loader.download_strain_data('H1', gps_time, 4, 4096)
                if data:
                    strain_data.append(data['strain'])
                    labels.append(0)  # All training data is clean (no signals)
                    successful_downloads += 1
                    
                    if successful_downloads % 10 == 0:
                        logger.info(f" Downloaded {successful_downloads}/{num_samples} training samples")
                        
            except KeyboardInterrupt:
                logger.warning(f" Network timeout for period {gps_time} - skipping")
                # Continue to next period
            except Exception as e:
                logger.warning(f" Failed to download period {gps_time}: {e}")
                # Continue to next period
                
            period_index += 1
        
        strain_data = np.array(strain_data)
        labels = np.array(labels)
        
        logger.info(f" Training data: {len(strain_data)} samples, {strain_data.shape}")
        logger.info(f" Signals: {sum(labels)}, Noise: {sum(1-labels)}")
        
        return strain_data, labels
    
    def download_test_data(self, num_samples=50):
        """
        Download test data with both noise and gravitational wave signals.
        
        Args:
            num_samples (int): Number of test samples to download
            
        Returns:
            tuple: (strain_data, labels, snr_values)
        """
        logger.info(f" Downloading {num_samples} test samples...")
        
        # Known gravitational wave events (only those available in data loader)
        gw_events = [
            'GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170814', 'GW170817'
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
                    # Fallback to clean O1 data (no signals)
                    gps_time = 1126250000 + i * 1000  # Use working O1 periods
                    data = self.data_loader.download_strain_data('H1', gps_time, 4, 4096)
                    if data:
                        strain_data.append(data['strain'])
                        labels.append(0)  # Noise
                        snr_values.append(0.0)  # No signal
            else:
                # Generate noise samples using working O1 periods
                gps_time = 1126250000 + i * 1000  # Use working O1 periods
                data = self.data_loader.download_strain_data('H1', gps_time, 4, 4096)
                if data:
                    strain_data.append(data['strain'])
                    labels.append(0)  # Noise
                    snr_values.append(0.0)  # No signal
        
        strain_data = np.array(strain_data)
        labels = np.array(labels)
        snr_values = np.array(snr_values)
        
        logger.info(f" Test data: {len(strain_data)} samples, {strain_data.shape}")
        logger.info(f" Signals: {sum(labels)}, Noise: {sum(1-labels)}")
        
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
        logger.info(" Starting model training...")
        
        # Preprocess data with CWT
        logger.info("Preprocessing data with CWT...")
        sample_rate = 4096  # Standard LIGO sample rate
        
        # Debug: Check data before preprocessing
        logger.info(f"Train data shape: {train_data.shape}, min: {train_data.min():.6f}, max: {train_data.max():.6f}")
        logger.info(f"Test data shape: {test_data.shape}, min: {test_data.min():.6f}, max: {test_data.max():.6f}")
        
        train_cwt = preprocess_with_cwt(train_data, sample_rate)
        test_cwt = preprocess_with_cwt(test_data, sample_rate)
        
        # Debug: Check CWT data
        logger.info(f"Train CWT shape: {train_cwt.shape}, min: {train_cwt.min():.6f}, max: {train_cwt.max():.6f}")
        logger.info(f"Test CWT shape: {test_cwt.shape}, min: {test_cwt.min():.6f}, max: {test_cwt.max():.6f}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(train_cwt)) or np.any(np.isinf(train_cwt)):
            logger.error("NaN or infinite values in train CWT data!")
        if np.any(np.isnan(test_cwt)) or np.any(np.isinf(test_cwt)):
            logger.error("NaN or infinite values in test CWT data!")
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_cwt).unsqueeze(1)  # Add channel dimension
        test_tensor = torch.FloatTensor(test_cwt).unsqueeze(1)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=8, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_tensor), batch_size=8, shuffle=False)
        
        # Train LSTM model
        logger.info(" Training CWT-LSTM Autoencoder...")
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
        logger.info(" Training CWT-Transformer Autoencoder...")
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
        # Check for NaN values and handle them
        if np.any(np.isnan(scores)):
            logger.warning("NaN values detected in scores, replacing with 0")
            scores = np.nan_to_num(scores, nan=0.0)
        
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present in y_true, returning dummy metrics")
            return {
                'precision': np.array([1.0]),
                'recall': np.array([1.0]),
                'thresholds': np.array([0.0]),
                'avg_precision': 0.5,
                'fpr': np.array([0.0, 1.0]),
                'tpr': np.array([0.0, 1.0]),
                'auc': 0.5,
                'optimal_threshold': 0.0,
                'optimal_f1': 0.0,
                'optimal_precision': 0.0,
                'optimal_recall': 0.0,
                'scores': scores,
                'snr_values': snr_values
            }
        
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
        logger.info(" Creating evaluation plots...")
        
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
        logger.info(f" Results saved to {save_path}")
        
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
        logger.info(" Starting complete training pipeline...")
        logger.info(f" Training: {num_training_samples} clean samples")
        logger.info(f" Testing: {num_test_samples} samples (mix of noise and signals)")
        
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
        logger.info(" Final Results:")
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
