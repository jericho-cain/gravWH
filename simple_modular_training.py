#!/usr/bin/env python3
"""
Simple Modular Training Pipeline

This is a standalone version that avoids import issues.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score
import logging
import gc
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import decimate

# Add the project root to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gravitational_wave_hunter', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'gravitational_wave_hunter', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from ligo_data_loader import LIGODataLoader
from cwt_lstm_autoencoder import CWT_LSTM_Autoencoder, train_autoencoder, detect_anomalies, preprocess_with_cwt
from track_run import RunTracker

def purge_memory():
    """Purge memory to prevent accumulation between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    plt.clf()
    plt.close('all')
    gc.collect()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleModularTraining:
    """Simple modular training that uses pre-downloaded data."""
    
    def __init__(self, random_seed=42):
        """Initialize the training pipeline."""
        # Set random seed for reproducible results
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.data_loader = LIGODataLoader()
        self.lstm_model = None
        self.tracker = RunTracker()
        self.random_seed = random_seed
        
    def download_training_data(self, num_samples=100):
        """Download clean training data from cache."""
        logger.info(f"📥 Downloading {num_samples} training samples from cache...")
        
        # Get list of available cached GPS times
        available_times = self._get_available_gps_times()
        logger.info(f"📊 Found {len(available_times)} cached GPS times")
        
        if len(available_times) < num_samples:
            logger.warning(f"⚠️  Only {len(available_times)} cached samples available, reducing request from {num_samples}")
            num_samples = len(available_times)
        
        # Randomly sample from available times
        selected_times = random.sample(available_times, num_samples)
        
        strain_data = []
        
        for gps_time in selected_times:
            data = self.data_loader.download_strain_data('H1', gps_time, 4, 4096)
            if data:
                strain_data.append(data['strain'])
        
        strain_data = np.array(strain_data)
        logger.info(f"✅ Training data: {len(strain_data)} samples, {strain_data.shape}")
        
        return strain_data
    
    def load_test_data(self, num_samples=25):
        """Load test data from pre-downloaded GW events and cached noise."""
        logger.info(f"📥 Loading {num_samples} test samples from cache...")
        
        strain_data = []
        labels = []
        snr_values = []
        
        # Load pre-downloaded GW events
        gw_events_dir = "gw_events_cache"
        gw_files = []
        
        if os.path.exists(gw_events_dir):
            gw_files = [f for f in os.listdir(gw_events_dir) if f.endswith('_H1.npz')]
            logger.info(f"📊 Found {len(gw_files)} pre-downloaded GW events")
        
        # Load GW events (up to 50% of test samples)
        max_gw_events = min(len(gw_files), num_samples // 2)
        
        for i in range(max_gw_events):
            try:
                gw_file = gw_files[i]
                gw_path = os.path.join(gw_events_dir, gw_file)
                
                # Load GW event data
                gw_data = np.load(gw_path)
                strain_data.append(gw_data['strain'])
                labels.append(1)  # Signal
                snr_values.append(10.0)  # Assume SNR of 10 for real events
                
                logger.info(f"✅ Loaded GW event: {gw_data['event_name']}")
                
            except Exception as e:
                logger.warning(f"❌ Failed to load GW event {gw_file}: {e}")
                continue
        
        # Fill remaining with cached noise data
        remaining_samples = num_samples - len(strain_data)
        if remaining_samples > 0:
            logger.info(f"📊 Loading {remaining_samples} noise samples from cache...")
            
            available_times = self._get_available_gps_times()
            selected_times = random.sample(available_times, remaining_samples)
            
            for gps_time in selected_times:
                data = self.data_loader.download_strain_data('H1', gps_time, 4, 4096)
                if data:
                    strain_data.append(data['strain'])
                    labels.append(0)  # Noise
                    snr_values.append(0.0)  # No signal
        
        strain_data = np.array(strain_data)
        labels = np.array(labels)
        snr_values = np.array(snr_values)
        
        logger.info(f"✅ Test data: {len(strain_data)} samples, {strain_data.shape}")
        logger.info(f"📊 Signals: {sum(labels)}, Noise: {sum(1-labels)}")
        
        return strain_data, labels, snr_values
    
    def _get_available_gps_times(self):
        """Get list of available GPS times from cache directory."""
        import re
        
        cache_dir = "ligo_data_cache"
        available_times = []
        
        if not os.path.exists(cache_dir):
            logger.warning("Cache directory not found")
            return available_times
        
        # Scan cache directory for H1 files
        for filename in os.listdir(cache_dir):
            if filename.startswith("O1_H1_") and filename.endswith("_4_4096.npz"):
                # Extract GPS time from filename: O1_H1_1126256000_4_4096.npz
                match = re.search(r'O1_H1_(\d+)_4_4096\.npz', filename)
                if match:
                    gps_time = int(match.group(1))
                    available_times.append(gps_time)
        
        available_times.sort()
        return available_times
    
    def train_model(self, train_data, test_data, test_labels, test_snr):
        """Train the LSTM model."""
        logger.info("🚀 Starting model training...")
        
        # Capture hyperparameters for tracking (exact working configuration)
        hyperparameters = {
            "training_samples": len(train_data),
            "test_samples": len(test_data),
            "cwt_height": 8,
            "cwt_width": 4096,
            "optimizer": "SGD",
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 1e-5,
            "epochs": 50,
            "batch_size": 1,
            "latent_dim": 32,
            "lstm_hidden": 64,
            "mixed_precision": True,
            "memory_cleanup": True
        }
        
        # Downsample before CWT (exact working config)
        from scipy.signal import decimate
        
        decim = 4  # 4096 -> 1024 Hz (exact working config)
        logger.info(f"📊 Downsampling data from {4096} Hz to {4096//decim} Hz...")
        
        train_data_ds = np.array([decimate(x, decim, zero_phase=True).astype(np.float32) for x in train_data])
        test_data_ds = np.array([decimate(x, decim, zero_phase=True).astype(np.float32) for x in test_data])
        sample_rate_ds = 4096 // decim
        
        # Preprocess with CWT using downsampled data (exact working config)
        logger.info("🔄 Computing CWT transforms...")
        train_cwt = preprocess_with_cwt(train_data_ds, sample_rate_ds, target_height=8)
        test_cwt = preprocess_with_cwt(test_data_ds, sample_rate_ds, target_height=8)
        
        logger.info(f"✅ CWT shapes: train={train_cwt.shape}, test={test_cwt.shape}")
        
        # Store dimensions before clearing arrays (exact working config)
        input_height = train_cwt.shape[1]
        input_width = train_cwt.shape[2]
        
        # Update hyperparameters with actual sample counts
        actual_train_samples = len(train_cwt)
        actual_test_samples = len(test_cwt)
        hyperparameters.update({
            "actual_train_samples": actual_train_samples,
            "actual_test_samples": actual_test_samples,
            "cwt_width": input_width
        })
        
        # Convert to tensors and add channel dimension
        # CWT data is (samples, height, width), model expects (samples, channels, height, width)
        train_tensor = torch.FloatTensor(train_cwt).unsqueeze(1)  # Add channel dimension
        test_tensor = torch.FloatTensor(test_cwt).unsqueeze(1)    # Add channel dimension
        
        # Create data loaders
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        
        # Clean up training CWT data to save memory (keep test_cwt for evaluation)
        del train_cwt
        purge_memory()
        
        # Train LSTM model (exact working configuration)
        logger.info("🧠 Training CWT-LSTM Autoencoder...")
        self.lstm_model = CWT_LSTM_Autoencoder(
            input_height=input_height,
            input_width=input_width,
            latent_dim=32,
            lstm_hidden=64
        )
        
        # Dynamic epoch strategy based on sample count
        if actual_train_samples >= 90:
            num_epochs = 20
        else:
            num_epochs = 50
        
        logger.info(f"📊 Using {num_epochs} epochs for {actual_train_samples} samples")
        
        lstm_results = train_autoencoder(
            self.lstm_model, 
            train_loader, 
            num_epochs=num_epochs,
            lr=0.001,
            val_loader=None  # No validation to save memory
        )
        
        # Clean up training data
        del train_tensor, test_tensor, train_loader
        purge_memory()
        
        # Evaluate on test data
        logger.info("🔍 Evaluating model on test data...")
        test_tensor = torch.FloatTensor(test_cwt).unsqueeze(1)  # Add channel dimension
        test_loader = DataLoader(TensorDataset(test_tensor), batch_size=1, shuffle=False)
        
        lstm_scores = detect_anomalies(self.lstm_model, test_loader)
        
        # Trim test_labels to match actual test samples (some may have been filtered during CWT)
        if len(lstm_scores) != len(test_labels):
            logger.warning(f"Test data mismatch: {len(lstm_scores)} scores vs {len(test_labels)} labels")
            logger.warning("Truncating labels to match available samples")
            test_labels = test_labels[:len(lstm_scores)]
            test_snr = test_snr[:len(lstm_scores)]
        
        # Calculate metrics
        lstm_results = self._calculate_metrics(lstm_scores, test_labels)
        
        # Clean up test data
        del test_tensor, test_loader
        purge_memory()
        
        # Log successful run
        self.tracker.log_successful_run(hyperparameters, lstm_results)
        
        logger.info("✅ Model training completed successfully!")
        
        return lstm_results
    
    def _calculate_metrics(self, scores, test_labels):
        """Calculate evaluation metrics."""
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(test_labels, scores)
        avg_precision = average_precision_score(test_labels, scores)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(test_labels, scores)
        auc = roc_auc_score(test_labels, scores)
        
        return {
            'auc': auc,
            'avg_precision': avg_precision,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'scores': scores
        }
    
    def run_training(self, num_training_samples=100, num_test_samples=25):
        """Run the complete training pipeline."""
        try:
            logger.info(f"🚀 Starting simple modular training...")
            logger.info(f"📊 Training samples: {num_training_samples}, Test samples: {num_test_samples}")
            
            # Download training data
            train_data = self.download_training_data(num_training_samples)
            
            # Load test data
            test_data, test_labels, test_snr = self.load_test_data(num_test_samples)
            
            # Train model
            lstm_results = self.train_model(train_data, test_data, test_labels, test_snr)
            
            logger.info("✅ Simple modular training completed successfully!")
            return lstm_results
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            self.tracker.log_failed_run({"error": str(e)}, str(e))
            raise

if __name__ == "__main__":
    # Example usage
    trainer = SimpleModularTraining(random_seed=7)
    results = trainer.run_training(
        num_training_samples=100, 
        num_test_samples=25
    )
    
    print(f"🎉 Training completed!")
    print(f"📊 Results: AUC={results['auc']:.3f}, AP={results['avg_precision']:.3f}")
