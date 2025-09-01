#!/usr/bin/env python3
"""
View Precision-Recall Plots for CWT-LSTM Autoencoder
Creates and displays precision-recall analysis with publication-quality plots
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
import pywt
from sklearn.metrics import (precision_recall_curve, average_precision_score, 
                           roc_curve, roc_auc_score, classification_report,
                           precision_score, recall_score, f1_score)
import warnings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
device = torch.device('cpu')

# Configure matplotlib to keep plots open
plt.ion()  # Turn on interactive mode

def generate_realistic_chirp(t, m1=30, m2=30, distance=400):
    """Generate realistic gravitational wave chirp"""
    M_total = m1 + m2
    M_chirp = (m1 * m2)**(3/5) / (M_total)**(1/5)
    
    tc = t[-1]
    tau = tc - t
    tau[tau <= 0] = 1e-10
    
    f_0 = 35
    f = f_0 * (tau / tau[0])**(-3/8)
    f = np.clip(f, f_0, 512)
    
    phi = 2 * np.pi * np.cumsum(f) * (t[1] - t[0])
    
    amplitude = 1e-21 * (M_chirp / 30)**(5/6) * (400 / distance)
    envelope = np.sqrt(f / f_0) * np.exp(-((t - tc) / (tc/8))**2)
    
    h_plus = amplitude * envelope * np.cos(phi)
    h_cross = amplitude * envelope * np.sin(phi)
    
    return 0.5 * (h_plus + h_cross)

def generate_colored_noise(length, sample_rate, seed=None):
    """Generate realistic LIGO-like colored noise"""
    if seed is not None:
        np.random.seed(seed)
    
    freqs = np.fft.fftfreq(length, 1/sample_rate)
    freqs = freqs[:length//2 + 1]
    f = np.abs(freqs)
    f[0] = 1e-10
    
    psd = np.zeros_like(f)
    
    low_mask = f < 10
    psd[low_mask] = 1e-46 * (f[low_mask] / 10)**(-4)
    
    mid_mask = (f >= 10) & (f < 100)
    psd[mid_mask] = 1e-48 * (1 + (f[mid_mask] / 50)**2)
    
    high_mask = f >= 100
    psd[high_mask] = 1e-48 * (f[high_mask] / 100)**(1.5)
    
    white_noise = np.random.normal(0, 1, length//2 + 1) + 1j * np.random.normal(0, 1, length//2 + 1)
    white_noise[0] = white_noise[0].real
    if length % 2 == 0:
        white_noise[-1] = white_noise[-1].real
    
    colored_noise_fft = white_noise * np.sqrt(psd * sample_rate / 2)
    noise = np.fft.irfft(colored_noise_fft, n=length)
    
    return noise

def continuous_wavelet_transform(signal, sample_rate, scales=None):
    """Compute CWT using Morlet wavelets"""
    if scales is None:
        freqs = np.logspace(np.log10(20), np.log10(512), 64)
        scales = sample_rate / freqs
    
    coefficients, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=1/sample_rate)
    scalogram = np.abs(coefficients)
    
    return scalogram, frequencies

def preprocess_with_cwt(strain_data, sample_rate, target_height=64):
    """Preprocess strain data using CWT"""
    cwt_data = []
    
    for i, strain in enumerate(strain_data):
        sos = signal.butter(4, 20, btype='highpass', fs=sample_rate, output='sos')
        filtered = signal.sosfilt(sos, strain)
        whitened = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
        
        scalogram, freqs = continuous_wavelet_transform(whitened, sample_rate)
        
        if scalogram.shape[0] != target_height:
            from scipy.ndimage import zoom
            zoom_factor = target_height / scalogram.shape[0]
            scalogram = zoom(scalogram, (zoom_factor, 1), order=1)
        
        log_scalogram = np.log10(scalogram + 1e-10)
        normalized = (log_scalogram - np.mean(log_scalogram)) / (np.std(log_scalogram) + 1e-10)
        
        cwt_data.append(normalized)
    
    return np.array(cwt_data)

class SimpleCWTAutoencoder(nn.Module):
    """Simplified CWT Autoencoder"""
    def __init__(self, height, width, latent_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
        self.height = height
        self.width = width
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        if reconstructed.shape[-2:] != (self.height, self.width):
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=(self.height, self.width), mode='bilinear', align_corners=False
            )
        
        return reconstructed, latent

def train_autoencoder_silent(model, noise_loader, num_epochs=30, lr=0.001):
    """Train autoencoder silently"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model = model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        for batch in noise_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed, latent = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

def get_reconstruction_errors(model, test_loader):
    """Get reconstruction errors for all test samples"""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].to(device)
            reconstructed, latent = model(data)
            mse = torch.mean((reconstructed - data)**2, dim=(1, 2, 3))
            reconstruction_errors.extend(mse.cpu().numpy())
    
    return np.array(reconstruction_errors)

def create_precision_recall_plots(y_true, scores, snr_values=None):
    """Create separate precision-recall plots that stay open"""
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    avg_precision = average_precision_score(y_true, scores)
    
    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    
    # Calculate ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    auc_score = roc_auc_score(y_true, scores)
    
    # Create Figure 1: Main Precision-Recall Curve
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(recall, precision, 'b-', linewidth=3, label=f'AP = {avg_precision:.3f}')
    plt.plot(optimal_recall, optimal_precision, 'ro', markersize=12, 
             label=f'Optimal (F1={optimal_f1:.3f})')
    
    # Add baseline
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.8, 
                label=f'Random (AP={baseline:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # ROC Curve
    plt.subplot(2, 2, 2)
    plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Score Distribution
    plt.subplot(2, 2, 3)
    signal_scores = scores[y_true == 1]
    noise_scores = scores[y_true == 0]
    
    plt.hist(noise_scores, alpha=0.7, label=f'Noise (n={len(noise_scores)})', 
             bins=30, color='blue', density=True)
    plt.hist(signal_scores, alpha=0.7, label=f'Signals (n={len(signal_scores)})', 
             bins=30, color='red', density=True)
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Optimal Threshold = {optimal_threshold:.6f}')
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Precision and Recall vs Threshold
    plt.subplot(2, 2, 4)
    plt.plot(thresholds, precision[:-1], 'b-', linewidth=2, label='Precision')
    plt.plot(thresholds, recall[:-1], 'r-', linewidth=2, label='Recall')
    plt.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2,
                label='Optimal')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('precision_recall_main.png', dpi=150, bbox_inches='tight')
    logger.info("Main plot saved as 'precision_recall_main.png'")
    
    # Create Figure 2: Performance Analysis
    plt.figure(figsize=(15, 10))
    
    # Performance at different thresholds
    plt.subplot(2, 3, 1)
    thresholds_to_test = np.percentile(scores, [70, 75, 80, 85, 90, 95])
    threshold_metrics = []
    
    for thresh in thresholds_to_test:
        predictions = (scores > thresh).astype(int)
        if np.sum(predictions) > 0:
            prec = precision_score(y_true, predictions, zero_division=0)
            rec = recall_score(y_true, predictions, zero_division=0)
            f1 = f1_score(y_true, predictions, zero_division=0)
        else:
            prec = rec = f1 = 0
        threshold_metrics.append([prec, rec, f1])
    
    threshold_metrics = np.array(threshold_metrics)
    
    x_pos = np.arange(len(thresholds_to_test))
    width = 0.25
    
    bars1 = plt.bar(x_pos - width, threshold_metrics[:, 0], width, 
                    label='Precision', alpha=0.8, color='blue')
    bars2 = plt.bar(x_pos, threshold_metrics[:, 1], width, 
                    label='Recall', alpha=0.8, color='red')
    bars3 = plt.bar(x_pos + width, threshold_metrics[:, 2], width, 
                    label='F1-Score', alpha=0.8, color='green')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Percentile Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance at Different Thresholds', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [f'{int(p)}%' for p in [70, 75, 80, 85, 90, 95]])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Detection rate vs SNR (if provided)
    if snr_values is not None:
        plt.subplot(2, 3, 2)
        signal_mask = y_true == 1
        if np.any(signal_mask):
            signal_snrs = snr_values[signal_mask]
            signal_predictions = (scores[signal_mask] > optimal_threshold).astype(int)
            
            # Bin by SNR
            snr_bins = np.linspace(np.min(signal_snrs), np.max(signal_snrs), 6)
            detection_rates = []
            bin_centers = []
            counts = []
            
            for i in range(len(snr_bins)-1):
                mask = (signal_snrs >= snr_bins[i]) & (signal_snrs < snr_bins[i+1])
                if np.sum(mask) > 0:
                    rate = np.mean(signal_predictions[mask])
                    detection_rates.append(rate)
                    bin_centers.append((snr_bins[i] + snr_bins[i+1]) / 2)
                    counts.append(np.sum(mask))
            
            if detection_rates:
                bars = plt.bar(bin_centers, detection_rates, alpha=0.8, color='green', width=2)
                plt.xlabel('SNR', fontsize=12)
                plt.ylabel('Detection Rate', fontsize=12)
                plt.title('Detection Rate vs SNR', fontsize=14, fontweight='bold')
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                
                # Add count labels
                for bar, count in zip(bars, counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Confusion matrices at different thresholds
    confusion_thresholds = [80, 90, 95]
    for i, thresh_pct in enumerate(confusion_thresholds):
        plt.subplot(2, 3, 3 + i)
        thresh_val = np.percentile(scores, thresh_pct)
        predictions = (scores > thresh_val).astype(int)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, predictions)
        
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix\n{thresh_pct}% Threshold', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                plt.text(k, j, cm[j, k], ha="center", va="center", 
                        fontsize=14, fontweight='bold')
        
        plt.xticks([0, 1], ['Pred\nNoise', 'Pred\nSignal'])
        plt.yticks([0, 1], ['True\nNoise', 'True\nSignal'])
        
        if i == 2:  # Add colorbar to last subplot
            plt.colorbar(im)
    
    plt.tight_layout()
    plt.savefig('precision_recall_analysis.png', dpi=150, bbox_inches='tight')
    logger.info("Analysis plot saved as 'precision_recall_analysis.png'")
    
    return {
        'avg_precision': avg_precision,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'optimal_precision': optimal_precision,
        'optimal_recall': optimal_recall,
        'auc': auc_score
    }

def main():
    logger.info("Creating Precision-Recall Plots (Fixed Version)")
    logger.info("=" * 55)
    
    # Quick data generation (smaller dataset for speed)
    SAMPLE_RATE = 512
    DURATION = 4
    NUM_SAMPLES = 200
    SIGNAL_PROB = 0.3
    
    logger.info(f"Generating {NUM_SAMPLES} samples...")
    
    # Generate data
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    strain_data = []
    labels = []
    snr_values = []
    
    for i in range(NUM_SAMPLES):
        noise = generate_colored_noise(len(t), SAMPLE_RATE, seed=42+i)
        
        if np.random.random() < SIGNAL_PROB:
            m1 = np.random.uniform(25, 50)
            m2 = np.random.uniform(25, 50) 
            distance = np.random.uniform(300, 800)
            
            gw_signal = generate_realistic_chirp(t, m1, m2, distance)
            
            signal_power = np.std(gw_signal)
            noise_power = np.std(noise)
            target_snr = np.random.uniform(8, 25)
            
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
    
    strain_data = np.array(strain_data)
    labels = np.array(labels)
    snr_values = np.array(snr_values)
    
    logger.info(f"Dataset: {strain_data.shape}")
    logger.info(f"Signals: {np.sum(labels)}, Noise: {np.sum(1-labels)}")
    
    # Compute CWT (smaller size for speed)
    logger.info(f"🌊 Computing CWT representations...")
    cwt_data = preprocess_with_cwt(strain_data, SAMPLE_RATE, target_height=32)
    
    # Train autoencoder
    logger.info(f"Training autoencoder...")
    noise_indices = np.where(labels == 0)[0]
    noise_cwt = cwt_data[noise_indices]
    
    noise_dataset = TensorDataset(torch.FloatTensor(noise_cwt).unsqueeze(1))
    noise_loader = DataLoader(noise_dataset, batch_size=8, shuffle=True)
    
    height, width = cwt_data.shape[1], cwt_data.shape[2]
    model = SimpleCWTAutoencoder(height, width, latent_dim=64)
    
    train_autoencoder_silent(model, noise_loader, num_epochs=20, lr=0.001)
    
    # Get reconstruction errors
    logger.info(f"Computing reconstruction errors...")
    test_dataset = TensorDataset(torch.FloatTensor(cwt_data).unsqueeze(1))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    reconstruction_errors = get_reconstruction_errors(model, test_loader)
    
    # Create plots
    logger.info(f"Creating precision-recall plots...")
    results = create_precision_recall_plots(labels, reconstruction_errors, snr_values)
    
    # Print results
    logger.info(f"\nRESULTS:")
    logger.info(f"Average Precision: {results['avg_precision']:.3f}")
    logger.info(f"AUC: {results['auc']:.3f}")
    logger.info(f"At Optimal Threshold:")
    logger.info(f"  • Precision: {results['optimal_precision']:.1%}")
    logger.info(f"  • Recall: {results['optimal_recall']:.1%}")
    logger.info(f"  • F1-Score: {results['optimal_f1']:.1%}")
    
    logger.info(f"\nPLOTS CREATED:")
    logger.info(f"'precision_recall_main.png' - Main precision-recall curves")
    logger.info(f"'precision_recall_analysis.png' - Detailed analysis")
    
    # Keep plots open
    logger.info(f"\n⏸️  Plots are now open and saved as files.")
    logger.info(f"You can view the saved PNG files if the plots close.")
    
    # This will keep the script running and plots open
    try:
        input("\nPress Enter when you're done viewing the plots...")
    except KeyboardInterrupt:
        logger.info(f"\nClosing plots...")
    
    plt.show(block=False)  # Show plots without blocking
    
    return results

if __name__ == "__main__":
    results = main()

