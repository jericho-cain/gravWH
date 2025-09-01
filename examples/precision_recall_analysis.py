#!/usr/bin/env python3
"""
Comprehensive Precision-Recall Analysis for CWT-LSTM Autoencoder
Detailed analysis with multiple threshold evaluations and publication-quality plots
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

def train_autoencoder(model, noise_loader, num_epochs=50, lr=0.001):
    """Train autoencoder on noise-only data"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model = model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in noise_loader:
            data = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed, latent = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

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

def precision_recall_analysis(y_true, scores, title="Precision-Recall Analysis"):
    """Comprehensive precision-recall analysis"""
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Calculate average precision
    avg_precision = average_precision_score(y_true, scores)
    
    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    
    # Calculate metrics at different thresholds
    thresholds_to_test = np.percentile(scores, [70, 75, 80, 85, 90, 95])
    
    results = {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'avg_precision': avg_precision,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'optimal_precision': optimal_precision,
        'optimal_recall': optimal_recall,
        'thresholds_to_test': thresholds_to_test
    }
    
    return results

def plot_comprehensive_analysis(y_true, scores, snr_values=None):
    """Create comprehensive precision-recall plots"""
    
    # Calculate PR analysis
    pr_results = precision_recall_analysis(y_true, scores)
    
    # Calculate ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    auc_score = roc_auc_score(y_true, scores)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    
    # 1. Precision-Recall Curve
    axes[0, 0].plot(pr_results['recall'], pr_results['precision'], 'b-', linewidth=2, 
                    label=f'AP = {pr_results["avg_precision"]:.3f}')
    axes[0, 0].plot(pr_results['optimal_recall'], pr_results['optimal_precision'], 
                    'ro', markersize=10, label=f'Optimal (F1={pr_results["optimal_f1"]:.3f})')
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])
    
    # Add baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    axes[0, 0].axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, 
                       label=f'Random (AP={baseline:.3f})')
    
    # 2. ROC Curve
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1-Score vs Threshold
    axes[0, 2].plot(pr_results['thresholds'], 
                    2 * (pr_results['precision'][:-1] * pr_results['recall'][:-1]) / 
                    (pr_results['precision'][:-1] + pr_results['recall'][:-1] + 1e-10))
    axes[0, 2].axvline(x=pr_results['optimal_threshold'], color='red', linestyle='--', 
                       label=f'Optimal = {pr_results["optimal_threshold"]:.6f}')
    axes[0, 2].set_xlabel('Threshold')
    axes[0, 2].set_ylabel('F1-Score')
    axes[0, 2].set_title('F1-Score vs Threshold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Score Distribution
    signal_scores = scores[y_true == 1]
    noise_scores = scores[y_true == 0]
    
    axes[1, 0].hist(noise_scores, alpha=0.7, label='Noise', bins=30, color='blue', density=True)
    axes[1, 0].hist(signal_scores, alpha=0.7, label='Signals', bins=30, color='red', density=True)
    axes[1, 0].axvline(x=pr_results['optimal_threshold'], color='black', linestyle='--', 
                       label='Optimal Threshold')
    axes[1, 0].set_xlabel('Reconstruction Error')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Precision and Recall vs Threshold
    axes[1, 1].plot(pr_results['thresholds'], pr_results['precision'][:-1], 'b-', label='Precision')
    axes[1, 1].plot(pr_results['thresholds'], pr_results['recall'][:-1], 'r-', label='Recall')
    axes[1, 1].axvline(x=pr_results['optimal_threshold'], color='black', linestyle='--', 
                       label='Optimal')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance at Different Thresholds
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
    
    axes[1, 2].bar(x_pos - width, threshold_metrics[:, 0], width, label='Precision', alpha=0.8)
    axes[1, 2].bar(x_pos, threshold_metrics[:, 1], width, label='Recall', alpha=0.8)
    axes[1, 2].bar(x_pos + width, threshold_metrics[:, 2], width, label='F1-Score', alpha=0.8)
    
    axes[1, 2].set_xlabel('Percentile Threshold')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Metrics at Different Thresholds')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f'{int(p)}%' for p in [70, 75, 80, 85, 90, 95]])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Detection Rate vs SNR (if SNR values provided)
    if snr_values is not None:
        signal_mask = y_true == 1
        if np.any(signal_mask):
            signal_snrs = snr_values[signal_mask]
            signal_predictions = (scores[signal_mask] > pr_results['optimal_threshold']).astype(int)
            
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
                bars = axes[2, 0].bar(bin_centers, detection_rates, alpha=0.7, color='green')
                axes[2, 0].set_xlabel('SNR')
                axes[2, 0].set_ylabel('Detection Rate')
                axes[2, 0].set_title('Detection Rate vs SNR')
                axes[2, 0].set_ylim(0, 1)
                axes[2, 0].grid(True, alpha=0.3)
                
                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    axes[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                    f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # 8. Confusion Matrices at Different Thresholds
    for i, (thresh_pct, thresh_val) in enumerate(zip([80, 90, 95], 
                                                    [np.percentile(scores, p) for p in [80, 90, 95]])):
        predictions = (scores > thresh_val).astype(int)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, predictions)
        
        ax = axes[2, i] if i < 2 else axes[2, 2]
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'Confusion Matrix\n{thresh_pct}% Threshold')
        
        # Add text annotations
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, cm[j, k], ha="center", va="center", fontsize=12)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pred Noise', 'Pred Signal'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['True Noise', 'True Signal'])
        
        if i == 2:  # Add colorbar to last subplot
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig, pr_results

def main():
    logger.info("Comprehensive Precision-Recall Analysis")
    logger.info("=" * 50)
    
    # Configuration
    SAMPLE_RATE = 512
    DURATION = 4
    NUM_SAMPLES = 300  # More samples for better statistics
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
    
    print(f"Dataset: {strain_data.shape}")
    print(f"Signals: {np.sum(labels)}, Noise: {np.sum(1-labels)}")
    
    # Compute CWT
    print(f"\n🌊 Computing CWT representations...")
    cwt_data = preprocess_with_cwt(strain_data, SAMPLE_RATE, target_height=32)
    
    # Train autoencoder on noise only
    noise_indices = np.where(labels == 0)[0]
    noise_cwt = cwt_data[noise_indices]
    
    print(f"\nTraining autoencoder on {len(noise_cwt)} noise samples...")
    
    noise_dataset = TensorDataset(torch.FloatTensor(noise_cwt).unsqueeze(1))
    noise_loader = DataLoader(noise_dataset, batch_size=8, shuffle=True)
    
    # Create and train model
    height, width = cwt_data.shape[1], cwt_data.shape[2]
    model = SimpleCWTAutoencoder(height, width, latent_dim=64)
    
    train_autoencoder(model, noise_loader, num_epochs=40, lr=0.001)
    
    # Get reconstruction errors for all samples
    print(f"\nComputing reconstruction errors...")
    test_dataset = TensorDataset(torch.FloatTensor(cwt_data).unsqueeze(1))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    reconstruction_errors = get_reconstruction_errors(model, test_loader)
    
    # Create comprehensive analysis
    print(f"\nCreating precision-recall analysis...")
    
    fig, pr_results = plot_comprehensive_analysis(labels, reconstruction_errors, snr_values)
    
    # Print detailed results
    print(f"\nPRECISION-RECALL ANALYSIS RESULTS:")
    print(f"=" * 50)
    print(f"Average Precision: {pr_results['avg_precision']:.3f}")
    print(f"Optimal Threshold: {pr_results['optimal_threshold']:.6f}")
    print(f"At Optimal Threshold:")
    print(f"  • Precision: {pr_results['optimal_precision']:.1%}")
    print(f"  • Recall: {pr_results['optimal_recall']:.1%}")
    print(f"  • F1-Score: {pr_results['optimal_f1']:.1%}")
    
    # Test at different thresholds
    print(f"\nPerformance at Different Percentile Thresholds:")
    thresholds_to_test = np.percentile(reconstruction_errors, [70, 75, 80, 85, 90, 95])
    
    for percentile, thresh in zip([70, 75, 80, 85, 90, 95], thresholds_to_test):
        predictions = (reconstruction_errors > thresh).astype(int)
        
        if np.sum(predictions) > 0:
            prec = precision_score(labels, predictions, zero_division=0)
            rec = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
        else:
            prec = rec = f1 = 0
        
        print(f"  {percentile:2d}% threshold: Prec={prec:.1%}, Rec={rec:.1%}, F1={f1:.1%}")
    
    # Interpretation
    print(f"\nINTERPRETATION:")
    print(f"=" * 30)
    if pr_results['avg_precision'] > 0.7:
        print(f"✅ EXCELLENT: Average Precision > 0.7 indicates strong separation")
    elif pr_results['avg_precision'] > 0.5:
        print(f"✅ GOOD: Average Precision > 0.5 shows clear signal detection ability")
    else:
        print(f"⚠️ FAIR: Average Precision < 0.5 suggests difficulty in detection")
    
    print(f"\nOPTIMAL OPERATING POINT:")
    if pr_results['optimal_precision'] > 0.8 and pr_results['optimal_recall'] > 0.3:
        print(f"✅ Excellent balance of precision and recall")
    elif pr_results['optimal_precision'] > 0.9:
        print(f"Very conservative: High precision, low false alarms")
        print(f"   Good for avoiding false discoveries in astronomy")
    else:
        print(f"⚠️ May need threshold tuning for specific use case")
    
    # Save comprehensive analysis
    plt.savefig('results/precision_recall_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Generate standalone publication-quality figures
    print("\nGenerating individual publication plots...")
    
    # Figure 1: Precision-Recall Curve
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    ax_pr.plot(pr_results['recall'], pr_results['precision'], 'b-', linewidth=2.5, 
               label=f'AUPRC = {pr_results["avg_precision"]:.3f}')
    ax_pr.plot(pr_results['optimal_recall'], pr_results['optimal_precision'], 
               'ro', markersize=8, label=f'Optimal (F1={pr_results["optimal_f1"]:.3f})')
    ax_pr.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.5, label='Random Classifier')
    ax_pr.set_xlabel('Recall', fontsize=14)
    ax_pr.set_ylabel('Precision', fontsize=14)
    ax_pr.set_title('Precision-Recall Curve', fontsize=16)
    ax_pr.legend(fontsize=12)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: ROC Curve  
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUC-ROC = {auc_score:.3f}')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax_roc.set_xlabel('False Positive Rate', fontsize=14)
    ax_roc.set_ylabel('True Positive Rate', fontsize=14)
    ax_roc.set_title('ROC Curve', fontsize=16)
    ax_roc.legend(fontsize=12)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: SNR Performance (if SNR data exists)
    if len(snr_breakdown) > 0:
        fig_snr, ax_snr = plt.subplots(figsize=(10, 6))
        
        snr_values = sorted(snr_breakdown.keys())
        precision_values = [snr_breakdown[snr]['precision'] for snr in snr_values]
        recall_values = [snr_breakdown[snr]['recall'] for snr in snr_values]
        f1_values = [snr_breakdown[snr]['f1'] for snr in snr_values]
        
        ax_snr.plot(snr_values, precision_values, 'b-', linewidth=2.5, 
                    label='Precision', marker='o', markersize=6)
        ax_snr.plot(snr_values, recall_values, 'r-', linewidth=2.5, 
                    label='Recall', marker='s', markersize=6)
        ax_snr.plot(snr_values, f1_values, 'g-', linewidth=2.5, 
                    label='F1-Score', marker='^', markersize=6)
        
        # Add performance regions
        if max(snr_values) >= 15:
            ax_snr.axvspan(15, max(snr_values), alpha=0.2, color='green', label='High SNR (>15)')
        if 10 in snr_values and max(snr_values) >= 15:
            ax_snr.axvspan(10, 15, alpha=0.2, color='orange', label='Moderate SNR (10-15)')
        if min(snr_values) <= 10:
            ax_snr.axvspan(min(snr_values), min(10, max(snr_values)), alpha=0.2, color='red', label='Threshold SNR (<10)')
        
        ax_snr.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=14)
        ax_snr.set_ylabel('Detection Performance', fontsize=14)
        ax_snr.set_title('CWT-LSTM Autoencoder Performance vs Signal-to-Noise Ratio', fontsize=16)
        ax_snr.legend(fontsize=12, loc='lower right')
        ax_snr.grid(True, alpha=0.3)
        ax_snr.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig('results/snr_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SNR performance plot saved")
    else:
        print("⚠️ No SNR data available for standalone plot")
    
    print(f"✅ Publication plots saved to results/ directory:")
    print(f"   - precision_recall_curve.png")
    print(f"   - roc_curve.png") 
    if len(snr_breakdown) > 0:
        print(f"   - snr_performance.png")
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to 'precision_recall_comprehensive_analysis.png'")
    
    return pr_results

if __name__ == "__main__":
    results = main()

