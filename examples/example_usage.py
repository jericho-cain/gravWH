#!/usr/bin/env python3
"""
Example Usage of CWT-LSTM Autoencoder for Gravitational Wave Detection
Demonstrates the complete pipeline from data generation to anomaly detection
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pywt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def generate_example_data(num_samples=100, sample_rate=512, duration=4):
    """Generate simple example gravitational wave data"""
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    strain_data = []
    labels = []
    
    for i in range(num_samples):
        # Generate colored noise
        noise = np.random.normal(0, 1e-23, len(t))
        
        if i < num_samples // 3:  # 1/3 are signals
            # Simple chirp signal  
            f0, f1 = 35, 350
            freq = f0 + (f1 - f0) * (t / t[-1])**3
            phase = 2 * np.pi * np.cumsum(freq) * (t[1] - t[0])
            
            # Amplitude envelope
            envelope = np.exp(-((t - t[-1]/2) / (t[-1]/8))**2)
            signal = 1e-21 * envelope * np.sin(phase)
            
            # Scale to realistic SNR
            target_snr = np.random.uniform(10, 20)
            signal = signal * (target_snr * np.std(noise) / np.std(signal))
            
            combined = noise + signal
            label = 1
        else:
            combined = noise  
            label = 0
            
        strain_data.append(combined)
        labels.append(label)
    
    return np.array(strain_data), np.array(labels)

def compute_cwt(strain, sample_rate):
    """Compute Continuous Wavelet Transform"""
    
    # Preprocessing
    sos = signal.butter(4, 20, btype='highpass', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, strain)
    whitened = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
    
    # CWT with Morlet wavelets
    freqs = np.logspace(np.log10(20), np.log10(512), 32)
    scales = sample_rate / freqs
    
    coefficients, _ = pywt.cwt(whitened, scales, 'morl', sampling_period=1/sample_rate)
    scalogram = np.abs(coefficients)
    
    # Log transform and normalize
    log_scalogram = np.log10(scalogram + 1e-10)
    normalized = (log_scalogram - np.mean(log_scalogram)) / (np.std(log_scalogram) + 1e-10)
    
    return normalized

class SimpleCWTAutoencoder(nn.Module):
    """Simple CWT Autoencoder for demonstration"""
    
    def __init__(self, height, width, latent_dim=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
        self.height = height
        self.width = width
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        # Resize to original dimensions
        if reconstructed.shape[-2:] != (self.height, self.width):
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=(self.height, self.width), mode='bilinear', align_corners=False
            )
        
        return reconstructed, latent

def train_autoencoder(model, noise_data, num_epochs=20):
    """Train autoencoder on noise-only data"""
    
    dataset = TensorDataset(torch.FloatTensor(noise_data).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in loader:
            data = batch[0]
            
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(loader):.6f}")

def detect_gravitational_waves(model, test_data, threshold_percentile=90):
    """Detect gravitational waves using reconstruction error"""
    
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for i in range(len(test_data)):
            data = torch.FloatTensor(test_data[i:i+1]).unsqueeze(1)
            reconstructed, _ = model(data)
            error = torch.mean((reconstructed - data)**2).item()
            reconstruction_errors.append(error)
    
    # Set threshold
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    predictions = (np.array(reconstruction_errors) > threshold).astype(int)
    
    return predictions, reconstruction_errors, threshold

def main():
    print("🌌 CWT-LSTM Autoencoder Example")
    print("=" * 40)
    
    # Parameters
    SAMPLE_RATE = 512
    DURATION = 4
    NUM_SAMPLES = 100
    
    print(f"Generating {NUM_SAMPLES} example samples...")
    
    # Generate example data
    strain_data, labels = generate_example_data(NUM_SAMPLES, SAMPLE_RATE, DURATION)
    
    print(f"Generated {np.sum(labels)} signals and {np.sum(1-labels)} noise samples")
    
    # Compute CWT for all samples
    print("🌊 Computing CWT representations...")
    cwt_data = []
    for strain in strain_data:
        cwt = compute_cwt(strain, SAMPLE_RATE)
        cwt_data.append(cwt)
    cwt_data = np.array(cwt_data)
    
    print(f"✅ CWT data shape: {cwt_data.shape}")
    
    # Separate noise and all data
    noise_indices = np.where(labels == 0)[0]
    noise_cwt = cwt_data[noise_indices]
    
    print(f"Training autoencoder on {len(noise_cwt)} noise samples...")
    
    # Create and train model
    height, width = cwt_data.shape[1], cwt_data.shape[2]
    model = SimpleCWTAutoencoder(height, width, latent_dim=32)
    
    train_autoencoder(model, noise_cwt, num_epochs=15)
    
    # Detect gravitational waves
    print("Detecting gravitational waves...")
    predictions, errors, threshold = detect_gravitational_waves(model, cwt_data, threshold_percentile=85)
    
    # Calculate performance
    accuracy = np.mean(predictions == labels)
    true_positives = np.sum((labels == 1) & (predictions == 1))
    false_positives = np.sum((labels == 0) & (predictions == 1))
    false_negatives = np.sum((labels == 1) & (predictions == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1-Score: {f1:.1%}")
    print(f"Threshold: {threshold:.6f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Sample CWT scalograms
    plt.subplot(2, 3, 1)
    signal_idx = np.where(labels == 1)[0][0]
    plt.imshow(cwt_data[signal_idx], aspect='auto', origin='lower', cmap='viridis')
    plt.title('CWT: Gravitational Wave Signal')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    noise_idx = np.where(labels == 0)[0][0]
    plt.imshow(cwt_data[noise_idx], aspect='auto', origin='lower', cmap='viridis')
    plt.title('CWT: Noise Only')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    
    # Reconstruction error distribution
    plt.subplot(2, 3, 3)
    signal_errors = np.array(errors)[labels == 1]
    noise_errors = np.array(errors)[labels == 0]
    
    plt.hist(noise_errors, alpha=0.7, label='Noise', bins=15, color='blue', density=True)
    plt.hist(signal_errors, alpha=0.7, label='Signals', bins=15, color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Original time series
    plt.subplot(2, 3, 4)
    t = np.linspace(0, DURATION, len(strain_data[0]))
    plt.plot(t, strain_data[signal_idx], label='Signal', linewidth=1)
    plt.plot(t, strain_data[noise_idx], label='Noise', alpha=0.7, linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.title('Original Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance metrics
    plt.subplot(2, 3, 5)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.8)
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.1%}', ha='center', va='bottom')
    
    # Confusion matrix
    plt.subplot(2, 3, 6)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    
    plt.xticks([0, 1], ['Pred Noise', 'Pred Signal'])
    plt.yticks([0, 1], ['True Noise', 'True Signal'])
    
    plt.tight_layout()
    plt.savefig('example_usage_results.png', dpi=150, bbox_inches='tight')
    print(f"\nResults saved to 'example_usage_results.png'")
    
    plt.show()
    
    print(f"\nExample completed!")
    print(f"This demonstrates the CWT-LSTM autoencoder approach for gravitational wave detection.")
    print(f"For full analysis, run: python gravitational_wave_hunter/models/cwt_lstm_autoencoder.py")

if __name__ == "__main__":
    main()

