# CWT-LSTM Autoencoder API Reference

*Complete documentation for classes, functions, and modules.*

## Table of Contents
- [Core Models](#core-models)
- [Signal Processing](#signal-processing)
- [Data Generation](#data-generation)
- [Evaluation Tools](#evaluation-tools)
- [Utilities](#utilities)
- [Examples](#examples)

---

## Core Models

### SimpleCWTAutoencoder

Main autoencoder class for gravitational wave detection.

```python
class SimpleCWTAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for gravitational wave detection using CWT input.
    
    This model processes Continuous Wavelet Transform scalograms through
    LSTM layers to learn normal noise patterns and detect gravitational
    wave anomalies.
    """
```

#### Constructor

```python
def __init__(self, input_size=32, latent_dim=64, dropout=0.2):
    """
    Initialize the CWT-LSTM autoencoder.
    
    Args:
        input_size (int): Number of frequency bins in CWT (default: 32)
        latent_dim (int): Dimension of latent bottleneck layer (default: 64)
        dropout (float): Dropout rate for regularization (default: 0.2)
    
    Architecture:
        - Encoder: [input_size → 128 → 64 → 32 → latent_dim]
        - Decoder: [latent_dim → 32 → 64 → 128 → input_size]
    """
```

#### Methods

```python
def forward(self, x):
    """
    Forward pass through the autoencoder.
    
    Args:
        x (torch.Tensor): Input CWT scalogram
            Shape: (batch_size, sequence_length, input_size)
            
    Returns:
        torch.Tensor: Reconstructed scalogram
            Shape: (batch_size, sequence_length, input_size)
    """

def encode(self, x):
    """
    Encode input to latent representation.
    
    Args:
        x (torch.Tensor): Input CWT scalogram
            
    Returns:
        torch.Tensor: Latent representation
            Shape: (batch_size, latent_dim)
    """

def decode(self, z):
    """
    Decode latent representation to reconstruction.
    
    Args:
        z (torch.Tensor): Latent representation
            
    Returns:
        torch.Tensor: Reconstructed scalogram
    """

def detect_anomalies(self, data, threshold=0.5):
    """
    Detect gravitational wave anomalies in data.
    
    Args:
        data (torch.Tensor): CWT scalograms to analyze
        threshold (float): Detection threshold for reconstruction error
            
    Returns:
        dict: {
            'predictions': Binary predictions (0=noise, 1=GW),
            'errors': Reconstruction errors,
            'anomaly_scores': Normalized anomaly scores
        }
    """
```

#### Example Usage

```python
import torch
from gravitational_wave_hunter.models import SimpleCWTAutoencoder

# Initialize model
model = SimpleCWTAutoencoder(input_size=32, latent_dim=64)

# Generate sample data (batch_size=10, seq_length=2048, features=32)
data = torch.randn(10, 2048, 32)

# Forward pass
reconstruction = model(data)
print(f"Input shape: {data.shape}")
print(f"Output shape: {reconstruction.shape}")

# Detect anomalies
results = model.detect_anomalies(data, threshold=0.1)
print(f"Detected {results['predictions'].sum()} anomalies")
```

---

## Signal Processing

### CWT Functions

```python
def compute_cwt_scalogram(strain_data, fs=512, frequencies=None):
    """
    Compute Continuous Wavelet Transform scalogram for gravitational wave analysis.
    
    Args:
        strain_data (np.ndarray): 1D strain time series
        fs (int): Sampling frequency in Hz (default: 512)
        frequencies (np.ndarray): Frequency array for CWT
            If None, uses logarithmic spacing from 20-512 Hz
            
    Returns:
        np.ndarray: Log-normalized power scalogram
            Shape: (n_frequencies, n_timepoints)
            
    Example:
        >>> strain = generate_synthetic_signal(duration=4.0)
        >>> scalogram = compute_cwt_scalogram(strain)
        >>> print(f"Scalogram shape: {scalogram.shape}")
        Scalogram shape: (32, 2048)
    """

def preprocess_strain_data(strain_data, fs=512, highpass_freq=20):
    """
    Preprocess strain data for gravitational wave analysis.
    
    Applies:
    1. High-pass filtering to remove low-frequency noise
    2. Whitening based on estimated PSD
    3. Normalization
    
    Args:
        strain_data (np.ndarray): Raw strain time series
        fs (int): Sampling frequency
        highpass_freq (float): High-pass filter cutoff frequency
        
    Returns:
        np.ndarray: Preprocessed strain data
    """

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply Butterworth bandpass filter to strain data.
    
    Args:
        data (np.ndarray): Input time series
        lowcut (float): Low cutoff frequency (Hz)
        highcut (float): High cutoff frequency (Hz) 
        fs (int): Sampling frequency (Hz)
        order (int): Filter order (default: 4)
        
    Returns:
        np.ndarray: Filtered time series
    """
```

### Noise Generation

```python
def generate_ligo_noise(duration=4.0, fs=512, seed=None):
    """
    Generate realistic LIGO-like colored Gaussian noise.
    
    Uses Advanced LIGO design sensitivity curve to create
    colored noise that matches detector characteristics.
    
    Args:
        duration (float): Duration in seconds (default: 4.0)
        fs (int): Sampling frequency (default: 512)
        seed (int): Random seed for reproducibility
        
    Returns:
        np.ndarray: Colored noise time series
        
    Example:
        >>> noise = generate_ligo_noise(duration=4.0, fs=512)
        >>> print(f"Generated {len(noise)} samples of noise")
        Generated 2048 samples of noise
    """

def advanced_ligo_psd(frequencies):
    """
    Compute Advanced LIGO power spectral density.
    
    Args:
        frequencies (np.ndarray): Frequency array
        
    Returns:
        np.ndarray: PSD values [strain²/Hz]
        
    Reference:
        Advanced LIGO design sensitivity curve
    """
```

---

## Data Generation

### Synthetic Signals

```python
def generate_bbh_waveform(m1, m2, distance=100, duration=4.0, fs=512):
    """
    Generate binary black hole merger waveform.
    
    Uses post-Newtonian approximation for inspiral phase
    and simplified merger/ringdown.
    
    Args:
        m1, m2 (float): Component masses in solar masses
        distance (float): Luminosity distance in Mpc (default: 100)
        duration (float): Signal duration in seconds (default: 4.0)
        fs (int): Sampling frequency (default: 512)
        
    Returns:
        dict: {
            'strain': Gravitational wave strain time series,
            'time': Time array,
            'parameters': Source parameters,
            'snr': Optimal signal-to-noise ratio
        }
        
    Example:
        >>> signal = generate_bbh_waveform(m1=30, m2=25, distance=200)
        >>> print(f"Generated BBH signal with SNR: {signal['snr']:.1f}")
        Generated BBH signal with SNR: 12.3
    """

def inject_signal_into_noise(signal, noise, snr_target=None):
    """
    Inject gravitational wave signal into noise at specified SNR.
    
    Args:
        signal (np.ndarray): Gravitational wave strain
        noise (np.ndarray): Background noise
        snr_target (float): Target signal-to-noise ratio
            If None, uses signal's natural amplitude
            
    Returns:
        dict: {
            'strain': Combined signal + noise,
            'clean_signal': Pure signal component,
            'noise': Noise component,
            'snr_actual': Achieved SNR
        }
    """

class SyntheticDataGenerator:
    """
    Generator for synthetic gravitational wave datasets.
    
    Creates balanced datasets with realistic signal parameters
    and noise characteristics for training and testing.
    """
    
    def __init__(self, fs=512, duration=4.0, noise_type='ligo'):
        """
        Initialize synthetic data generator.
        
        Args:
            fs (int): Sampling frequency
            duration (float): Sample duration 
            noise_type (str): Noise model ('ligo', 'gaussian', 'colored')
        """
    
    def generate_dataset(self, n_samples, signal_fraction=0.5, 
                        snr_range=(8, 25), mass_range=(10, 80)):
        """
        Generate complete training/testing dataset.
        
        Args:
            n_samples (int): Total number of samples
            signal_fraction (float): Fraction containing signals (0.0-1.0)
            snr_range (tuple): SNR range for injected signals
            mass_range (tuple): Mass range for black holes (solar masses)
            
        Returns:
            dict: {
                'data': CWT scalograms (n_samples, n_freq, n_time),
                'labels': Binary labels (n_samples,),
                'metadata': Sample parameters and characteristics
            }
        """
```

---

## Evaluation Tools

### Performance Metrics

```python
def compute_detection_metrics(y_true, y_pred, y_scores=None):
    """
    Compute comprehensive detection performance metrics.
    
    Args:
        y_true (np.ndarray): True binary labels
        y_pred (np.ndarray): Predicted binary labels  
        y_scores (np.ndarray): Prediction scores/probabilities
        
    Returns:
        dict: {
            'accuracy': Overall accuracy,
            'precision': Precision (true positives / predicted positives),
            'recall': Recall (true positives / actual positives), 
            'f1_score': F1 score (harmonic mean of precision/recall),
            'auc_roc': Area under ROC curve (if y_scores provided),
            'auc_prc': Area under precision-recall curve,
            'confusion_matrix': 2x2 confusion matrix
        }
    """

def precision_recall_analysis(y_true, y_scores, n_thresholds=1000):
    """
    Comprehensive precision-recall analysis across thresholds.
    
    Args:
        y_true (np.ndarray): True binary labels
        y_scores (np.ndarray): Anomaly scores/reconstruction errors
        n_thresholds (int): Number of thresholds to evaluate
        
    Returns:
        dict: {
            'thresholds': Threshold values,
            'precisions': Precision at each threshold,
            'recalls': Recall at each threshold,
            'f1_scores': F1 score at each threshold,
            'optimal_threshold': Threshold maximizing F1,
            'operating_points': Key operating points for different use cases
        }
    """

def find_operating_points(precisions, recalls, thresholds):
    """
    Find key operating points for different detection scenarios.
    
    Returns:
        dict: Operating points with precision, recall, and use cases
    """
```

### Visualization

```python
def plot_precision_recall_curve(precisions, recalls, auc_prc=None, 
                               operating_points=None):
    """
    Plot precision-recall curve with operating points.
    
    Args:
        precisions (np.ndarray): Precision values
        recalls (np.ndarray): Recall values
        auc_prc (float): Area under PR curve
        operating_points (dict): Key operating points to highlight
        
    Returns:
        matplotlib.figure.Figure: Precision-recall plot
    """

def plot_roc_curve(fpr, tpr, auc_roc=None):
    """
    Plot ROC curve showing true positive vs false positive rates.
    
    Args:
        fpr (np.ndarray): False positive rates
        tpr (np.ndarray): True positive rates  
        auc_roc (float): Area under ROC curve
        
    Returns:
        matplotlib.figure.Figure: ROC plot
    """

def plot_cwt_scalogram(scalogram, frequencies=None, time=None, 
                      title="CWT Scalogram"):
    """
    Visualize CWT scalogram with proper frequency/time axes.
    
    Args:
        scalogram (np.ndarray): 2D scalogram (freq × time)
        frequencies (np.ndarray): Frequency values for y-axis
        time (np.ndarray): Time values for x-axis
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: Scalogram visualization
    """
```

---

## Utilities

### Configuration Management

```python
class Config:
    """
    Configuration management for gravitational wave detection pipeline.
    
    Centralizes all hyperparameters, file paths, and settings.
    """
    
    # Model parameters
    MODEL_CONFIG = {
        'input_size': 32,
        'latent_dim': 64,
        'dropout': 0.2,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'epochs': 100
    }
    
    # Signal processing parameters  
    SIGNAL_CONFIG = {
        'sampling_frequency': 512,
        'duration': 4.0,
        'n_frequencies': 32,
        'freq_range': (20, 512),
        'highpass_cutoff': 20
    }
    
    # Detection parameters
    DETECTION_CONFIG = {
        'default_threshold': 0.1,
        'precision_target': 0.90,
        'snr_range': (8, 25)
    }

def load_config(config_file=None):
    """Load configuration from YAML file or use defaults."""
    
def save_config(config, config_file):
    """Save configuration to YAML file."""
```

### Logging and Monitoring

```python
def setup_logging(log_level='INFO', log_file=None):
    """
    Set up logging for gravitational wave detection pipeline.
    
    Args:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file (str): Optional log file path
    """

class ModelTracker:
    """
    Track model training progress and performance metrics.
    
    Integrates with popular MLOps tools like Weights & Biases,
    TensorBoard, or MLflow.
    """
    
    def __init__(self, experiment_name, tracking_backend='wandb'):
        """Initialize experiment tracking."""
        
    def log_metrics(self, metrics, step=None):
        """Log training/validation metrics."""
        
    def log_model(self, model, model_name):
        """Save model checkpoint with metadata."""
        
    def log_artifacts(self, artifact_paths):
        """Log plots, results, and other artifacts."""
```

---

## Examples

### Complete Training Pipeline

```python
from gravitational_wave_hunter.models import SimpleCWTAutoencoder
from gravitational_wave_hunter.data import SyntheticDataGenerator
from gravitational_wave_hunter.evaluation import compute_detection_metrics

# 1. Generate training data
generator = SyntheticDataGenerator(fs=512, duration=4.0)
train_data = generator.generate_dataset(
    n_samples=10000, 
    signal_fraction=0.0  # Noise-only for unsupervised training
)

# 2. Initialize and train model
model = SimpleCWTAutoencoder(input_size=32, latent_dim=64)
model.train_model(train_data['data'], epochs=100, batch_size=64)

# 3. Generate test data with signals
test_data = generator.generate_dataset(
    n_samples=2000,
    signal_fraction=0.5  # Balanced test set
)

# 4. Evaluate performance
results = model.detect_anomalies(test_data['data'], threshold=0.1)
metrics = compute_detection_metrics(
    test_data['labels'], 
    results['predictions'],
    results['errors']
)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

### Custom Signal Analysis

```python
# Load your own data
import numpy as np
from gravitational_wave_hunter.signal_processing import compute_cwt_scalogram

# Load strain data (example with LIGO data format)
strain_data = np.load('your_strain_data.npy')

# Preprocess and compute CWT
scalogram = compute_cwt_scalogram(strain_data, fs=4096)

# Load pre-trained model
model = SimpleCWTAutoencoder.load_pretrained('path/to/model.pth')

# Detect anomalies
results = model.detect_anomalies(scalogram[None, ...], threshold=0.1)

if results['predictions'][0]:
    print(f"Gravitational wave detected with score: {results['errors'][0]:.4f}")
else:
    print("No gravitational wave detected")
```

### Batch Processing

```python
from pathlib import Path
import torch

def process_data_directory(data_dir, model_path, output_dir):
    """
    Process all strain data files in a directory.
    
    Args:
        data_dir (str): Directory containing .npy strain files
        model_path (str): Path to trained model
        output_dir (str): Directory for results
    """
    
    # Load model
    model = SimpleCWTAutoencoder.load_pretrained(model_path)
    
    # Process all files
    data_files = list(Path(data_dir).glob('*.npy'))
    results = []
    
    for file_path in data_files:
        # Load and preprocess data
        strain_data = np.load(file_path)
        scalogram = compute_cwt_scalogram(strain_data)
        
        # Detect gravitational waves
        detection_result = model.detect_anomalies(
            scalogram[None, ...], 
            threshold=0.1
        )
        
        results.append({
            'file': file_path.name,
            'detection': bool(detection_result['predictions'][0]),
            'score': float(detection_result['errors'][0]),
            'timestamp': file_path.stat().st_mtime
        })
    
    # Save results
    import json
    with open(Path(output_dir) / 'detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Usage
results = process_data_directory(
    data_dir='./ligo_data/',
    model_path='./models/trained_model.pth', 
    output_dir='./results/'
)
```

---

## Error Handling

### Common Exceptions

```python
class GWDetectionError(Exception):
    """Base exception for gravitational wave detection errors."""

class InvalidDataShapeError(GWDetectionError):
    """Raised when input data has incorrect shape."""

class ModelNotTrainedError(GWDetectionError):
    """Raised when attempting to use untrained model."""

class InsufficientDataError(GWDetectionError):
    """Raised when insufficient data for reliable analysis."""

# Example usage with error handling
try:
    results = model.detect_anomalies(data, threshold=0.1)
except InvalidDataShapeError as e:
    print(f"Data shape error: {e}")
    # Handle reshaping or reprocessing
except ModelNotTrainedError as e:
    print(f"Model not ready: {e}")
    # Load pre-trained model or train first
```

---

*This API reference covers all major components of the CWT-LSTM autoencoder gravitational wave detection system. For additional examples and tutorials, see the complete documentation and example notebooks.*
