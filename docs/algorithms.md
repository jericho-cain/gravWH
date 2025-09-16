# CWT-LSTM Autoencoder: Technical Algorithm Guide

*Deep dive into the algorithms, mathematics, and implementation details.*

## Table of Contents
- [Algorithm Overview](#algorithm-overview)
- [Continuous Wavelet Transform](#continuous-wavelet-transform)
- [LSTM Autoencoder Architecture](#lstm-autoencoder-architecture)
- [Training Strategy](#training-strategy)
- [Detection Pipeline](#detection-pipeline)
- [Performance Optimization](#performance-optimization)

---

## Algorithm Overview

###  Core Concept
The CWT-LSTM Autoencoder uses **unsupervised anomaly detection** to identify gravitational wave signals:

1. **Learn normal patterns**: Train autoencoder on noise-only data
2. **Detect anomalies**: High reconstruction error indicates gravitational waves
3. **Time-frequency preprocessing**: CWT captures chirp evolution optimally

###  Pipeline Architecture
```
Raw Strain Data (2048 samples, 512 Hz)
    ↓ Preprocessing
Filtered & Whitened Time Series
    ↓ CWT (Continuous Wavelet Transform)
Time-Frequency Scalogram (32 × 2048)
    ↓ LSTM Autoencoder
Reconstruction Error (scalar)
    ↓ Thresholding
Detection Decision (0/1)
```

---

## Continuous Wavelet Transform

###  Mathematical Foundation

The CWT transforms time-series data into time-frequency representation:

```math
W(a,b) = \frac{1}{\sqrt{a}} \int_{-∞}^{∞} h(t) \psi^*\left(\frac{t-b}{a}\right) dt
```

Where:
- `h(t)`: Input strain time series
- `ψ(t)`: Mother wavelet (Morlet)
- `a`: Scale parameter (∝ 1/frequency) 
- `b`: Translation parameter (time shift)
- `*`: Complex conjugation

### Morlet Wavelet
We use the **Morlet wavelet** optimized for gravitational wave analysis:

```math
ψ(t) = π^{-1/4} e^{iω₀t} e^{-t²/2}
```

**Why Morlet?**
- **Complex-valued**: Provides amplitude and phase information
- **Gaussian envelope**: Optimal time-frequency localization
- **ω₀ = 6**: Balance between time and frequency resolution
- **Chirp-like shape**: Naturally matches gravitational wave morphology

###  Scale Selection
Logarithmically spaced scales covering LIGO's sensitive frequency band:

```python
# Frequency range: 20-512 Hz (LIGO sensitive band)
frequencies = np.logspace(np.log10(20), np.log10(512), 32)
scales = w * fs / (2 * frequencies * np.pi)  # w=6 for Morlet
```

**Key Parameters**:
- **32 frequency bins**: Balance resolution vs computational cost
- **20-512 Hz range**: Advanced LIGO sensitivity curve
- **Log spacing**: More resolution at low frequencies (longer wavelengths)

### Implementation Details

```python
def compute_cwt(strain_data, fs=512):
    """
    Compute CWT scalogram for gravitational wave analysis
    
    Args:
        strain_data: 1D array of strain measurements
        fs: Sampling frequency (Hz)
    
    Returns:
        scalogram: 2D array (frequencies × time)
    """
    # Define scales for frequency range 20-512 Hz
    frequencies = np.logspace(np.log10(20), np.log10(512), 32)
    scales = 6 * fs / (2 * frequencies * np.pi)
    
    # Compute CWT using Morlet wavelet
    coefficients, _ = pywt.cwt(strain_data, scales, 'morl', 
                              sampling_period=1/fs)
    
    # Convert to power scalogram
    scalogram = np.abs(coefficients) ** 2
    
    # Log normalization for neural network input
    scalogram = np.log10(scalogram + 1e-12)
    
    return scalogram
```

###  Why CWT for Gravitational Waves?

1. **Chirp Evolution**: Gravitational waves sweep from low to high frequency
2. **Time Localization**: Merger happens in milliseconds
3. **Frequency Resolution**: Different mass ratios → different frequency ranges
4. **Phase Information**: Complex wavelets preserve signal phase

**Visual Comparison**:
```
Time Domain:     ∼∼∼∼∼∿∿∿∿WWWWW (hard to see pattern)
                     ↓ CWT
Time-Frequency:  
  High f  |  . . . . ▓▓▓██████  ← Merger phase
  Med f   |  . . ▓▓▓▓████████▓   ← Late inspiral  
  Low f   |  ▓▓████████▓▓. . .   ← Early inspiral
          └────────────────────→ Time
          (chirp pattern clearly visible!)
```

---

## LSTM Autoencoder Architecture

### Neural Network Design

Our LSTM autoencoder processes CWT scalograms through temporal sequence modeling:

```
Input: CWT Scalogram (32 × 2048)
  ↓ Reshape to sequence
Sequence: (2048 timesteps × 32 features)
  ↓ LSTM Encoder
Encoder: [128, 64, 32] → Latent (64 dims)
  ↓ LSTM Decoder  
Decoder: [32, 64, 128] → Reconstruction
  ↓ Reshape
Output: Reconstructed Scalogram (32 × 2048)
```

### LSTM Cell Mathematics

Each LSTM cell implements:

```math
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)      \text{ (Forget gate)}
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)      \text{ (Input gate)}  
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)   \text{ (Candidate values)}
C_t = f_t * C_{t-1} + i_t * C̃_t          \text{ (Cell state)}
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      \text{ (Output gate)}
h_t = o_t * tanh(C_t)                    \text{ (Hidden state)}
```

**Why LSTM?**
- **Long-term dependencies**: Remembers early inspiral when processing merger
- **Gradient flow**: Avoids vanishing gradients for long sequences
- **Selective memory**: Forget gate removes irrelevant noise information

###  Architecture Choices

**Encoder Design**:
```python
# Hierarchical compression
LSTM(input_size=32, hidden_size=128)  # Full frequency representation
LSTM(input_size=128, hidden_size=64)  # Intermediate compression
LSTM(input_size=64, hidden_size=32)   # High-level features
Linear(32 → 64)                       # Latent bottleneck
```

**Decoder Design**:
```python
# Symmetric expansion
Linear(64 → 32)                       # Latent to features
LSTM(input_size=32, hidden_size=32)   # Feature reconstruction
LSTM(input_size=32, hidden_size=64)   # Intermediate expansion
LSTM(input_size=64, hidden_size=128)  # Full reconstruction
Linear(128 → 32)                      # Output layer
```

**Key Design Principles**:
- **Bottleneck forcing**: 64-dim latent space compresses information
- **Symmetric architecture**: Encoder/decoder mirror each other
- **Progressive compression**: Gradual dimensionality reduction
- **Temporal coherence**: Maintains sequence relationships

###  Loss Function

**Mean Squared Error** between input and reconstruction:

```math
L = \frac{1}{N} \sum_{i=1}^{N} ||X_i - \hat{X}_i||²
```

Where:
- `N`: Batch size
- `X_i`: Input CWT scalogram
- `X̂_i`: Reconstructed scalogram

**Why MSE?**
- **Pixel-wise comparison**: Sensitive to frequency-specific anomalies
- **Stable gradients**: Well-behaved optimization landscape
- **Interpretable**: Reconstruction error directly relates to anomaly strength

---

## Training Strategy

### Unsupervised Learning Approach

**Core Philosophy**: Train only on noise to learn "normal" patterns.

```python
# Training data composition
training_data = {
    'noise_only': 40000 samples,     # Pure Gaussian noise
    'signal_noise': 0 samples        # NO gravitational waves!
}

# This forces the model to learn noise characteristics
# Gravitational waves will have high reconstruction error
```

###  Data Generation

**Synthetic Noise**:
```python
def generate_ligo_noise(duration=4.0, fs=512):
    """Generate realistic LIGO-like colored noise"""
    # Advanced LIGO sensitivity curve
    frequencies = np.fft.fftfreq(int(duration * fs), 1/fs)
    psd = advanced_ligo_psd(frequencies)
    
    # Generate colored Gaussian noise
    white_noise = np.random.normal(0, 1, int(duration * fs))
    colored_noise = apply_psd_coloring(white_noise, psd)
    
    return colored_noise
```

**Advanced LIGO Power Spectral Density**:
```math
S_n(f) = S_0 \left[ \left(\frac{f}{f_0}\right)^{-4.14} + 5 + 3\left(\frac{f}{f_0}\right)^2 \right]
```

### ⚙️ Training Configuration

**Hyperparameters**:
```yaml
# Model architecture
input_size: 32          # CWT frequency bins
sequence_length: 2048   # Time samples  
hidden_sizes: [128, 64, 32]
latent_size: 64
dropout: 0.2

# Training parameters
batch_size: 64
learning_rate: 1e-3
weight_decay: 1e-5
epochs: 100
early_stopping: 15      # Patience

# Optimization
optimizer: Adam
scheduler: ExponentialLR
decay_factor: 0.95
gradient_clip: 1.0
```

**Training Loop**:
```python
for epoch in range(epochs):
    for batch in noise_dataloader:
        # Forward pass
        reconstruction = model(batch)
        loss = mse_loss(reconstruction, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Learning rate decay
    scheduler.step()
    
    # Early stopping on validation loss plateau
    if validation_loss_plateau(patience=15):
        break
```

---

## Detection Pipeline

###  Anomaly Detection Strategy

Once trained, the model detects gravitational waves through reconstruction error analysis:

```python
def detect_gravitational_waves(model, test_data, threshold):
    """
    Detect gravitational waves using reconstruction error
    
    Args:
        model: Trained LSTM autoencoder
        test_data: CWT scalograms to analyze
        threshold: Detection threshold
    
    Returns:
        predictions: Binary classification (0=noise, 1=GW)
        errors: Reconstruction errors for each sample
    """
    model.eval()
    with torch.no_grad():
        reconstructions = model(test_data)
        errors = torch.mean((test_data - reconstructions) ** 2, dim=(1,2))
        predictions = (errors > threshold).int()
    
    return predictions, errors
```

###  Threshold Optimization

**Precision-Recall Analysis**:
```python
def optimize_threshold(errors, true_labels):
    """Find optimal threshold balancing precision and recall"""
    thresholds = np.linspace(errors.min(), errors.max(), 1000)
    
    precisions, recalls, f1_scores = [], [], []
    
    for threshold in thresholds:
        predictions = (errors > threshold).astype(int)
        
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Find threshold maximizing F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, precisions, recalls
```

###  Operating Points

We define multiple operating points for different use cases:

```python
operating_points = {
    'max_precision': {
        'threshold': 0.95,
        'precision': 100.0,
        'recall': 1.4,
        'use_case': 'Official discoveries'
    },
    'optimal': {
        'threshold': 0.85,
        'precision': 90.6,
        'recall': 67.6,
        'use_case': 'Recommended balance'
    },
    'f1_optimal': {
        'threshold': 0.75,
        'precision': 89.3,
        'recall': 70.4,
        'use_case': 'Research surveys'
    }
}
```

---

## Performance Optimization

###  Computational Efficiency

**CWT Optimization**:
```python
# Use FFT-based CWT for speed
coefficients = scipy.signal.cwt(data, scales, 'morl')

# Parallel processing for batch CWT
from multiprocessing import Pool
with Pool() as pool:
    scalograms = pool.map(compute_cwt, batch_data)
```

**GPU Acceleration**:
```python
# Move model and data to GPU
model = model.cuda()
data = data.cuda()

# Mixed precision training
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
```

###  Memory Management

**Gradient Checkpointing**:
```python
# Trade compute for memory
model = torch.utils.checkpoint.checkpoint_sequential(
    model, segments=4, input=data
)
```

**Batch Size Optimization**:
```python
# Dynamic batch sizing based on GPU memory
max_batch_size = find_max_batch_size(model, gpu_memory=10e9)
dataloader = DataLoader(dataset, batch_size=max_batch_size)
```

###  Hyperparameter Tuning

**Grid Search Strategy**:
```python
param_grid = {
    'hidden_sizes': [[64,32,16], [128,64,32], [256,128,64]],
    'latent_size': [32, 64, 128],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'dropout': [0.1, 0.2, 0.3]
}

best_params = grid_search_cv(model_class, param_grid, 
                            cv_folds=5, metric='f1_score')
```

###  Performance Monitoring

**Training Metrics**:
```python
metrics = {
    'train_loss': [],
    'val_loss': [],
    'reconstruction_error_std': [],
    'gradient_norm': [],
    'learning_rate': []
}

# Log every epoch
wandb.log(metrics)  # Use weights & biases for tracking
```

---

## Algorithm Advantages

###  Compared to Traditional Methods

**vs Matched Filtering**:
- ✅ **Template-free**: No need for pre-computed waveforms
- ✅ **Discovery potential**: Can find unknown signal types
- ✅ **Computational efficiency**: Single model vs thousands of templates
- ⚠️ **Lower SNR sensitivity**: Trade-off for generality

**vs Other ML Approaches**:
- ✅ **Time-frequency aware**: CWT preserves chirp structure
- ✅ **Unsupervised**: No labeled training data required
- ✅ **Temporal modeling**: LSTM captures evolution over time
- ✅ **Interpretable**: Reconstruction error has physical meaning

###  Limitations and Future Work

**Current Limitations**:
- Synthetic data only (reality gap)
- Single detector analysis
- Limited to inspiral-merger signals
- No parameter estimation

**Future Improvements**:
- Real LIGO data validation
- Multi-detector coincidence
- Parameter estimation network
- Transformer architecture exploration

---

##  Mathematical Derivations

### CWT Discretization
For digital implementation, the continuous integral becomes:

```math
W[a,n] = \frac{1}{\sqrt{a}} \sum_{k} h[k] \psi^*\left(\frac{k-n}{a}\right)
```

### LSTM Backpropagation Through Time
Gradients flow through the recurrent connections:

```math
\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W}
```

### Reconstruction Error Statistics
For Gaussian noise input, reconstruction errors follow:

```math
E \sim \chi^2(N) \text{ where } N = \text{effective degrees of freedom}
```

---

*This technical guide provides the mathematical foundation and implementation details for reproducing and extending the CWT-LSTM autoencoder approach to gravitational wave detection.*
