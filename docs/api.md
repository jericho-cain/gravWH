# API Reference

This document provides a comprehensive reference for the Gravitational Wave Hunter API.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Loading](#data-loading)
3. [Neural Network Models](#neural-network-models)
4. [Signal Processing](#signal-processing)
5. [Utilities](#utilities)
6. [Visualization](#visualization)
7. [Command Line Interface](#command-line-interface)

## Core Classes

### GWDetector

The main interface for gravitational wave detection.

```python
class GWDetector:
    def __init__(
        self,
        model_type: str = 'cnn_lstm',
        sample_rate: int = 4096,
        segment_length: float = 8.0,
        device: str = 'auto',
        config: Optional[Config] = None,
    ) -> None
```

**Parameters:**
- `model_type`: Neural network architecture ('cnn_lstm', 'wavenet', 'transformer', 'autoencoder')
- `sample_rate`: Sample rate of input data in Hz
- `segment_length`: Length of input segments in seconds
- `device`: Computation device ('cpu', 'cuda', 'auto')
- `config`: Configuration object for advanced settings

**Methods:**

#### `load_pretrained(model_path)`
Load a pre-trained model from file.

```python
def load_pretrained(self, model_path: Union[str, Path]) -> None
```

**Parameters:**
- `model_path`: Path to the saved model file

**Example:**
```python
detector = GWDetector(model_type='cnn_lstm')
detector.load_pretrained('models/gw_detector_v1.pth')
```

#### `detect(strain_data, threshold, overlap, preprocess)`
Detect gravitational waves in strain data.

```python
def detect(
    self,
    strain_data: np.ndarray,
    threshold: float = 0.5,
    overlap: float = 0.5,
    preprocess: bool = True,
) -> Dict[str, Union[np.ndarray, List[Tuple[float, float]]]]
```

**Parameters:**
- `strain_data`: Input strain data as 1D numpy array
- `threshold`: Detection threshold (0-1)
- `overlap`: Overlap fraction between segments (0-1)
- `preprocess`: Whether to apply preprocessing

**Returns:**
Dictionary containing:
- `'detections'`: List of (start_time, end_time) tuples
- `'scores'`: Detection scores for each segment
- `'times'`: Time stamps for each segment
- `'segments'`: Processed input segments

**Example:**
```python
results = detector.detect(strain_data, threshold=0.7)
detections = results['detections']
scores = results['scores']
```

#### `train(train_loader, val_loader, num_epochs, learning_rate, save_path)`
Train the detection model.

```python
def train(
    self,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[float]]
```

**Parameters:**
- `train_loader`: PyTorch DataLoader for training data
- `val_loader`: Optional DataLoader for validation data
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `save_path`: Optional path to save best model

**Returns:**
Dictionary with training history (losses, metrics)

**Example:**
```python
history = detector.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=1e-3,
    save_path='best_model.pth'
)
```

#### `evaluate(test_loader)`
Evaluate model performance on test data.

```python
def evaluate(self, test_loader: DataLoader) -> Dict[str, float]
```

**Parameters:**
- `test_loader`: DataLoader containing test data

**Returns:**
Dictionary with evaluation metrics

**Example:**
```python
metrics = detector.evaluate(test_loader)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

## Data Loading

### LIGO Data Functions

#### `load_ligo_data(detector, start_time, duration, sample_rate, cache_dir, preprocess, config)`
Load LIGO strain data for a specified time period.

```python
def load_ligo_data(
    detector: str,
    start_time: int,
    duration: int,
    sample_rate: int = 4096,
    cache_dir: Optional[Union[str, Path]] = None,
    preprocess: bool = True,
    config: Optional[Config] = None,
) -> np.ndarray
```

**Parameters:**
- `detector`: Detector name ('H1', 'L1', 'V1')
- `start_time`: GPS start time
- `duration`: Duration in seconds
- `sample_rate`: Desired sample rate in Hz
- `cache_dir`: Directory to cache downloaded data
- `preprocess`: Whether to apply standard preprocessing
- `config`: Configuration object for preprocessing parameters

**Returns:**
Strain data as numpy array

**Example:**
```python
data = load_ligo_data('H1', 1126259446, 4096, sample_rate=4096)
```

#### `load_event_data(event_name, detectors, duration, sample_rate, preprocess, config)`
Load gravitational wave event data from known detections.

```python
def load_event_data(
    event_name: str,
    detectors: Optional[List[str]] = None,
    duration: int = 32,
    sample_rate: int = 4096,
    preprocess: bool = True,
    config: Optional[Config] = None,
) -> Dict[str, np.ndarray]
```

**Parameters:**
- `event_name`: Name of the event (e.g., 'GW150914', 'GW170817')
- `detectors`: List of detectors to load data from
- `duration`: Duration around event time in seconds
- `sample_rate`: Desired sample rate in Hz
- `preprocess`: Whether to apply standard preprocessing
- `config`: Configuration object

**Returns:**
Dictionary mapping detector names to strain data arrays

**Example:**
```python
event_data = load_event_data('GW150914', ['H1', 'L1'])
h1_data = event_data['H1']
l1_data = event_data['L1']
```

### Dataset Classes

#### `GWDataset`
PyTorch Dataset for gravitational wave data.

```python
class GWDataset(Dataset):
    def __init__(
        self,
        data_files: Union[List[Union[str, Path]], List[np.ndarray]],
        labels: Optional[List[int]] = None,
        segment_length: float = 8.0,
        sample_rate: int = 4096,
        overlap: float = 0.5,
        augment: bool = False,
        config: Optional[Config] = None,
    ) -> None
```

**Parameters:**
- `data_files`: List of paths to data files or numpy arrays
- `labels`: Optional labels for supervised learning
- `segment_length`: Length of each training segment in seconds
- `sample_rate`: Sample rate of the data
- `overlap`: Overlap between consecutive segments
- `augment`: Whether to apply data augmentation
- `config`: Configuration object for preprocessing

**Example:**
```python
dataset = GWDataset(
    data_files=['data1.npy', 'data2.npy'],
    labels=[0, 1],  # 0=noise, 1=signal
    segment_length=8.0,
    sample_rate=4096
)
```

#### `create_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory)`
Create a PyTorch DataLoader from a GWDataset.

```python
def create_dataloader(
    dataset: GWDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader
```

**Parameters:**
- `dataset`: GWDataset instance
- `batch_size`: Number of samples per batch
- `shuffle`: Whether to shuffle the data
- `num_workers`: Number of subprocesses for data loading
- `pin_memory`: Whether to pin memory for faster GPU transfer

**Returns:**
PyTorch DataLoader

**Example:**
```python
loader = create_dataloader(dataset, batch_size=64)
for batch in loader:
    # Process batch
    pass
```

## Neural Network Models

All models inherit from the `BaseGWModel` class and implement the following interface:

### BaseGWModel

```python
class BaseGWModel(nn.Module, ABC):
    def __init__(
        self,
        input_length: int,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ) -> None
```

**Methods:**

#### `forward(x)`
Forward pass of the model (abstract method).

#### `get_model_info()`
Get model information and metadata.

```python
def get_model_info(self) -> Dict[str, Any]
```

**Returns:**
Dictionary containing model information

#### `count_parameters(trainable_only)`
Count the number of parameters in the model.

```python
def count_parameters(self, trainable_only: bool = False) -> int
```

#### `predict_proba(x)`
Get prediction probabilities.

```python
def predict_proba(self, x: torch.Tensor) -> torch.Tensor
```

#### `predict(x, threshold)`
Get discrete predictions.

```python
def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor
```

### CNNLSTM

CNN-LSTM hybrid model for gravitational wave detection.

```python
class CNNLSTM(BaseGWModel):
    def __init__(
        self,
        input_length: int,
        num_filters: int = 64,
        lstm_hidden_size: int = 128,
        num_classes: int = 1,
        num_conv_layers: int = 4,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_bidirectional_lstm: bool = True,
    ) -> None
```

**Parameters:**
- `input_length`: Length of input time series
- `num_filters`: Number of filters in convolutional layers
- `lstm_hidden_size`: Hidden size of LSTM layers
- `num_classes`: Number of output classes
- `num_conv_layers`: Number of convolutional layers
- `num_lstm_layers`: Number of LSTM layers
- `dropout_rate`: Dropout rate for regularization
- `use_batch_norm`: Whether to use batch normalization
- `use_bidirectional_lstm`: Whether to use bidirectional LSTM

**Example:**
```python
model = CNNLSTM(
    input_length=32768,
    num_filters=64,
    lstm_hidden_size=128,
    num_classes=1
)
```

### WaveNet

WaveNet model with dilated convolutions.

```python
class WaveNet(BaseGWModel):
    def __init__(
        self,
        input_length: int,
        num_layers: int = 10,
        num_channels: int = 32,
        num_classes: int = 1,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        use_skip_connections: bool = True,
    ) -> None
```

**Parameters:**
- `input_length`: Length of input time series
- `num_layers`: Number of convolutional layers
- `num_channels`: Number of channels in each layer
- `num_classes`: Number of output classes
- `kernel_size`: Size of convolutional kernels
- `dropout_rate`: Dropout rate for regularization
- `use_skip_connections`: Whether to use skip connections

**Example:**
```python
model = WaveNet(
    input_length=32768,
    num_layers=10,
    num_channels=32
)
```

### GWTransformer

Transformer model for gravitational wave detection.

```python
class GWTransformer(BaseGWModel):
    def __init__(
        self,
        input_length: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_classes: int = 1,
        d_ff: Optional[int] = None,
        dropout_rate: float = 0.1,
        max_seq_length: int = 50000,
    ) -> None
```

**Parameters:**
- `input_length`: Length of input time series
- `d_model`: Model dimension
- `num_heads`: Number of attention heads
- `num_layers`: Number of transformer layers
- `num_classes`: Number of output classes
- `d_ff`: Feed-forward network dimension
- `dropout_rate`: Dropout rate
- `max_seq_length`: Maximum sequence length for positional encoding

**Example:**
```python
model = GWTransformer(
    input_length=32768,
    d_model=512,
    num_heads=8,
    num_layers=6
)
```

### GWAutoencoder

Autoencoder for unsupervised gravitational wave detection.

```python
class GWAutoencoder(BaseGWModel):
    def __init__(
        self,
        input_length: int,
        encoding_dim: int = 128,
        num_layers: int = 4,
        num_filters: int = 32,
        dropout_rate: float = 0.1,
        use_variational: bool = False,
    ) -> None
```

**Parameters:**
- `input_length`: Length of input time series
- `encoding_dim`: Dimension of the encoded representation
- `num_layers`: Number of encoder/decoder layers
- `num_filters`: Base number of filters
- `dropout_rate`: Dropout rate for regularization
- `use_variational`: Whether to use variational autoencoder

**Methods:**

#### `encode(x)`
Encode input to latent representation.

#### `decode(z)`
Decode latent representation to output.

#### `anomaly_score(x)`
Compute anomaly scores for input samples.

**Example:**
```python
model = GWAutoencoder(
    input_length=32768,
    encoding_dim=128,
    num_layers=4
)

# For anomaly detection
scores = model.anomaly_score(test_data)
```

## Signal Processing

### Preprocessing Pipeline

#### `preprocess_strain_data(strain_data, sample_rate, config)`
Apply comprehensive preprocessing pipeline to strain data.

```python
def preprocess_strain_data(
    strain_data: np.ndarray,
    sample_rate: int = 4096,
    config: Optional[Config] = None,
) -> np.ndarray
```

**Parameters:**
- `strain_data`: Input strain data as 1D numpy array
- `sample_rate`: Sample rate of the data in Hz
- `config`: Configuration object with preprocessing parameters

**Returns:**
Preprocessed strain data

**Example:**
```python
processed = preprocess_strain_data(raw_data, sample_rate=4096)
```

### Individual Processing Functions

#### `bandpass_filter(data, lowcut, highcut, sample_rate, order)`
Apply a Butterworth bandpass filter to the data.

```python
def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    sample_rate: int,
    order: int = 4,
) -> np.ndarray
```

**Example:**
```python
filtered = bandpass_filter(data, 20, 2000, 4096, order=6)
```

#### `whiten_data(data, sample_rate, segment_length, overlap, fftlength)`
Whiten gravitational wave strain data by normalizing by the power spectral density.

```python
def whiten_data(
    data: np.ndarray,
    sample_rate: int,
    segment_length: int = 4,
    overlap: float = 0.5,
    fftlength: Optional[int] = None,
) -> np.ndarray
```

**Example:**
```python
whitened = whiten_data(strain_data, sample_rate=4096)
```

#### `remove_glitches(data, threshold, window_size)`
Remove transient glitches from strain data.

```python
def remove_glitches(
    data: np.ndarray,
    threshold: float = 20.0,
    window_size: int = 1024,
) -> np.ndarray
```

**Example:**
```python
clean_data = remove_glitches(noisy_data, threshold=15.0)
```

#### `normalize_data(data, method)`
Normalize strain data using specified method.

```python
def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
) -> np.ndarray
```

**Parameters:**
- `method`: Normalization method ('standard', 'minmax', 'robust')

**Example:**
```python
normalized = normalize_data(data, method='standard')
```

## Utilities

### Configuration

#### `Config`
Configuration class for gravitational wave detection parameters.

```python
@dataclass
class Config:
    # Data preprocessing parameters
    remove_dc_offset: bool = True
    apply_bandpass: bool = True
    bandpass_low: float = 20.0
    bandpass_high: float = 2000.0
    filter_order: int = 6
    
    # Whitening parameters
    apply_whitening: bool = True
    whitening_segment_length: int = 4
    whitening_overlap: float = 0.5
    
    # Glitch removal parameters
    remove_glitches: bool = True
    glitch_threshold: float = 20.0
    glitch_window_size: int = 1024
    
    # Normalization parameters
    normalize: bool = True
    normalization_method: str = 'standard'
    
    # Model parameters
    num_classes: int = 2
    cnn_filters: int = 64
    lstm_hidden_size: int = 128
    # ... (see full class for all parameters)
```

**Methods:**

#### `save(file_path)`
Save configuration to file.

```python
def save(self, file_path: Union[str, Path]) -> None
```

#### `load(file_path)`
Load configuration from file.

```python
@classmethod
def load(cls, file_path: Union[str, Path]) -> 'Config'
```

#### `validate()`
Validate configuration parameters.

**Example:**
```python
config = Config()
config.bandpass_low = 30.0
config.save('config.yaml')

loaded_config = Config.load('config.yaml')
loaded_config.validate()
```

### Metrics

#### `calculate_detection_metrics(y_true, y_pred, threshold, average)`
Calculate comprehensive detection performance metrics.

```python
def calculate_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = 'binary',
) -> Dict[str, float]
```

**Parameters:**
- `y_true`: True binary labels (0=noise, 1=signal)
- `y_pred`: Predicted probabilities or binary predictions
- `threshold`: Decision threshold for converting probabilities to predictions
- `average`: Averaging strategy for multi-class

**Returns:**
Dictionary containing various performance metrics

**Example:**
```python
metrics = calculate_detection_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

#### `detection_statistics(detections, true_events, coincidence_window, total_duration)`
Calculate detection statistics comparing detected events to true events.

```python
def detection_statistics(
    detections: List[Tuple[float, float]],
    true_events: List[Tuple[float, float]],
    coincidence_window: float = 1.0,
    total_duration: Optional[float] = None,
) -> Dict[str, Union[int, float]]
```

**Example:**
```python
stats = detection_statistics(detections, true_events, total_duration=100.0)
print(f"Detection efficiency: {stats['detection_efficiency']:.3f}")
```

### Helper Functions

#### `setup_logging(level, log_file, format_string)`
Set up logging configuration.

```python
def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger
```

#### `save_results(results, file_path, format, include_metadata)`
Save results to file in various formats.

```python
def save_results(
    results: Dict[str, Any],
    file_path: Union[str, Path],
    format: str = 'auto',
    include_metadata: bool = True,
) -> None
```

#### `seed_everything(seed)`
Set random seeds for reproducibility.

```python
def seed_everything(seed: int = 42) -> None
```

## Visualization

### Plotting Functions

#### `plot_strain_data(strain_data, sample_rate, time_offset, title, figsize, save_path, show_plot)`
Plot gravitational wave strain data.

```python
def plot_strain_data(
    strain_data: np.ndarray,
    sample_rate: int = 4096,
    time_offset: float = 0.0,
    title: str = "Gravitational Wave Strain Data",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure
```

**Example:**
```python
fig = plot_strain_data(strain, sample_rate=4096)
```

#### `plot_spectrogram(strain_data, sample_rate, nperseg, noverlap, title, figsize, freq_range, save_path, show_plot)`
Plot spectrogram of gravitational wave strain data.

```python
def plot_spectrogram(
    strain_data: np.ndarray,
    sample_rate: int = 4096,
    nperseg: int = 512,
    noverlap: Optional[int] = None,
    title: str = "Strain Data Spectrogram",
    figsize: Tuple[int, int] = (12, 8),
    freq_range: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure
```

**Example:**
```python
fig = plot_spectrogram(strain, freq_range=(20, 1000))
```

#### `plot_detection_results(strain_data, detection_results, sample_rate, time_offset, title, figsize, save_path, show_plot)`
Plot detection results overlaid on strain data.

```python
def plot_detection_results(
    strain_data: np.ndarray,
    detection_results: Dict,
    sample_rate: int = 4096,
    time_offset: float = 0.0,
    title: str = "Gravitational Wave Detection Results",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure
```

**Example:**
```python
results = detector.detect(strain_data)
fig = plot_detection_results(strain_data, results)
```

#### `plot_training_history(history, title, figsize, save_path, show_plot)`
Plot model training history.

```python
def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure
```

**Example:**
```python
history = detector.train(train_loader, val_loader)
fig = plot_training_history(history)
```

## Command Line Interface

The package provides command-line scripts for common tasks:

### Training Script

```bash
gw-train --data-dir /path/to/data --labels-file labels.json --model-type cnn_lstm --num-epochs 100
```

**Options:**
- `--data-dir`: Directory containing training data files
- `--labels-file`: JSON file containing labels
- `--model-type`: Type of model ('cnn_lstm', 'wavenet', 'transformer', 'autoencoder')
- `--num-epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--output-dir`: Directory for outputs
- `--config`: Path to configuration file

### Detection Script

```bash
gw-detect --model-path model.pth --data-file data.npy --threshold 0.7 --save-plots
```

**Options:**
- `--model-path`: Path to trained model
- `--data-file`: Path to data file
- `--ligo-data`: Load LIGO data (detector, start_time, duration)
- `--event-data`: Load event data (event_name, detectors)
- `--threshold`: Detection threshold
- `--output-dir`: Output directory
- `--save-plots`: Save detection plots

### Evaluation Script

```bash
gw-evaluate --model-path model.pth --test-data /path/to/test --test-labels labels.json
```

**Options:**
- `--model-path`: Path to trained model
- `--test-data`: Directory with test data
- `--test-labels`: Test labels file
- `--compare-models`: Compare multiple models
- `--save-plots`: Save evaluation plots
- `--detailed-analysis`: Perform detailed analysis

## Error Handling

### Common Exceptions

#### `ValueError`
Raised for invalid input parameters or data.

#### `FileNotFoundError`
Raised when required files are not found.

#### `RuntimeError`
Raised for runtime errors during training or detection.

#### `ImportError`
Raised when optional dependencies are not available.

### Example Error Handling

```python
try:
    detector = GWDetector(model_type='cnn_lstm')
    detector.load_pretrained('model.pth')
    results = detector.detect(strain_data)
except FileNotFoundError:
    print("Model file not found")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Detection failed: {e}")
```

## Best Practices

### Performance

1. **Use GPU acceleration** when available
2. **Batch processing** for multiple files
3. **Memory management** for large datasets
4. **Preprocessing caching** for repeated use

### Data Handling

1. **Validate input data** before processing
2. **Use appropriate data types** (float32 vs float64)
3. **Handle missing data** gracefully
4. **Normalize inputs** for stable training

### Model Training

1. **Monitor training metrics** to avoid overfitting
2. **Use validation sets** for model selection
3. **Save checkpoints** during long training runs
4. **Tune hyperparameters** systematically

### Code Organization

1. **Use configuration objects** for parameter management
2. **Log important events** for debugging
3. **Handle exceptions** appropriately
4. **Document usage patterns** with examples

---

This API reference provides comprehensive documentation for the Gravitational Wave Hunter framework. For additional examples and tutorials, see the [documentation](README.md) and [Jupyter notebooks](notebooks/).
