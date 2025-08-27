"""
Core plotting functions for gravitational wave visualization.

This module provides essential plotting capabilities for visualizing
gravitational wave data, detection results, and model performance.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some interactive features will be limited.")

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_strain_data(
    strain_data: np.ndarray,
    sample_rate: int = 4096,
    time_offset: float = 0.0,
    title: str = "Gravitational Wave Strain Data",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure:
    """
    Plot gravitational wave strain data.
    
    Args:
        strain_data: 1D array of strain values
        sample_rate: Sampling rate in Hz
        time_offset: Time offset for x-axis
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> strain = np.random.randn(4096 * 10)
        >>> fig = plot_strain_data(strain, sample_rate=4096)
    """
    # Create time axis
    duration = len(strain_data) / sample_rate
    time = np.linspace(time_offset, time_offset + duration, len(strain_data))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot strain data
    ax.plot(time, strain_data, 'b-', linewidth=0.5, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Strain', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add some statistics
    stats_text = (
        f'Duration: {duration:.2f} s\n'
        f'Sample Rate: {sample_rate} Hz\n'
        f'RMS: {np.sqrt(np.mean(strain_data**2)):.2e}\n'
        f'Peak: {np.max(np.abs(strain_data)):.2e}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


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
) -> Figure:
    """
    Plot spectrogram of gravitational wave strain data.
    
    Args:
        strain_data: 1D array of strain values
        sample_rate: Sampling rate in Hz
        nperseg: Length of each segment for STFT
        noverlap: Number of points to overlap between segments
        title: Plot title
        figsize: Figure size
        freq_range: Frequency range to plot (min_freq, max_freq)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    from scipy import signal as scipy_signal
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Compute spectrogram
    frequencies, times, Sxx = scipy_signal.spectrogram(
        strain_data, 
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann'
    )
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spectrogram
    im = ax.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='viridis')
    
    # Set frequency range if specified
    if freq_range:
        ax.set_ylim(freq_range)
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power Spectral Density (dB)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


def plot_detection_results(
    strain_data: np.ndarray,
    detection_results: Dict,
    sample_rate: int = 4096,
    time_offset: float = 0.0,
    title: str = "Gravitational Wave Detection Results",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure:
    """
    Plot detection results overlaid on strain data.
    
    Args:
        strain_data: Original strain data
        detection_results: Results from detector.detect() method
        sample_rate: Sampling rate in Hz
        time_offset: Time offset for plotting
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    # Create time axis
    duration = len(strain_data) / sample_rate
    time = np.linspace(time_offset, time_offset + duration, len(strain_data))
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Strain data with detections
    ax1.plot(time, strain_data, 'b-', linewidth=0.5, alpha=0.7, label='Strain Data')
    
    # Highlight detections
    detections = detection_results.get('detections', [])
    for i, (start_time, end_time) in enumerate(detections):
        ax1.axvspan(start_time + time_offset, end_time + time_offset, 
                   alpha=0.3, color='red', label='Detection' if i == 0 else "")
    
    ax1.set_ylabel('Strain', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detection scores
    if 'scores' in detection_results and 'times' in detection_results:
        detection_times = detection_results['times'] + time_offset
        scores = detection_results['scores']
        
        ax2.plot(detection_times, scores, 'g-', linewidth=2, label='Detection Score')
        
        # Add threshold line if we can infer it
        if detections:
            # Find minimum score among detections
            threshold = min(scores[np.searchsorted(detection_results['times'], 
                                                 np.array(detections)[:, 0] - time_offset)])
            ax2.axhline(y=threshold, color='red', linestyle='--', 
                       label=f'Threshold (~{threshold:.3f})')
        
        ax2.set_ylabel('Detection Score', fontsize=12)
        ax2.set_title('Detection Confidence Over Time', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Spectrogram
    from scipy import signal as scipy_signal
    
    frequencies, times_spec, Sxx = scipy_signal.spectrogram(
        strain_data, 
        fs=sample_rate,
        nperseg=512,
        noverlap=256,
        window='hann'
    )
    
    times_spec += time_offset
    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    
    im = ax3.pcolormesh(times_spec, frequencies, Sxx_db, 
                       shading='gouraud', cmap='viridis', alpha=0.8)
    
    # Overlay detection regions on spectrogram
    for start_time, end_time in detections:
        ax3.axvspan(start_time + time_offset, end_time + time_offset, 
                   alpha=0.3, color='red')
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Frequency (Hz)', fontsize=12)
    ax3.set_title('Spectrogram with Detections', fontsize=12)
    ax3.set_ylim(0, 1000)  # Focus on interesting frequency range
    
    # Add colorbar for spectrogram
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('PSD (dB)', fontsize=10)
    
    plt.tight_layout()
    
    # Add detection summary
    summary_text = (
        f'Total Detections: {len(detections)}\n'
        f'Data Duration: {duration:.2f} s\n'
        f'Detection Rate: {len(detections)/duration*60:.1f} per minute'
    )
    
    fig.text(0.02, 0.02, summary_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure:
    """
    Plot model training history.
    
    Args:
        history: Dictionary with training metrics
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    # Determine what metrics are available
    available_metrics = list(history.keys())
    
    # Create subplots based on available metrics
    if 'train_loss' in available_metrics and 'val_loss' in available_metrics:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        axes = [ax1, ax2, ax3, ax4]
    elif 'train_loss' in available_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        axes = [ax1]
    
    plot_idx = 0
    
    # Plot loss
    if 'train_loss' in available_metrics:
        ax = axes[plot_idx]
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in available_metrics:
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot accuracy
    if 'train_accuracy' in available_metrics and plot_idx < len(axes):
        ax = axes[plot_idx]
        epochs = range(1, len(history['train_accuracy']) + 1)
        
        ax.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in available_metrics:
            ax.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot learning rate if available
    if 'learning_rate' in available_metrics and plot_idx < len(axes):
        ax = axes[plot_idx]
        epochs = range(1, len(history['learning_rate']) + 1)
        
        ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot additional metrics
    other_metrics = [k for k in available_metrics 
                    if k not in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'learning_rate']]
    
    if other_metrics and plot_idx < len(axes):
        ax = axes[plot_idx]
        
        for metric in other_metrics[:3]:  # Plot up to 3 additional metrics
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], label=metric.replace('_', ' ').title(), linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Additional Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


def plot_model_architecture(
    model,
    input_shape: Tuple[int, ...],
    title: str = "Model Architecture",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure:
    """
    Plot a visual representation of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input data
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get model summary information
    model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
    
    # Create a simple visualization
    y_pos = 0.9
    x_center = 0.5
    layer_height = 0.08
    
    # Title
    ax.text(x_center, 0.95, title, ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Model type and parameters
    model_type = model_info.get('model_type', type(model).__name__)
    num_params = model_info.get('num_parameters', model.count_parameters() if hasattr(model, 'count_parameters') else 'Unknown')
    
    ax.text(x_center, 0.88, f'{model_type}', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.text(x_center, 0.84, f'Parameters: {num_params:,}' if isinstance(num_params, int) else f'Parameters: {num_params}',
            ha='center', va='center', fontsize=12)
    
    # Draw layers
    layer_names = []
    layer_params = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_names.append(f"{name}: {type(module).__name__}")
            if hasattr(module, 'weight') and module.weight is not None:
                layer_params.append(module.weight.numel())
            else:
                layer_params.append(0)
    
    # Limit number of layers shown
    max_layers = 15
    if len(layer_names) > max_layers:
        layer_names = layer_names[:max_layers] + ['...']
        layer_params = layer_params[:max_layers] + [0]
    
    for i, (name, params) in enumerate(zip(layer_names, layer_params)):
        y = y_pos - i * layer_height
        
        # Draw layer box
        width = 0.6
        height = 0.05
        rect = patches.Rectangle((x_center - width/2, y - height/2), width, height,
                               linewidth=1, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        
        # Add layer text
        display_name = name if len(name) < 40 else name[:37] + '...'
        ax.text(x_center, y, display_name, ha='center', va='center', fontsize=8)
        
        # Add parameter count if significant
        if params > 0:
            ax.text(x_center + width/2 + 0.05, y, f'{params:,}', 
                   ha='left', va='center', fontsize=7)
    
    # Input/Output information
    ax.text(0.05, 0.1, f'Input Shape: {input_shape}', ha='left', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    output_shape = model_info.get('num_classes', 'Unknown')
    ax.text(0.95, 0.1, f'Output Classes: {output_shape}', ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


def create_comparison_plot(
    data_dict: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = "Data Comparison",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Figure:
    """
    Create a comparison plot of multiple data arrays.
    
    Args:
        data_dict: Dictionary mapping names to data arrays
        labels: Optional list of labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
        show_plot: Whether to display
        
    Returns:
        Matplotlib Figure object
    """
    n_plots = len(data_dict)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_plots))
    
    for i, (name, data) in enumerate(data_dict.items()):
        ax = axes[i]
        time = np.arange(len(data)) / 4096  # Assume 4096 Hz
        
        ax.plot(time, data, color=colors[i], linewidth=1, alpha=0.8)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add RMS value
        rms = np.sqrt(np.mean(data**2))
        ax.text(0.02, 0.98, f'RMS: {rms:.2e}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig
