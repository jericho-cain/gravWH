"""
Helper utilities for gravitational wave detection.

This module provides various utility functions for logging, file operations,
result saving/loading, and other common tasks.
"""

import logging
import json
import pickle
from typing import Any, Dict, Optional, Union
from pathlib import Path
import datetime
import sys
import os

import numpy as np


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging(level='DEBUG', log_file='gw_detection.log')
        >>> logger.info("Starting gravitational wave detection")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[]
    )
    
    # Get root logger
    logger = logging.getLogger('gravitational_wave_hunter')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
        
    Returns:
        Path object for the directory
        
    Example:
        >>> output_dir = ensure_directory('results/experiment_1')
        >>> print(f"Output directory: {output_dir}")
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_results(
    results: Dict[str, Any],
    file_path: Union[str, Path],
    format: str = 'auto',
    include_metadata: bool = True,
) -> None:
    """
    Save results to file in various formats.
    
    Args:
        results: Dictionary containing results to save
        file_path: Path to save file
        format: File format ('json', 'pickle', 'auto')
        include_metadata: Whether to include metadata (timestamp, etc.)
        
    Example:
        >>> results = {'accuracy': 0.95, 'loss': 0.123}
        >>> save_results(results, 'experiment_results.json')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format == 'auto':
        if file_path.suffix.lower() == '.json':
            format = 'json'
        elif file_path.suffix.lower() in ['.pkl', '.pickle']:
            format = 'pickle'
        else:
            format = 'json'  # Default to JSON
    
    # Add metadata if requested
    if include_metadata:
        results_with_metadata = {
            'data': results,
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'version': '0.1.0',
                'format_version': '1.0',
            }
        }
    else:
        results_with_metadata = results
    
    # Save in appropriate format
    if format == 'json':
        with open(file_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=_json_serializer)
    elif format == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(results_with_metadata, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(
    file_path: Union[str, Path],
    format: str = 'auto',
) -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        file_path: Path to file to load
        format: File format ('json', 'pickle', 'auto')
        
    Returns:
        Dictionary containing loaded results
        
    Example:
        >>> results = load_results('experiment_results.json')
        >>> print(f"Accuracy: {results['accuracy']}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    # Determine format
    if format == 'auto':
        if file_path.suffix.lower() == '.json':
            format = 'json'
        elif file_path.suffix.lower() in ['.pkl', '.pickle']:
            format = 'pickle'
        else:
            # Try to determine from content
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                format = 'json'
            except (json.JSONDecodeError, UnicodeDecodeError):
                format = 'pickle'
    
    # Load in appropriate format
    if format == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif format == 'pickle':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Extract data if metadata wrapper exists
    if isinstance(data, dict) and 'data' in data and 'metadata' in data:
        return data['data']
    else:
        return data


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy arrays and other objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        # Let the default serializer handle it
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_experiment_directory(
    base_dir: Union[str, Path] = 'experiments',
    experiment_name: Optional[str] = None,
    timestamp: bool = True,
) -> Path:
    """
    Create a directory for an experiment with optional timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        timestamp: Whether to include timestamp in directory name
        
    Returns:
        Path to the created experiment directory
        
    Example:
        >>> exp_dir = create_experiment_directory('experiments', 'cnn_lstm_test')
        >>> print(f"Experiment directory: {exp_dir}")
    """
    base_dir = Path(base_dir)
    
    if experiment_name is None:
        experiment_name = 'experiment'
    
    if timestamp:
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{experiment_name}_{timestamp_str}"
    else:
        dir_name = experiment_name
    
    experiment_dir = base_dir / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir


def save_model_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    checkpoint_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> Path:
    """
    Save a model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Optional filename (will generate if not provided)
        
    Returns:
        Path to saved checkpoint
    """
    import torch
    
    checkpoint_dir = ensure_directory(checkpoint_dir)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:04d}.pth"
    
    checkpoint_path = checkpoint_dir / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_model_checkpoint(
    checkpoint_path: Union[str, Path],
    model,
    optimizer=None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        Dictionary with checkpoint information
    """
    import torch
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
        
    Example:
        >>> duration_str = format_duration(3661.5)
        >>> print(duration_str)  # "1h 1m 1.5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_seconds = seconds % 3600
        minutes = int(remaining_seconds // 60)
        seconds_remainder = remaining_seconds % 60
        return f"{hours}h {minutes}m {seconds_remainder:.1f}s"


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """
    Format large numbers with appropriate units.
    
    Args:
        number: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted number string
        
    Example:
        >>> formatted = format_number(1234567)
        >>> print(formatted)  # "1.23M"
    """
    if abs(number) >= 1e9:
        return f"{number/1e9:.{precision}f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.{precision}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and reproducibility.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        else:
            cuda_version = None
            gpu_count = 0
            gpu_names = []
    except ImportError:
        torch_version = None
        cuda_available = False
        cuda_version = None
        gpu_count = 0
        gpu_names = []
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'torch_version': torch_version,
        'cuda_available': cuda_available,
        'cuda_version': cuda_version,
        'gpu_count': gpu_count,
        'gpu_names': gpu_names,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    
    return system_info


def validate_config(config) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if hasattr(config, 'validate'):
        config.validate()
    else:
        # Basic validation for dict-like configs
        if hasattr(config, 'get'):
            # Check some common parameters
            if 'learning_rate' in config:
                lr = config.get('learning_rate')
                if not (0 < lr < 1):
                    raise ValueError(f"Invalid learning rate: {lr}")
            
            if 'batch_size' in config:
                batch_size = config.get('batch_size')
                if not (1 <= batch_size <= 1024):
                    raise ValueError(f"Invalid batch size: {batch_size}")


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def print_banner(title: str, width: int = 80, char: str = '=') -> None:
    """
    Print a banner with title.
    
    Args:
        title: Title text
        width: Banner width
        char: Character to use for banner
    """
    print(char * width)
    print(f"{title:^{width}}")
    print(char * width)
