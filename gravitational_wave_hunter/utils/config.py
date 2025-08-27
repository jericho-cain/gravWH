"""
Configuration management for gravitational wave detection.

This module provides configuration classes and utilities for managing
parameters across the gravitational wave detection pipeline.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """
    Configuration class for gravitational wave detection parameters.
    
    This class contains all configurable parameters for data preprocessing,
    model training, and detection. It supports loading from and saving to
    JSON and YAML files.
    
    Attributes:
        Data preprocessing parameters:
            remove_dc_offset: Whether to remove DC offset from data
            apply_bandpass: Whether to apply bandpass filtering
            bandpass_low: Low cutoff frequency for bandpass filter (Hz)
            bandpass_high: High cutoff frequency for bandpass filter (Hz)
            filter_order: Order of the bandpass filter
            
        Whitening parameters:
            apply_whitening: Whether to apply data whitening
            whitening_segment_length: Segment length for PSD estimation (seconds)
            whitening_overlap: Overlap fraction for PSD estimation
            
        Glitch removal parameters:
            remove_glitches: Whether to remove glitches
            glitch_threshold: Threshold for glitch detection (standard deviations)
            glitch_window_size: Window size for glitch detection
            
        Normalization parameters:
            normalize: Whether to normalize data
            normalization_method: Method for normalization ('standard', 'minmax', 'robust')
            
        Model parameters:
            num_classes: Number of output classes
            cnn_filters: Number of CNN filters
            lstm_hidden_size: LSTM hidden layer size
            wavenet_layers: Number of WaveNet layers
            wavenet_channels: Number of WaveNet channels
            transformer_d_model: Transformer model dimension
            transformer_heads: Number of transformer attention heads
            transformer_layers: Number of transformer layers
            autoencoder_encoding_dim: Autoencoder encoding dimension
            autoencoder_layers: Number of autoencoder layers
            
        Training parameters:
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            
        Detection parameters:
            detection_threshold: Threshold for detection
            segment_overlap: Overlap between detection segments
            
    Example:
        >>> config = Config()
        >>> config.bandpass_low = 30.0
        >>> config.save('config.yaml')
        >>> loaded_config = Config.load('config.yaml')
    """
    
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
    wavenet_layers: int = 10
    wavenet_channels: int = 32
    transformer_d_model: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 6
    autoencoder_encoding_dim: int = 128
    autoencoder_layers: int = 4
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Detection parameters
    detection_threshold: float = 0.5
    segment_overlap: float = 0.5
    
    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration file
            
        Supports both JSON (.json) and YAML (.yaml, .yml) formats.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Supported formats: .json, .yaml, .yml"
            )
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Config object with loaded parameters
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Supported formats: .json, .yaml, .yml"
            )
        
        return cls(**config_dict)
    
    def update(self, **kwargs: Any) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Example:
            >>> config = Config()
            >>> config.update(learning_rate=0.001, batch_size=64)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            Config object
        """
        return cls(**config_dict)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate frequency ranges
        if self.bandpass_low >= self.bandpass_high:
            raise ValueError(
                f"bandpass_low ({self.bandpass_low}) must be less than "
                f"bandpass_high ({self.bandpass_high})"
            )
        
        if self.bandpass_low < 0:
            raise ValueError(f"bandpass_low must be non-negative, got {self.bandpass_low}")
        
        # Validate thresholds
        if not 0 <= self.detection_threshold <= 1:
            raise ValueError(
                f"detection_threshold must be between 0 and 1, got {self.detection_threshold}"
            )
        
        if not 0 <= self.segment_overlap < 1:
            raise ValueError(
                f"segment_overlap must be between 0 and 1, got {self.segment_overlap}"
            )
        
        if not 0 < self.validation_split < 1:
            raise ValueError(
                f"validation_split must be between 0 and 1, got {self.validation_split}"
            )
        
        # Validate positive integers
        positive_int_params = [
            'filter_order', 'glitch_window_size', 'num_classes',
            'cnn_filters', 'lstm_hidden_size', 'wavenet_layers',
            'wavenet_channels', 'transformer_d_model', 'transformer_heads',
            'transformer_layers', 'autoencoder_encoding_dim', 'autoencoder_layers',
            'batch_size', 'num_epochs', 'early_stopping_patience'
        ]
        
        for param in positive_int_params:
            value = getattr(self, param)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{param} must be a positive integer, got {value}")
        
        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        # Validate normalization method
        valid_norm_methods = ['standard', 'minmax', 'robust']
        if self.normalization_method not in valid_norm_methods:
            raise ValueError(
                f"normalization_method must be one of {valid_norm_methods}, "
                f"got {self.normalization_method}"
            )
    
    def __repr__(self) -> str:
        """Return string representation of configuration."""
        params = []
        for key, value in asdict(self).items():
            params.append(f"{key}={value}")
        
        return f"Config({', '.join(params)})"


def create_default_config() -> Config:
    """
    Create a default configuration object.
    
    Returns:
        Config object with default parameters
    """
    return Config()


def load_config_from_env(prefix: str = "GW_") -> Config:
    """
    Load configuration from environment variables.
    
    Args:
        prefix: Prefix for environment variable names
        
    Returns:
        Config object with parameters from environment variables
        
    Example:
        >>> # Set environment variables:
        >>> # export GW_LEARNING_RATE=0.001
        >>> # export GW_BATCH_SIZE=64
        >>> config = load_config_from_env()
    """
    import os
    
    config = Config()
    config_dict = asdict(config)
    
    for key in config_dict.keys():
        env_var = f"{prefix}{key.upper()}"
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Try to convert to appropriate type
            original_value = getattr(config, key)
            if isinstance(original_value, bool):
                value = value.lower() in ['true', '1', 'yes', 'on']
            elif isinstance(original_value, int):
                value = int(value)
            elif isinstance(original_value, float):
                value = float(value)
            # String values are used as-is
            
            setattr(config, key, value)
    
    return config
