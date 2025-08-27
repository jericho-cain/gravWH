"""
Main detector class for gravitational wave detection using deep learning.

This module provides the primary interface for gravitational wave detection,
including model loading, training, and inference capabilities.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .models.cnn_lstm import CNNLSTM
from .models.wavenet import WaveNet
from .models.transformer import GWTransformer
from .models.autoencoder import GWAutoencoder
from .signal_processing.preprocessing import preprocess_strain_data
from .visualization.plotting import plot_detection_results
from .utils.config import Config
from .utils.metrics import calculate_detection_metrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GWDetector:
    """
    Main gravitational wave detector using deep learning models.
    
    This class provides a unified interface for gravitational wave detection
    using various neural network architectures. It handles model loading,
    training, inference, and result visualization.
    
    Args:
        model_type: Type of neural network model to use.
            Options: 'cnn_lstm', 'wavenet', 'transformer', 'autoencoder'
        sample_rate: Sampling rate of input data in Hz
        segment_length: Length of input segments in seconds
        device: PyTorch device for computation ('cpu', 'cuda', 'auto')
        config: Optional configuration object for advanced settings
        
    Example:
        >>> detector = GWDetector(model_type='cnn_lstm', sample_rate=4096)
        >>> detector.load_pretrained('models/gw_detector_v1.pth')
        >>> detections = detector.detect(strain_data)
    """
    
    SUPPORTED_MODELS = {
        'cnn_lstm': CNNLSTM,
        'wavenet': WaveNet,
        'transformer': GWTransformer,
        'autoencoder': GWAutoencoder,
    }
    
    def __init__(
        self,
        model_type: str = 'cnn_lstm',
        sample_rate: int = 4096,
        segment_length: float = 8.0,
        device: str = 'auto',
        config: Optional[Config] = None,
    ) -> None:
        """Initialize the gravitational wave detector."""
        self.model_type = model_type
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.config = config or Config()
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Validate model type
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(self.SUPPORTED_MODELS.keys())}"
            )
            
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Training state
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.is_trained = False
        
    def _create_model(self) -> nn.Module:
        """Create the neural network model based on model_type."""
        model_class = self.SUPPORTED_MODELS[self.model_type]
        
        # Calculate input dimensions
        input_length = int(self.sample_rate * self.segment_length)
        
        # Model-specific parameters
        if self.model_type == 'cnn_lstm':
            return model_class(
                input_length=input_length,
                num_filters=self.config.cnn_filters,
                lstm_hidden_size=self.config.lstm_hidden_size,
                num_classes=self.config.num_classes,
            )
        elif self.model_type == 'wavenet':
            return model_class(
                input_length=input_length,
                num_layers=self.config.wavenet_layers,
                num_channels=self.config.wavenet_channels,
                num_classes=self.config.num_classes,
            )
        elif self.model_type == 'transformer':
            return model_class(
                input_length=input_length,
                d_model=self.config.transformer_d_model,
                num_heads=self.config.transformer_heads,
                num_layers=self.config.transformer_layers,
                num_classes=self.config.num_classes,
            )
        elif self.model_type == 'autoencoder':
            return model_class(
                input_length=input_length,
                encoding_dim=self.config.autoencoder_encoding_dim,
                num_layers=self.config.autoencoder_layers,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_pretrained(self, model_path: Union[str, Path]) -> None:
        """
        Load a pre-trained model from file.
        
        Args:
            model_path: Path to the saved model file (.pth or .pt)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            self.is_trained = True
            logger.info(f"Successfully loaded model from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save the current model to file.
        
        Args:
            save_path: Path where to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'sample_rate': self.sample_rate,
            'segment_length': self.segment_length,
            'config': self.config.__dict__ if self.config else None,
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def detect(
        self,
        strain_data: np.ndarray,
        threshold: float = 0.5,
        overlap: float = 0.5,
        preprocess: bool = True,
    ) -> Dict[str, Union[np.ndarray, List[Tuple[float, float]]]]:
        """
        Detect gravitational waves in strain data.
        
        Args:
            strain_data: Input strain data as 1D numpy array
            threshold: Detection threshold (0-1 for classification models)
            overlap: Overlap fraction between segments (0-1)
            preprocess: Whether to apply preprocessing to the data
            
        Returns:
            Dictionary containing:
                - 'detections': List of (start_time, end_time) tuples for detections
                - 'scores': Detection scores for each segment
                - 'times': Time stamps for each segment
                - 'segments': Processed input segments
                
        Raises:
            RuntimeError: If model is not trained/loaded
            ValueError: If input data is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained or loaded before detection. "
                "Use load_pretrained() or train() first."
            )
            
        if strain_data.ndim != 1:
            raise ValueError("Input strain_data must be 1D array")
            
        self.model.eval()
        
        # Preprocess data if requested
        if preprocess:
            strain_data = preprocess_strain_data(
                strain_data, 
                sample_rate=self.sample_rate,
                config=self.config
            )
        
        # Create overlapping segments
        segments, times = self._create_segments(strain_data, overlap)
        
        # Run inference
        scores = []
        detections = []
        
        with torch.no_grad():
            for i, segment in enumerate(segments):
                # Convert to tensor and add batch dimension
                input_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                
                # Get model prediction
                if self.model_type == 'autoencoder':
                    # For autoencoder, use reconstruction error
                    reconstructed = self.model(input_tensor)
                    score = torch.mean((input_tensor - reconstructed) ** 2).item()
                else:
                    # For classification models
                    output = self.model(input_tensor)
                    if output.shape[1] > 1:  # Multi-class
                        score = torch.softmax(output, dim=1)[0, 1].item()  # Probability of GW class
                    else:  # Binary
                        score = torch.sigmoid(output)[0, 0].item()
                
                scores.append(score)
                
                # Check if detection threshold is exceeded
                if score > threshold:
                    start_time = times[i]
                    end_time = start_time + self.segment_length
                    detections.append((start_time, end_time))
        
        return {
            'detections': detections,
            'scores': np.array(scores),
            'times': np.array(times),
            'segments': segments,
        }
    
    def _create_segments(
        self, 
        data: np.ndarray, 
        overlap: float
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Create overlapping segments from continuous data.
        
        Args:
            data: Input data array
            overlap: Overlap fraction between segments
            
        Returns:
            Tuple of (segments_list, time_stamps_list)
        """
        segment_samples = int(self.sample_rate * self.segment_length)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        times = []
        
        for start_idx in range(0, len(data) - segment_samples + 1, hop_samples):
            end_idx = start_idx + segment_samples
            segment = data[start_idx:end_idx]
            
            # Ensure segment has correct length
            if len(segment) == segment_samples:
                segments.append(segment)
                times.append(start_idx / self.sample_rate)
        
        return segments, times
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the gravitational wave detection model.
        
        Args:
            train_loader: PyTorch DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_path: Optional path to save best model
            
        Returns:
            Dictionary with training history (losses, metrics)
        """
        # Setup training components
        if self.model_type == 'autoencoder':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                # Save best model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(save_path)
                    
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )
        
        self.is_trained = True
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            if self.model_type == 'autoencoder':
                loss = self.criterion(outputs, data)
                # For autoencoder, we don't have traditional accuracy
                accuracy = 0.0
            else:
                loss = self.criterion(outputs.squeeze(), targets.float())
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                accuracy = correct / total if total > 0 else 0.0
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                
                if self.model_type == 'autoencoder':
                    loss = self.criterion(outputs, data)
                    accuracy = 0.0
                else:
                    loss = self.criterion(outputs.squeeze(), targets.float())
                    
                    predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
                    accuracy = correct / total if total > 0 else 0.0
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, accuracy
    
    def evaluate(
        self, 
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded before evaluation")
            
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                
                if self.model_type != 'autoencoder':
                    predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                    all_predictions.extend(predictions)
                    all_targets.extend(targets.numpy())
        
        if self.model_type != 'autoencoder':
            metrics = calculate_detection_metrics(
                np.array(all_targets), 
                np.array(all_predictions)
            )
            return metrics
        else:
            # For autoencoder, return reconstruction-based metrics
            return {'reconstruction_error': 0.0}  # Placeholder
    
    def plot_detection_results(
        self, 
        strain_data: np.ndarray,
        detection_results: Dict,
        time_offset: float = 0.0,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot detection results with strain data.
        
        Args:
            strain_data: Original strain data
            detection_results: Results from detect() method
            time_offset: Time offset for plotting
            save_path: Optional path to save plot
        """
        plot_detection_results(
            strain_data=strain_data,
            detection_results=detection_results,
            sample_rate=self.sample_rate,
            time_offset=time_offset,
            save_path=save_path,
        )
