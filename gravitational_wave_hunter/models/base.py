"""
Base class for gravitational wave detection models.

This module provides the base class and common functionality for all
gravitational wave detection neural network models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class BaseGWModel(nn.Module, ABC):
    """
    Abstract base class for gravitational wave detection models.
    
    This class defines the common interface and functionality that all
    gravitational wave detection models should implement.
    
    Args:
        input_length: Length of input time series
        num_classes: Number of output classes (2 for binary classification)
        dropout_rate: Dropout rate for regularization
        
    Example:
        >>> class MyGWModel(BaseGWModel):
        ...     def __init__(self, input_length, num_classes=2):
        ...         super().__init__(input_length, num_classes)
        ...         self.layer = nn.Linear(input_length, num_classes)
        ...     
        ...     def forward(self, x):
        ...         return self.layer(x)
    """
    
    def __init__(
        self,
        input_length: int,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize the base gravitational wave model."""
        super().__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Model metadata
        self.model_type = self.__class__.__name__
        self.version = "1.0"
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.model_type,
            'version': self.version,
            'input_length': self.input_length,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'num_parameters': self.count_parameters(),
            'trainable_parameters': self.count_parameters(trainable_only=True),
        }
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self, batch_size: int = 1) -> Dict[str, float]:
        """
        Estimate memory usage of the model.
        
        Args:
            batch_size: Batch size for memory calculation
            
        Returns:
            Dictionary with memory usage estimates in MB
        """
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        
        # Calculate activation memory (rough estimate)
        input_memory = batch_size * self.input_length * 4  # float32
        
        # This is a simplified calculation
        # In practice, activation memory depends on the specific architecture
        activation_memory = input_memory * 2  # Rough estimate
        
        total_memory = param_memory + activation_memory
        
        return {
            'parameters_mb': param_memory / (1024 * 1024),
            'activations_mb': activation_memory / (1024 * 1024),
            'total_mb': total_memory / (1024 * 1024),
        }
    
    def init_weights(self) -> None:
        """
        Initialize model weights using appropriate initialization schemes.
        
        This method should be overridden by subclasses to implement
        model-specific weight initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def freeze_layers(self, layer_names: Optional[list] = None) -> None:
        """
        Freeze specified layers or all layers if none specified.
        
        Args:
            layer_names: List of layer names to freeze. If None, freeze all layers.
        """
        if layer_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = False
    
    def unfreeze_layers(self, layer_names: Optional[list] = None) -> None:
        """
        Unfreeze specified layers or all layers if none specified.
        
        Args:
            layer_names: List of layer names to unfreeze. If None, unfreeze all layers.
        """
        if layer_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = True
    
    def summary(self, input_shape: Optional[Tuple[int, ...]] = None) -> str:
        """
        Generate a summary of the model architecture.
        
        Args:
            input_shape: Shape of input tensor (excluding batch dimension)
            
        Returns:
            String containing model summary
        """
        if input_shape is None:
            input_shape = (self.input_length,)
        
        summary_lines = []
        summary_lines.append(f"Model: {self.model_type}")
        summary_lines.append("=" * 50)
        
        # Add layer information
        total_params = 0
        trainable_params = 0
        
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                num_params = sum(p.numel() for p in module.parameters())
                num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                summary_lines.append(f"{name:30} {str(type(module).__name__):20} {num_params:>10,}")
                total_params += num_params
                trainable_params += num_trainable
        
        summary_lines.append("=" * 50)
        summary_lines.append(f"Total parameters: {total_params:,}")
        summary_lines.append(f"Trainable parameters: {trainable_params:,}")
        summary_lines.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Add memory usage
        memory_info = self.get_memory_usage()
        summary_lines.append(f"Model size: {memory_info['parameters_mb']:.2f} MB")
        
        return "\n".join(summary_lines)
    
    def validate_input(self, x: torch.Tensor) -> None:
        """
        Validate input tensor shape and type.
        
        Args:
            x: Input tensor to validate
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input must be a torch.Tensor, got {type(x)}")
        
        if x.dim() < 2:
            raise ValueError(f"Input must have at least 2 dimensions (batch, sequence), got {x.dim()}")
        
        if x.shape[-1] != self.input_length:
            raise ValueError(
                f"Input sequence length must be {self.input_length}, got {x.shape[-1]}"
            )
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor of prediction probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                # Binary classification with single output
                probs = torch.sigmoid(logits)
            else:
                # Multi-class classification
                probs = torch.softmax(logits, dim=-1)
        return probs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get discrete predictions.
        
        Args:
            x: Input tensor
            threshold: Decision threshold for binary classification
            
        Returns:
            Tensor of discrete predictions
        """
        probs = self.predict_proba(x)
        
        if self.num_classes == 1:
            # Binary classification
            predictions = (probs > threshold).long()
        else:
            # Multi-class classification
            predictions = torch.argmax(probs, dim=-1)
        
        return predictions
