#!/usr/bin/env python3
"""
Training script for gravitational wave detection models.

This script provides a command-line interface for training neural networks
on gravitational wave data using various architectures and configurations.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from ..detector import GWDetector
from ..data.loader import GWDataset, create_dataloader
from ..utils.config import Config
from ..utils.helpers import (
    setup_logging, create_experiment_directory, save_results,
    seed_everything, print_banner, format_duration
)
from ..utils.metrics import calculate_detection_metrics
from ..visualization.plotting import plot_training_history


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train gravitational wave detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Directory containing training data files'
    )
    parser.add_argument(
        '--labels-file', type=str, 
        help='JSON file containing labels for supervised training'
    )
    parser.add_argument(
        '--validation-split', type=float, default=0.2,
        help='Fraction of data to use for validation'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-type', type=str, default='cnn_lstm',
        choices=['cnn_lstm', 'wavenet', 'transformer', 'autoencoder'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--input-length', type=int, default=32768,
        help='Length of input sequences'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=4096,
        help='Sample rate of input data'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--early-stopping', type=int, default=10,
        help='Early stopping patience (0 to disable)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default='experiments',
        help='Base directory for experiment outputs'
    )
    parser.add_argument(
        '--experiment-name', type=str,
        help='Name of experiment (auto-generated if not provided)'
    )
    parser.add_argument(
        '--save-model', action='store_true',
        help='Save trained model'
    )
    parser.add_argument(
        '--save-plots', action='store_true',
        help='Save training plots'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress non-essential output'
    )
    
    return parser.parse_args()


def load_data_files(data_dir: Path) -> list:
    """Load list of data files from directory."""
    data_files = []
    
    # Look for numpy files
    for ext in ['*.npy', '*.npz']:
        data_files.extend(list(data_dir.glob(ext)))
    
    # Look for HDF5 files
    for ext in ['*.h5', '*.hdf5']:
        data_files.extend(list(data_dir.glob(ext)))
    
    return sorted(data_files)


def load_labels(labels_file: Path) -> Dict[str, int]:
    """Load labels from JSON file."""
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    return labels


def create_datasets(
    data_files: list,
    labels: Dict[str, int] = None,
    validation_split: float = 0.2,
    config: Config = None,
    **dataset_kwargs
) -> tuple:
    """Create training and validation datasets."""
    
    # Prepare labels list if provided
    if labels:
        labels_list = []
        for file_path in data_files:
            file_name = Path(file_path).stem
            if file_name in labels:
                labels_list.append(labels[file_name])
            else:
                raise ValueError(f"No label found for file: {file_name}")
    else:
        labels_list = None
    
    # Create full dataset
    full_dataset = GWDataset(
        data_files=data_files,
        labels=labels_list,
        config=config,
        **dataset_kwargs
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


def setup_training(args: argparse.Namespace) -> tuple:
    """Set up training environment and components."""
    
    # Set up random seeds
    seed_everything(args.seed)
    
    # Create experiment directory
    exp_dir = create_experiment_directory(
        base_dir=args.output_dir,
        experiment_name=args.experiment_name or f"{args.model_type}_training",
        timestamp=True
    )
    
    # Set up logging
    log_file = exp_dir / 'training.log' if not args.quiet else None
    logger = setup_logging(
        level=args.log_level,
        log_file=log_file
    )
    
    # Load or create configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    config.update(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping,
    )
    
    # Save configuration
    config.save(exp_dir / 'config.yaml')
    
    return exp_dir, logger, config


def main() -> int:
    """Main training function."""
    args = parse_arguments()
    
    try:
        # Setup
        exp_dir, logger, config = setup_training(args)
        
        print_banner("Gravitational Wave Detection - Model Training")
        logger.info("Starting training session")
        logger.info(f"Experiment directory: {exp_dir}")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Device: {args.device}")
        
        # Load data
        logger.info("Loading data...")
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        data_files = load_data_files(data_dir)
        if not data_files:
            raise ValueError(f"No data files found in {data_dir}")
        
        logger.info(f"Found {len(data_files)} data files")
        
        # Load labels if provided
        labels = None
        if args.labels_file:
            labels_file = Path(args.labels_file)
            if not labels_file.exists():
                raise FileNotFoundError(f"Labels file not found: {labels_file}")
            labels = load_labels(labels_file)
            logger.info(f"Loaded labels for {len(labels)} files")
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset, val_dataset = create_datasets(
            data_files=data_files,
            labels=labels,
            validation_split=args.validation_split,
            config=config,
            segment_length=args.input_length / args.sample_rate,
            sample_rate=args.sample_rate,
            augment=True  # Enable augmentation for training
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Create detector/model
        logger.info("Initializing model...")
        detector = GWDetector(
            model_type=args.model_type,
            sample_rate=args.sample_rate,
            segment_length=args.input_length / args.sample_rate,
            device=args.device,
            config=config
        )
        
        # Log model information
        if hasattr(detector.model, 'get_model_info'):
            model_info = detector.model.get_model_info()
            logger.info(f"Model parameters: {model_info.get('num_parameters', 'Unknown'):,}")
            logger.info(f"Trainable parameters: {model_info.get('trainable_parameters', 'Unknown'):,}")
        
        # Training
        logger.info("Starting training...")
        start_time = time.time()
        
        history = detector.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_path=exp_dir / 'best_model.pth' if args.save_model else None
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {format_duration(training_time)}")
        
        # Save results
        results = {
            'training_history': history,
            'final_metrics': {
                'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'train_accuracy': history['train_accuracy'][-1] if history['train_accuracy'] else None,
                'val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else None,
            },
            'training_time_seconds': training_time,
            'model_type': args.model_type,
            'total_epochs': len(history['train_loss']) if history['train_loss'] else 0,
        }
        
        save_results(results, exp_dir / 'training_results.json')
        logger.info("Training results saved")
        
        # Create plots if requested
        if args.save_plots and history:
            logger.info("Creating training plots...")
            
            plot_training_history(
                history,
                title=f"{args.model_type.upper()} Training History",
                save_path=exp_dir / 'training_history.png',
                show_plot=False
            )
            
            logger.info("Training plots saved")
        
        # Final evaluation on validation set
        if val_loader and labels is not None:
            logger.info("Performing final evaluation...")
            
            eval_metrics = detector.evaluate(val_loader)
            logger.info("Validation metrics:")
            for metric, value in eval_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Save evaluation results
            save_results(eval_metrics, exp_dir / 'validation_metrics.json')
        
        logger.info("Training session completed successfully")
        logger.info(f"All outputs saved to: {exp_dir}")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Training failed: {str(e)}")
        else:
            print(f"Error: {str(e)}", file=sys.stderr)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
