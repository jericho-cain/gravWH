#!/usr/bin/env python3
"""
Evaluation script for gravitational wave detection models.

This script provides comprehensive evaluation of trained models including
performance metrics, ROC curves, and detection statistics.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..detector import GWDetector
from ..data.loader import GWDataset, create_dataloader
from ..utils.config import Config
from ..utils.helpers import (
    setup_logging, ensure_directory, save_results, 
    format_duration, load_results
)
from ..utils.metrics import (
    calculate_detection_metrics, evaluate_model_performance,
    detection_statistics, plot_roc_curve, plot_precision_recall_curve,
    plot_confusion_matrix
)
from ..visualization.plotting import create_comparison_plot


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate gravitational wave detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--model-type', type=str, default='cnn_lstm',
        choices=['cnn_lstm', 'wavenet', 'transformer', 'autoencoder'],
        help='Type of model'
    )
    
    # Data arguments
    parser.add_argument(
        '--test-data', type=str, required=True,
        help='Directory containing test data files'
    )
    parser.add_argument(
        '--test-labels', type=str,
        help='JSON file containing test labels'
    )
    parser.add_argument(
        '--ground-truth', type=str,
        help='JSON file containing ground truth events for detection statistics'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Detection threshold'
    )
    parser.add_argument(
        '--thresholds', type=str,
        help='Comma-separated list of thresholds for ROC analysis'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=4096,
        help='Sample rate of input data'
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for evaluation'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--save-plots', action='store_true',
        help='Save evaluation plots'
    )
    parser.add_argument(
        '--detailed-analysis', action='store_true',
        help='Perform detailed analysis including per-sample results'
    )
    
    # Comparison
    parser.add_argument(
        '--compare-models', nargs='+',
        help='Paths to additional models for comparison'
    )
    parser.add_argument(
        '--baseline-results', type=str,
        help='Path to baseline results for comparison'
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


def load_test_data(
    test_data_dir: Path,
    labels_file: Path = None,
    config: Config = None,
    **dataset_kwargs
) -> DataLoader:
    """Load test dataset."""
    
    # Find data files
    data_files = []
    for ext in ['*.npy', '*.npz', '*.h5', '*.hdf5']:
        data_files.extend(list(test_data_dir.glob(ext)))
    
    if not data_files:
        raise ValueError(f"No data files found in {test_data_dir}")
    
    data_files = sorted(data_files)
    
    # Load labels if provided
    labels = None
    if labels_file and labels_file.exists():
        with open(labels_file, 'r') as f:
            label_dict = json.load(f)
        
        labels = []
        for file_path in data_files:
            file_name = file_path.stem
            if file_name in label_dict:
                labels.append(label_dict[file_name])
            else:
                raise ValueError(f"No label found for file: {file_name}")
    
    # Create dataset
    dataset = GWDataset(
        data_files=data_files,
        labels=labels,
        config=config,
        augment=False,  # No augmentation for testing
        **dataset_kwargs
    )
    
    return create_dataloader(dataset, shuffle=False), data_files


def evaluate_single_model(
    model_path: Path,
    test_loader: DataLoader,
    args: argparse.Namespace,
    config: Config = None
) -> Dict[str, Any]:
    """Evaluate a single model."""
    
    # Set up detector
    detector = GWDetector(
        model_type=args.model_type,
        sample_rate=args.sample_rate,
        device=args.device,
        config=config
    )
    
    # Load model
    detector.load_pretrained(model_path)
    
    # Basic evaluation
    metrics = evaluate_model_performance(
        detector.model,
        test_loader,
        device=detector.device,
        threshold=args.threshold
    )
    
    # Detailed evaluation if requested
    detailed_results = None
    if args.detailed_analysis:
        detailed_results = perform_detailed_analysis(
            detector, test_loader, args.threshold
        )
    
    return {
        'model_path': str(model_path),
        'metrics': metrics,
        'detailed_results': detailed_results,
    }


def perform_detailed_analysis(
    detector: GWDetector,
    test_loader: DataLoader,
    threshold: float
) -> Dict[str, Any]:
    """Perform detailed per-sample analysis."""
    
    detector.model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    sample_info = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(detector.device)
            targets = targets.to(detector.device)
            
            outputs = detector.model(data)
            
            # Convert to probabilities
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs).squeeze()
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]
            
            # Store results
            batch_probs = probs.cpu().numpy()
            batch_targets = targets.cpu().numpy()
            batch_preds = (batch_probs >= threshold).astype(int)
            
            all_probabilities.extend(batch_probs)
            all_targets.extend(batch_targets)
            all_predictions.extend(batch_preds)
            
            # Store sample information
            for i in range(len(batch_probs)):
                sample_info.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'true_label': int(batch_targets[i]),
                    'predicted_label': int(batch_preds[i]),
                    'probability': float(batch_probs[i]),
                    'correct': batch_targets[i] == batch_preds[i],
                })
    
    # Calculate per-threshold metrics
    thresholds = np.linspace(0, 1, 101)
    threshold_metrics = []
    
    for thresh in thresholds:
        preds = (np.array(all_probabilities) >= thresh).astype(int)
        metrics = calculate_detection_metrics(
            np.array(all_targets), 
            preds,
            threshold=thresh
        )
        metrics['threshold'] = thresh
        threshold_metrics.append(metrics)
    
    return {
        'sample_results': sample_info,
        'threshold_analysis': threshold_metrics,
        'summary_stats': {
            'total_samples': len(all_targets),
            'positive_samples': sum(all_targets),
            'negative_samples': len(all_targets) - sum(all_targets),
            'correct_predictions': sum(t == p for t, p in zip(all_targets, all_predictions)),
        }
    }


def compare_models(
    model_paths: List[Path],
    test_loader: DataLoader,
    args: argparse.Namespace,
    config: Config = None
) -> Dict[str, Any]:
    """Compare multiple models."""
    
    results = {}
    
    for i, model_path in enumerate(model_paths):
        model_name = f"model_{i+1}_{model_path.stem}"
        print(f"Evaluating {model_name}...")
        
        result = evaluate_single_model(model_path, test_loader, args, config)
        results[model_name] = result
    
    # Create comparison summary
    comparison = {
        'models': list(results.keys()),
        'metrics_comparison': {},
    }
    
    # Extract key metrics for comparison
    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    for metric in key_metrics:
        comparison['metrics_comparison'][metric] = {}
        for model_name, result in results.items():
            if metric in result['metrics']:
                comparison['metrics_comparison'][metric][model_name] = result['metrics'][metric]
    
    return {
        'individual_results': results,
        'comparison': comparison,
    }


def create_evaluation_plots(
    results: Dict[str, Any],
    output_dir: Path,
    test_loader: DataLoader = None
) -> None:
    """Create evaluation plots."""
    
    if 'individual_results' in results:
        # Multiple model comparison
        for model_name, result in results['individual_results'].items():
            model_dir = output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            create_single_model_plots(result, model_dir, model_name)
        
        # Create comparison plots
        create_comparison_plots(results, output_dir)
    
    else:
        # Single model
        create_single_model_plots(results, output_dir, "model")


def create_single_model_plots(
    result: Dict[str, Any],
    output_dir: Path,
    model_name: str
) -> None:
    """Create plots for a single model."""
    
    # This is a simplified version - in practice you'd need the actual
    # predictions and targets to create ROC curves, etc.
    
    # Create a summary plot with key metrics
    metrics = result['metrics']
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics.keys())
    metric_values = [metrics[name] for name in metric_names if isinstance(metrics[name], (int, float))]
    metric_names = [name for name in metric_names if isinstance(metrics[name], (int, float))]
    
    bars = ax.bar(metric_names, metric_values)
    ax.set_ylabel('Value')
    ax.set_title(f'Performance Metrics - {model_name}')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_plots(results: Dict[str, Any], output_dir: Path) -> None:
    """Create comparison plots for multiple models."""
    
    import matplotlib.pyplot as plt
    
    comparison = results['comparison']
    metrics_comp = comparison['metrics_comparison']
    
    # Create comparison bar plot
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    available_metrics = [m for m in metrics_to_plot if m in metrics_comp]
    
    if not available_metrics:
        return
    
    n_metrics = len(available_metrics)
    n_models = len(comparison['models'])
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        model_names = list(metrics_comp[metric].keys())
        values = list(metrics_comp[metric].values())
        
        bars = ax.bar(range(len(model_names)), values)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([name.replace('_', '\n') for name in model_names], rotation=0)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main() -> int:
    """Main evaluation function."""
    args = parse_arguments()
    
    try:
        # Set up output directory
        output_dir = ensure_directory(args.output_dir)
        
        # Set up logging
        log_file = output_dir / 'evaluation.log' if not args.quiet else None
        logger = setup_logging(
            level=args.log_level,
            log_file=log_file
        )
        
        logger.info("Starting model evaluation")
        logger.info(f"Output directory: {output_dir}")
        
        # Load configuration
        config = None
        if args.config:
            config = Config.load(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        
        # Load test data
        logger.info("Loading test data...")
        test_data_dir = Path(args.test_data)
        labels_file = Path(args.test_labels) if args.test_labels else None
        
        test_loader, data_files = load_test_data(
            test_data_dir=test_data_dir,
            labels_file=labels_file,
            config=config,
            segment_length=8.0,  # Default segment length
            sample_rate=args.sample_rate,
        )
        
        logger.info(f"Loaded test dataset with {len(test_loader.dataset)} samples")
        
        # Evaluate model(s)
        start_time = time.time()
        
        if args.compare_models:
            # Multiple model comparison
            model_paths = [Path(args.model_path)] + [Path(p) for p in args.compare_models]
            logger.info(f"Comparing {len(model_paths)} models")
            
            results = compare_models(model_paths, test_loader, args, config)
        else:
            # Single model evaluation
            model_path = Path(args.model_path)
            logger.info(f"Evaluating model: {model_path}")
            
            results = evaluate_single_model(model_path, test_loader, args, config)
        
        eval_time = time.time() - start_time
        logger.info(f"Evaluation completed in {format_duration(eval_time)}")
        
        # Save results
        save_results(results, output_dir / 'evaluation_results.json')
        logger.info("Evaluation results saved")
        
        # Create plots
        if args.save_plots:
            logger.info("Creating evaluation plots...")
            create_evaluation_plots(results, output_dir, test_loader)
            logger.info("Evaluation plots saved")
        
        # Print summary
        if not args.quiet:
            print("\n" + "="*60)
            print("MODEL EVALUATION RESULTS")
            print("="*60)
            
            if 'metrics' in results:
                # Single model
                metrics = results['metrics']
                print(f"Model: {Path(args.model_path).name}")
                print(f"Test Samples: {len(test_loader.dataset)}")
                print("\nPerformance Metrics:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            
            elif 'comparison' in results:
                # Multiple models
                comparison = results['comparison']
                print(f"Compared Models: {len(comparison['models'])}")
                print(f"Test Samples: {len(test_loader.dataset)}")
                print("\nMetrics Comparison:")
                
                for metric, model_values in comparison['metrics_comparison'].items():
                    print(f"\n{metric.replace('_', ' ').title()}:")
                    for model, value in model_values.items():
                        print(f"  {model}: {value:.4f}")
                    
                    # Find best model for this metric
                    best_model = max(model_values.items(), key=lambda x: x[1])
                    print(f"  Best: {best_model[0]} ({best_model[1]:.4f})")
            
            print("="*60)
        
        logger.info("Evaluation session completed successfully")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Evaluation failed: {str(e)}")
        else:
            print(f"Error: {str(e)}", file=sys.stderr)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
