#!/usr/bin/env python3
"""
Model Evaluation Script

This script loads a trained model and evaluates it on test data.
Run this after training is complete.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Add the project root to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gravitational_wave_hunter', 'data'))
from modular_training_pipeline import ModularTrainingPipeline
from ligo_data_loader import LIGODataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model_path=None, num_test_samples=25, random_seed=7):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path (str): Path to saved model (optional)
        num_test_samples (int): Number of test samples to evaluate
        random_seed (int): Random seed for reproducibility
    """
    
    logger.info("🔍 Starting model evaluation...")
    
    # Initialize pipeline
    pipeline = ModularTrainingPipeline(random_seed=random_seed)
    
    # Load test data
    logger.info(f"📥 Loading {num_test_samples} test samples...")
    test_data, test_labels, test_snr = pipeline.load_test_data(num_test_samples)
    
    # For now, we'll retrain a model (in a real scenario, you'd load a saved model)
    # TODO: Add model loading functionality
    logger.info("🧠 Training model for evaluation...")
    
    # Use minimal training data for evaluation
    train_data = pipeline.download_training_data(50)  # Reduced for speed
    
    # Train model
    lstm_results, transformer_results = pipeline.train_models(
        train_data, test_data, test_labels, test_snr
    )
    
    # Extract scores and labels
    scores = lstm_results['scores']
    precision = lstm_results['precision']
    recall = lstm_results['recall']
    thresholds = lstm_results['thresholds']
    
    logger.info(f"📊 Evaluation Results:")
    logger.info(f"   AUC: {lstm_results['auc']:.3f}")
    logger.info(f"   Average Precision: {lstm_results['avg_precision']:.3f}")
    
    # Find optimal threshold (max precision)
    max_precision_idx = np.argmax(precision)
    threshold_idx = min(max_precision_idx, len(thresholds) - 1)
    optimal_threshold = thresholds[threshold_idx]
    
    # Generate predictions
    predictions = (scores >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    tp = cm[1, 1]
    fp = cm[0, 1]
    tn = cm[0, 0]
    fn = cm[1, 0]
    
    logger.info(f"\n📈 Confusion Matrix:")
    logger.info(f"   True Positives: {tp}")
    logger.info(f"   False Positives: {fp}")
    logger.info(f"   True Negatives: {tn}")
    logger.info(f"   False Negatives: {fn}")
    
    # Calculate metrics
    prec = precision_score(test_labels, predictions, zero_division=0)
    rec = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    logger.info(f"\n🎯 Performance Metrics:")
    logger.info(f"   Precision: {prec:.3f}")
    logger.info(f"   Recall: {rec:.3f}")
    logger.info(f"   F1-Score: {f1:.3f}")
    logger.info(f"   Optimal Threshold: {optimal_threshold:.4f}")
    
    # Analyze missed detections
    missed_signals = np.where((test_labels == 1) & (predictions == 0))[0]
    detected_signals = np.where((test_labels == 1) & (predictions == 1))[0]
    
    logger.info(f"\n🔍 Signal Analysis:")
    logger.info(f"   Total signals in test set: {np.sum(test_labels == 1)}")
    logger.info(f"   Detected signals: {len(detected_signals)}")
    logger.info(f"   Missed signals: {len(missed_signals)}")
    
    if len(missed_signals) > 0:
        logger.info(f"\n❌ Missed Signals:")
        for i, missed_idx in enumerate(missed_signals):
            missed_score = scores[missed_idx]
            logger.info(f"   Signal {i+1}: Score={missed_score:.4f}, Threshold={optimal_threshold:.4f}")
    
    if len(detected_signals) > 0:
        detected_scores = scores[detected_signals]
        logger.info(f"\n✅ Detected Signals:")
        logger.info(f"   Score range: {np.min(detected_scores):.4f} - {np.max(detected_scores):.4f}")
        logger.info(f"   Mean score: {np.mean(detected_scores):.4f}")
    
    # Create evaluation plots
    create_evaluation_plots(lstm_results, test_labels, test_snr)
    
    return {
        'auc': lstm_results['auc'],
        'avg_precision': lstm_results['avg_precision'],
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold,
        'missed_signals': len(missed_signals),
        'detected_signals': len(detected_signals)
    }

def create_evaluation_plots(results, test_labels, test_snr):
    """Create evaluation plots."""
    logger.info("📊 Creating evaluation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gravitational Wave Detection Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Precision-Recall Curve
    ax1 = axes[0, 0]
    ax1.plot(results['recall'], results['precision'], 'b-', linewidth=2, 
            label=f'CWT-LSTM (AP={results["avg_precision"]:.3f})')
    
    # Add baseline
    baseline = np.mean(test_labels)
    ax1.axhline(y=baseline, color='gray', linestyle='--', alpha=0.8, 
               label=f'Random (AP={baseline:.3f})')
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: ROC Curve
    ax2 = axes[0, 1]
    ax2.plot(results['fpr'], results['tpr'], 'b-', linewidth=2,
            label=f'CWT-LSTM (AUC={results["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random')
    
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Plot 3: Confusion Matrix
    ax3 = axes[1, 0]
    plot_confusion_matrix(ax3, results, test_labels)
    
    # Plot 4: Reconstruction Error Distribution
    ax4 = axes[1, 1]
    plot_reconstruction_errors(ax4, results, test_labels)
    
    plt.tight_layout()
    
    # Save plots
    os.makedirs('results', exist_ok=True)
    plot_path = 'results/modular_evaluation_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"📊 Evaluation plots saved to {plot_path}")
    
    plt.show()

def plot_confusion_matrix(ax, results, test_labels):
    """Plot confusion matrix."""
    scores = results['scores']
    precision = results['precision']
    recall = results['recall']
    thresholds = results['thresholds']
    
    # Find threshold with maximum precision
    max_precision_idx = np.argmax(precision)
    threshold_idx = min(max_precision_idx, len(thresholds) - 1)
    threshold = thresholds[threshold_idx]
    precision_val = precision[max_precision_idx]
    recall_val = recall[max_precision_idx]
    
    # Generate predictions
    predictions = (scores >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (Max Precision)\nP={precision_val:.3f}, R={recall_val:.3f}')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Noise', 'Signal'])
    ax.set_yticklabels(['Noise', 'Signal'])
    
    # Calculate and display F1 score
    f1 = f1_score(test_labels, predictions)
    ax.text(0.5, -0.15, f'F1={f1:.3f}', transform=ax.transAxes, 
           ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

def plot_reconstruction_errors(ax, results, test_labels):
    """Plot reconstruction error distribution."""
    scores = results['scores']
    
    # Separate scores by true labels
    noise_scores = scores[test_labels == 0]
    signal_scores = scores[test_labels == 1]
    
    # Create histogram
    bins = np.linspace(scores.min(), scores.max(), 30)
    
    ax.hist(noise_scores, bins=bins, alpha=0.6, label=f'Noise (n={len(noise_scores)})', 
           color='blue', density=True)
    ax.hist(signal_scores, bins=bins, alpha=0.6, label=f'Signals (n={len(signal_scores)})', 
           color='red', density=True)
    
    # Add vertical line for optimal threshold
    optimal_threshold = results.get('optimal_threshold', 0.5)
    ax.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2, 
              label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    ax.set_xlabel('Reconstruction Error (Anomaly Score)')
    ax.set_ylabel('Density')
    ax.set_title('Reconstruction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    noise_mean = np.mean(noise_scores) if len(noise_scores) > 0 else 0
    signal_mean = np.mean(signal_scores) if len(signal_scores) > 0 else 0
    separation = abs(signal_mean - noise_mean)
    
    stats_text = f'Noise μ={noise_mean:.3f}\nSignal μ={signal_mean:.3f}\nSeparation={separation:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_model(num_test_samples=25, random_seed=7)
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📊 Final Results:")
    print(f"   AUC: {results['auc']:.3f}")
    print(f"   Average Precision: {results['avg_precision']:.3f}")
    print(f"   Precision: {results['precision']:.3f}")
    print(f"   Recall: {results['recall']:.3f}")
    print(f"   F1-Score: {results['f1']:.3f}")
    print(f"   Detected Signals: {results['detected_signals']}")
    print(f"   Missed Signals: {results['missed_signals']}")
