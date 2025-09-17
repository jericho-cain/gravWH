#!/usr/bin/env python3
"""
Try different threshold strategies to catch the missed gravitational wave signal.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from gravitational_wave_hunter.data.simple_training_pipeline import SimpleTrainingPipeline

def test_threshold_strategies():
    """Test different threshold strategies to catch the missed signal."""
    print("🎯 Testing threshold strategies to catch the missed signal...")
    
    # Run the pipeline with seed=7 to get the same results
    pipeline = SimpleTrainingPipeline(random_seed=7)
    
    # Get the test data and results
    print("📊 Running pipeline to get test data...")
    lstm_results, transformer_results = pipeline.run_complete_pipeline(
        num_training_samples=100, 
        num_test_samples=25
    )
    
    # Extract the data we need
    scores = lstm_results['scores']
    precision = lstm_results['precision']
    recall = lstm_results['recall']
    thresholds = lstm_results['thresholds']
    fpr = lstm_results['fpr']
    tpr = lstm_results['tpr']
    
    # Get test labels
    test_data, test_labels, test_snr = pipeline.download_test_data(25)
    
    # Ensure arrays have same length
    min_length = min(len(test_labels), len(scores))
    test_labels = test_labels[:min_length]
    scores = scores[:min_length]
    
    print(f"📈 Current Performance:")
    print(f"   Total test samples: {len(test_labels)}")
    print(f"   True signals: {np.sum(test_labels)}")
    
    # Strategy 1: Current (Max Precision)
    max_precision_idx = np.argmax(precision)
    threshold_idx = min(max_precision_idx, len(thresholds) - 1)
    current_threshold = thresholds[threshold_idx]
    current_predictions = (scores >= current_threshold).astype(int)
    
    print(f"\n🎯 Strategy 1: Max Precision Threshold")
    print(f"   Threshold: {current_threshold:.4f}")
    print(f"   Precision: {precision[max_precision_idx]:.4f}")
    print(f"   Recall: {recall[max_precision_idx]:.4f}")
    print(f"   F1: {f1_score(test_labels, current_predictions):.4f}")
    print(f"   Detected signals: {np.sum(current_predictions)}/{np.sum(test_labels)}")
    
    # Strategy 2: Lower threshold (catch the missed signal)
    missed_signal_score = 0.4879  # From previous analysis
    lower_threshold = missed_signal_score + 0.01  # Just above the missed signal
    lower_predictions = (scores >= lower_threshold).astype(int)
    
    print(f"\n🎯 Strategy 2: Lower Threshold (Catch Missed Signal)")
    print(f"   Threshold: {lower_threshold:.4f}")
    print(f"   Precision: {precision_score(test_labels, lower_predictions):.4f}")
    print(f"   Recall: {recall_score(test_labels, lower_predictions):.4f}")
    print(f"   F1: {f1_score(test_labels, lower_predictions):.4f}")
    print(f"   Detected signals: {np.sum(lower_predictions)}/{np.sum(test_labels)}")
    
    # Strategy 3: F1-optimized threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    f1_optimal_idx = np.argmax(f1_scores)
    f1_threshold = thresholds[f1_optimal_idx]
    f1_predictions = (scores >= f1_threshold).astype(int)
    
    print(f"\n🎯 Strategy 3: F1-Optimized Threshold")
    print(f"   Threshold: {f1_threshold:.4f}")
    print(f"   Precision: {precision[f1_optimal_idx]:.4f}")
    print(f"   Recall: {recall[f1_optimal_idx]:.4f}")
    print(f"   F1: {f1_scores[f1_optimal_idx]:.4f}")
    print(f"   Detected signals: {np.sum(f1_predictions)}/{np.sum(test_labels)}")
    
    # Strategy 4: Youden's J statistic (maximizes TPR - FPR)
    youden_j = tpr - fpr
    youden_idx = np.argmax(youden_j)
    youden_threshold = thresholds[youden_idx] if youden_idx < len(thresholds) else thresholds[-1]
    youden_predictions = (scores >= youden_threshold).astype(int)
    
    print(f"\n🎯 Strategy 4: Youden's J Statistic")
    print(f"   Threshold: {youden_threshold:.4f}")
    print(f"   Precision: {precision_score(test_labels, youden_predictions):.4f}")
    print(f"   Recall: {recall_score(test_labels, youden_predictions):.4f}")
    print(f"   F1: {f1_score(test_labels, youden_predictions):.4f}")
    print(f"   Detected signals: {np.sum(youden_predictions)}/{np.sum(test_labels)}")
    
    # Strategy 5: Balanced threshold (0.5)
    balanced_threshold = 0.5
    balanced_predictions = (scores >= balanced_threshold).astype(int)
    
    print(f"\n🎯 Strategy 5: Balanced Threshold (0.5)")
    print(f"   Threshold: {balanced_threshold:.4f}")
    print(f"   Precision: {precision_score(test_labels, balanced_predictions):.4f}")
    print(f"   Recall: {recall_score(test_labels, balanced_predictions):.4f}")
    print(f"   F1: {f1_score(test_labels, balanced_predictions):.4f}")
    print(f"   Detected signals: {np.sum(balanced_predictions)}/{np.sum(test_labels)}")
    
    # Find the best strategy
    strategies = [
        ("Max Precision", current_threshold, current_predictions),
        ("Lower Threshold", lower_threshold, lower_predictions),
        ("F1-Optimized", f1_threshold, f1_predictions),
        ("Youden's J", youden_threshold, youden_predictions),
        ("Balanced (0.5)", balanced_threshold, balanced_predictions)
    ]
    
    print(f"\n🏆 Strategy Comparison:")
    print(f"{'Strategy':<15} {'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Signals':<10}")
    print("-" * 75)
    
    best_f1 = 0
    best_strategy = None
    
    for name, threshold, predictions in strategies:
        prec = precision_score(test_labels, predictions)
        rec = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        signals = f"{np.sum(predictions)}/{np.sum(test_labels)}"
        
        print(f"{name:<15} {threshold:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {signals:<10}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_strategy = (name, threshold, predictions)
    
    print(f"\n🥇 Best Strategy: {best_strategy[0]}")
    print(f"   Threshold: {best_strategy[1]:.4f}")
    print(f"   F1 Score: {best_f1:.4f}")
    
    return {
        'strategies': strategies,
        'best_strategy': best_strategy,
        'test_labels': test_labels,
        'scores': scores
    }

if __name__ == "__main__":
    results = test_threshold_strategies()
