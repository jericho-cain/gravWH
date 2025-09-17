#!/usr/bin/env python3
"""
Analyze the missed gravitational wave detection to understand why it was missed.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from gravitational_wave_hunter.data.simple_training_pipeline import SimpleTrainingPipeline

def analyze_missed_detection():
    """Analyze the missed gravitational wave detection."""
    print("🔍 Analyzing missed gravitational wave detection...")
    
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
    
    # Get test labels directly from the pipeline's test data
    print("📊 Getting test labels from pipeline...")
    test_data, test_labels, test_snr = pipeline.download_test_data(25)
    
    if test_labels is None:
        print("❌ Test labels not available")
        return
    
    # Find optimal threshold (max precision)
    max_precision_idx = np.argmax(precision)
    threshold_idx = min(max_precision_idx, len(thresholds) - 1)
    optimal_threshold = thresholds[threshold_idx]
    
    # Generate predictions
    predictions = (scores >= optimal_threshold).astype(int)
    
    # Ensure arrays have same length
    min_length = min(len(test_labels), len(predictions), len(scores))
    test_labels = test_labels[:min_length]
    predictions = predictions[:min_length]
    scores = scores[:min_length]
    
    print(f"📈 Analysis Results:")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   Total test samples: {len(test_labels)}")
    print(f"   True signals: {np.sum(test_labels)}")
    print(f"   Predicted signals: {np.sum(predictions)}")
    
    # Find missed detections (false negatives)
    false_negatives = (test_labels == 1) & (predictions == 0)
    missed_indices = np.where(false_negatives)[0]
    
    print(f"   Missed detections: {len(missed_indices)}")
    
    if len(missed_indices) > 0:
        print(f"\n🎯 Analyzing missed detection(s):")
        
        for i, missed_idx in enumerate(missed_indices):
            print(f"\n   Missed Signal #{i+1} (Index {missed_idx}):")
            print(f"   - True label: {test_labels[missed_idx]} (Signal)")
            print(f"   - Predicted: {predictions[missed_idx]} (Noise)")
            print(f"   - Score: {scores[missed_idx]:.4f}")
            print(f"   - Threshold: {optimal_threshold:.4f}")
            print(f"   - Distance from threshold: {scores[missed_idx] - optimal_threshold:.4f}")
            
            # Compare with detected signals
            detected_signals = (test_labels == 1) & (predictions == 1)
            detected_indices = np.where(detected_signals)[0]
            
            if len(detected_indices) > 0:
                detected_scores = scores[detected_indices]
                print(f"   - Detected signal scores: {detected_scores}")
                print(f"   - Min detected score: {np.min(detected_scores):.4f}")
                print(f"   - Max detected score: {np.max(detected_scores):.4f}")
                print(f"   - Mean detected score: {np.mean(detected_scores):.4f}")
                
                # Calculate how close the missed signal was
                min_detected = np.min(detected_scores)
                missed_score = scores[missed_idx]
                gap = min_detected - missed_score
                print(f"   - Gap from closest detection: {gap:.4f}")
                
                if gap < 0.1:
                    print(f"   - ⚠️  Very close to detection threshold!")
                elif gap < 0.2:
                    print(f"   - 🔶 Moderately close to detection threshold")
                else:
                    print(f"   - 🔴 Far from detection threshold")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   True Positives: {np.sum((test_labels == 1) & (predictions == 1))}")
    print(f"   False Positives: {np.sum((test_labels == 0) & (predictions == 1))}")
    print(f"   True Negatives: {np.sum((test_labels == 0) & (predictions == 0))}")
    print(f"   False Negatives: {np.sum((test_labels == 1) & (predictions == 0))}")
    
    # Score distribution analysis
    signal_scores = scores[test_labels == 1]
    noise_scores = scores[test_labels == 0]
    
    print(f"\n📈 Score Distribution:")
    print(f"   Signal scores - Min: {np.min(signal_scores):.4f}, Max: {np.max(signal_scores):.4f}, Mean: {np.mean(signal_scores):.4f}")
    print(f"   Noise scores - Min: {np.min(noise_scores):.4f}, Max: {np.max(noise_scores):.4f}, Mean: {np.mean(noise_scores):.4f}")
    
    return {
        'missed_indices': missed_indices,
        'missed_scores': scores[missed_indices] if len(missed_indices) > 0 else [],
        'detected_scores': scores[(test_labels == 1) & (predictions == 1)],
        'optimal_threshold': optimal_threshold,
        'test_labels': test_labels,
        'predictions': predictions,
        'scores': scores
    }

if __name__ == "__main__":
    results = analyze_missed_detection()
