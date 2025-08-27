"""
Metrics and evaluation utilities for gravitational wave detection.

This module provides comprehensive metrics for evaluating the performance
of gravitational wave detection models.
"""

from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


def calculate_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = 'binary',
) -> Dict[str, float]:
    """
    Calculate comprehensive detection performance metrics.
    
    Args:
        y_true: True binary labels (0=noise, 1=signal)
        y_pred: Predicted probabilities or binary predictions
        threshold: Decision threshold for converting probabilities to predictions
        average: Averaging strategy for multi-class ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
        Dictionary containing various performance metrics
        
    Example:
        >>> y_true = np.array([0, 0, 1, 1, 0, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        >>> metrics = calculate_detection_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # Convert probabilities to binary predictions if needed
    if y_pred.dtype == float and np.all((y_pred >= 0) & (y_pred <= 1)):
        y_pred_binary = (y_pred >= threshold).astype(int)
        probabilities = y_pred
    else:
        y_pred_binary = y_pred.astype(int)
        probabilities = None
    
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average=average, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Calculate specific rates
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Detection Efficiency (same as recall/sensitivity)
        detection_efficiency = sensitivity
        
        # False Alarm Rate (per unit time - requires additional info)
        # For now, we'll use FPR as a proxy
        false_alarm_rate = fpr
        
    else:
        # Multi-class case
        sensitivity = recall
        specificity = np.nan
        fpr = np.nan
        fnr = np.nan
        detection_efficiency = recall
        false_alarm_rate = np.nan
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'detection_efficiency': detection_efficiency,
        'false_alarm_rate': false_alarm_rate,
    }
    
    # Add AUC metrics if probabilities are available
    if probabilities is not None:
        try:
            if average == 'binary':
                auc_roc = roc_auc_score(y_true, probabilities)
                
                # Calculate AUC-PR
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)
                auc_pr = np.trapz(precision_curve, recall_curve)
                
                metrics['auc_roc'] = auc_roc
                metrics['auc_pr'] = auc_pr
            else:
                # Multi-class AUC
                auc_roc = roc_auc_score(y_true, probabilities, average=average, multi_class='ovr')
                metrics['auc_roc'] = auc_roc
                metrics['auc_pr'] = np.nan
                
        except ValueError as e:
            warnings.warn(f"Could not calculate AUC metrics: {e}")
            metrics['auc_roc'] = np.nan
            metrics['auc_pr'] = np.nan
    
    return metrics


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores (probabilities)
        pos_label: Label of positive class
        
    Returns:
        Tuple of (false_positive_rates, true_positive_rates, thresholds)
        
    Example:
        >>> fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
        >>> auc = np.trapz(tpr, fpr)
    """
    return roc_curve(y_true, y_scores, pos_label=pos_label)


def compute_precision_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores (probabilities)
        pos_label: Label of positive class
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    return precision_recall_curve(y_true, y_scores, pos_label=pos_label)


def calculate_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    method: str = 'power',
) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    Args:
        signal: Signal data
        noise: Noise data (or background)
        method: Method for SNR calculation ('power', 'amplitude', 'rms')
        
    Returns:
        SNR value in dB
        
    Example:
        >>> signal = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.randn(1000) * 0.1
        >>> noise = np.random.randn(1000) * 0.1
        >>> snr = calculate_snr(signal, noise)
    """
    if method == 'power':
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        snr_linear = signal_power / noise_power if noise_power > 0 else np.inf
    
    elif method == 'amplitude':
        signal_amp = np.max(np.abs(signal))
        noise_amp = np.std(noise)
        snr_linear = signal_amp / noise_amp if noise_amp > 0 else np.inf
    
    elif method == 'rms':
        signal_rms = np.sqrt(np.mean(signal ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        snr_linear = signal_rms / noise_rms if noise_rms > 0 else np.inf
    
    else:
        raise ValueError(f"Unknown SNR method: {method}")
    
    # Convert to dB
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
    
    return snr_db


def detection_statistics(
    detections: List[Tuple[float, float]],
    true_events: List[Tuple[float, float]],
    coincidence_window: float = 1.0,
    total_duration: Optional[float] = None,
) -> Dict[str, Union[int, float]]:
    """
    Calculate detection statistics comparing detected events to true events.
    
    Args:
        detections: List of (start_time, end_time) tuples for detections
        true_events: List of (start_time, end_time) tuples for true events
        coincidence_window: Time window for considering a detection as coincident
        total_duration: Total observation time for calculating rates
        
    Returns:
        Dictionary with detection statistics
        
    Example:
        >>> detections = [(10.5, 11.0), (25.2, 25.8)]
        >>> true_events = [(10.0, 11.5), (30.0, 31.0)]
        >>> stats = detection_statistics(detections, true_events, total_duration=100.0)
    """
    # Convert to arrays for easier manipulation
    det_times = np.array([det[0] for det in detections])  # Use start times
    true_times = np.array([evt[0] for evt in true_events])  # Use start times
    
    # Find coincident detections
    coincident_detections = 0
    detected_events = set()
    
    for i, det_time in enumerate(det_times):
        # Check if this detection is within coincidence window of any true event
        time_diffs = np.abs(true_times - det_time)
        coincident_mask = time_diffs <= coincidence_window
        
        if np.any(coincident_mask):
            coincident_detections += 1
            # Mark the closest true event as detected
            closest_event_idx = np.argmin(time_diffs)
            detected_events.add(closest_event_idx)
    
    # Calculate statistics
    n_detections = len(detections)
    n_true_events = len(true_events)
    n_detected_events = len(detected_events)
    
    # True positives: coincident detections
    true_positives = coincident_detections
    
    # False positives: detections not coincident with true events
    false_positives = n_detections - true_positives
    
    # False negatives: true events not detected
    false_negatives = n_true_events - n_detected_events
    
    # Detection efficiency
    detection_efficiency = n_detected_events / n_true_events if n_true_events > 0 else 0.0
    
    # Precision
    precision = true_positives / n_detections if n_detections > 0 else 0.0
    
    stats = {
        'n_detections': n_detections,
        'n_true_events': n_true_events,
        'n_detected_events': n_detected_events,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'detection_efficiency': detection_efficiency,
        'precision': precision,
    }
    
    # Add rate information if total duration is provided
    if total_duration is not None:
        stats.update({
            'detection_rate': n_detections / total_duration,
            'true_event_rate': n_true_events / total_duration,
            'false_positive_rate': false_positives / total_duration,
        })
    
    return stats


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    import seaborn as sns
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    # Calculate PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pr = np.trapz(precision, recall)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {auc_pr:.3f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='r', linestyle='--', linewidth=1, 
               label=f'Random Classifier ({baseline:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def evaluate_model_performance(
    model,
    test_loader,
    device: str = 'cpu',
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        threshold: Decision threshold
        
    Returns:
        Dictionary with performance metrics
    """
    import torch
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            
            # Convert outputs to probabilities
            if outputs.shape[1] == 1:
                # Binary classification
                probs = torch.sigmoid(outputs).squeeze()
            else:
                # Multi-class classification
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            
            all_probabilities.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_scores = np.array(all_probabilities)
    
    # Calculate metrics
    metrics = calculate_detection_metrics(y_true, y_scores, threshold=threshold)
    
    return metrics
