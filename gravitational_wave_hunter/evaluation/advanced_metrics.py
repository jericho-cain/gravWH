"""
Advanced metrics computation with interpolation and downsampling for gravitational wave detection.

Adapted from sophisticated metrics class that handles smooth ROC/PR curves and 
comprehensive evaluation metrics with linear interpolation.
"""

from typing import Sequence, Union
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, roc_curve

# Silence sklearn warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn


class GravitationalWaveMetrics:
    """
    Advanced metrics computing object for gravitational wave detection.
    
    Handles downsampling of ROC/PR curves, comprehensive metrics computation,
    and linear interpolation for threshold selection.
    """

    def __init__(
        self,
        y_true: Union[list, np.ndarray],
        y_score: Union[list, np.ndarray],
        pos_label: int = 1,
    ):
        """
        Initialize the metrics object for gravitational wave detection.

        Args:
            y_true: True binary labels (0=noise, 1=signal)
            y_score: Model scores for positive class (reconstruction errors)
            pos_label: Positive label (1 for gravitational wave signals)
        """
        self.y_true = np.array(y_true)
        self.y_score = np.array(y_score)
        self.pos_label = pos_label
        
        # Compute ROC curve
        self.fpr, self.tpr, self.roc_thr = roc_curve(
            self.y_true, self.y_score, pos_label=pos_label, drop_intermediate=True
        )
        
        # Compute AUC
        self.roc_auc = auc(self.fpr, self.tpr)
        
        # Compute Average Precision
        self.labels = (self.y_true == pos_label)
        self.avg_precision = average_precision_score(self.labels, self.y_score)
        
        # Downsample curves for smooth plotting
        self._downsample_curves()
        
        # Compute comprehensive metrics
        self._compute_confusion_matrices()
        self._compute_all_metrics()
        
        # Create metrics dataframe
        self._create_metrics_dataframe()

    def _downsample_curves(self):
        """Create smooth ROC curve with interpolation."""
        # For small datasets, create more points through interpolation
        if len(self.fpr) < 20:  # Small dataset - interpolate for smooth curves
            # Create more FPR points for smooth interpolation
            fpr_interp = np.linspace(0, 1, 100)
            tpr_interp = np.interp(fpr_interp, self.fpr, self.tpr)
            
            # Create corresponding thresholds (approximate)
            thresholds_interp = np.interp(fpr_interp, self.fpr, self.roc_thr)
            
            self.fpr_ds = fpr_interp
            self.tpr_ds = tpr_interp
            self.roc_thr_ds = thresholds_interp
        else:
            # Large dataset - use downsampling
            ds_indices = [0]
            fpr_ds, tpr_ds, roc_thr_ds = [self.fpr[0]], [self.tpr[0]], [self.roc_thr[0]]
            
            n = 1
            for i, j, k in zip(self.fpr[1:-1], self.tpr[1:-1], self.roc_thr[1:-1]):
                # Distance threshold: 0.005 for smooth curves
                if ((i - fpr_ds[-1]) ** 2 + (j - tpr_ds[-1]) ** 2) ** 0.5 >= 0.005:
                    fpr_ds.append(i)
                    tpr_ds.append(j)
                    roc_thr_ds.append(k)
                    ds_indices.append(n)
                    n += 1
            
            # Always include the last point
            fpr_ds.append(self.fpr[-1])
            tpr_ds.append(self.tpr[-1])
            roc_thr_ds.append(self.roc_thr[-1])
            ds_indices.append(len(self.y_score) - 1)
            
            self.fpr_ds = np.array(fpr_ds)
            self.tpr_ds = np.array(tpr_ds)
            self.roc_thr_ds = np.array(roc_thr_ds)
            self.ds_indices = ds_indices

    def _compute_confusion_matrices(self):
        """Compute confusion matrices at each threshold."""
        self.cm_data = {"tps": [], "tns": [], "fns": [], "fps": []}
        
        n_pos = np.sum(self.labels)
        n_neg = len(self.labels) - n_pos
        
        for threshold in self.roc_thr_ds:
            preds = self.y_score >= threshold
            
            tps = np.sum(self.labels & preds)
            fns = n_pos - tps
            fps = np.sum(~self.labels & preds)
            tns = n_neg - fps
            
            self.cm_data["tps"].append(tps)
            self.cm_data["fns"].append(fns)
            self.cm_data["tns"].append(tns)
            self.cm_data["fps"].append(fps)

    def _compute_all_metrics(self):
        """Compute all evaluation metrics at each threshold."""
        tps = np.array(self.cm_data["tps"])
        tns = np.array(self.cm_data["tns"])
        fps = np.array(self.cm_data["fps"])
        fns = np.array(self.cm_data["fns"])
        
        # Precision
        denom = tps + fps
        denom[denom == 0.0] = 1  # Avoid division by zero
        self.precision = tps / denom
        
        # Recall (TPR)
        self.recall = tps / (tps + fns)
        
        # F1 Score
        denom = self.precision + self.recall
        denom[denom == 0.0] = 1
        self.f1 = 2 * self.precision * self.recall / denom
        
        # Accuracy
        self.accuracy = (tps + tns) / (tps + tns + fps + fns)
        
        # Specificity (TNR)
        self.specificity = tns / (tns + fps)
        
        # Negative Predictive Value
        denom = tns + fns
        denom[denom == 0.0] = 1
        self.npv = tns / denom

    def _create_metrics_dataframe(self):
        """Create comprehensive metrics dataframe."""
        self.metrics_df = pd.DataFrame({
            "THRESHOLD": self.roc_thr_ds,
            "FPR": self.fpr_ds,
            "TPR": self.tpr_ds,
            "PRECISION": self.precision,
            "RECALL": self.recall,
            "F1": self.f1,
            "ACCURACY": self.accuracy,
            "SPECIFICITY": self.specificity,
            "NPV": self.npv,
            "TP": self.cm_data["tps"],
            "TN": self.cm_data["tns"],
            "FP": self.cm_data["fps"],
            "FN": self.cm_data["fns"],
            "AUC": self.roc_auc,
            "AP": self.avg_precision
        })

    def get_threshold_at_precision(self, target_precision: float) -> float:
        """
        Get threshold at target precision using linear interpolation.
        
        Args:
            target_precision: Target precision value
            
        Returns:
            Interpolated threshold value
        """
        return self._interpolate_threshold("PRECISION", target_precision)

    def get_threshold_at_recall(self, target_recall: float) -> float:
        """
        Get threshold at target recall using linear interpolation.
        
        Args:
            target_recall: Target recall value
            
        Returns:
            Interpolated threshold value
        """
        return self._interpolate_threshold("RECALL", target_recall)

    def get_threshold_at_fpr(self, target_fpr: float) -> float:
        """
        Get threshold at target FPR using linear interpolation.
        
        Args:
            target_fpr: Target false positive rate
            
        Returns:
            Interpolated threshold value
        """
        return self._interpolate_threshold("FPR", target_fpr)

    def _interpolate_threshold(self, metric_name: str, target_value: float) -> float:
        """Linear interpolation for threshold estimation."""
        df = self.metrics_df[[metric_name, "THRESHOLD"]].sort_values(by=metric_name)
        
        if target_value < df[metric_name].min():
            return None  # Target below achievable range
        elif target_value > df[metric_name].max():
            return None  # Target above achievable range
        else:
            return round(
                np.interp(
                    target_value,
                    df[metric_name],
                    df["THRESHOLD"],
                    left=df["THRESHOLD"].iloc[0],
                    right=df["THRESHOLD"].iloc[-1],
                ),
                4,
            )

    def get_summary_metrics(self) -> dict:
        """Get summary of key metrics."""
        return {
            "auc": self.roc_auc,
            "avg_precision": self.avg_precision,
            "precision": self.precision,
            "recall": self.recall,
            "fpr": self.fpr_ds,
            "tpr": self.tpr_ds,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "specificity": self.specificity,
            "npv": self.npv,
            "thresholds": self.roc_thr_ds
        }

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get complete metrics dataframe."""
        return self.metrics_df.copy()
