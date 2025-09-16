import pytest
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from gravitational_wave_hunter.models.cwt_lstm_autoencoder import SimpleCWTAutoencoder


class TestEvaluation:
    """Test evaluation metrics and performance calculation."""
    
    def test_precision_calculation(self):
        """Test precision calculation."""
        # Create dummy predictions and labels
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        precision = precision_score(y_true, y_pred)
        
        assert isinstance(precision, float)
        assert 0 <= precision <= 1
        assert np.isfinite(precision)
    
    def test_recall_calculation(self):
        """Test recall calculation."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        recall = recall_score(y_true, y_pred)
        
        assert isinstance(recall, float)
        assert 0 <= recall <= 1
        assert np.isfinite(recall)
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        f1 = f1_score(y_true, y_pred)
        
        assert isinstance(f1, float)
        assert 0 <= f1 <= 1
        assert np.isfinite(f1)
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert np.isfinite(accuracy)
    
    def test_reconstruction_error_calculation(self):
        """Test reconstruction error calculation."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        with torch.no_grad():
            reconstructed, latent = model(x)
            mse_error = torch.nn.functional.mse_loss(reconstructed, x)
        
        assert isinstance(mse_error, torch.Tensor)
        assert mse_error.item() >= 0
        assert torch.isfinite(mse_error)
    
    def test_threshold_based_classification(self):
        """Test threshold-based classification from reconstruction error."""
        # Simulate reconstruction errors for noise and signal samples
        noise_errors = np.random.exponential(0.1, 100)  # Small errors for noise
        signal_errors = np.random.exponential(0.5, 100)  # Larger errors for signals
        
        # Combine and create labels
        all_errors = np.concatenate([noise_errors, signal_errors])
        true_labels = np.concatenate([np.zeros(100), np.ones(100)])
        
        # Test different thresholds
        thresholds = [0.2, 0.3, 0.4, 0.5]
        
        for threshold in thresholds:
            predicted_labels = (all_errors > threshold).astype(int)
            
            precision = precision_score(true_labels, predicted_labels)
            recall = recall_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            
            assert 0 <= precision <= 1
            assert 0 <= recall <= 1
            assert 0 <= f1 <= 1
            assert np.isfinite(precision)
            assert np.isfinite(recall)
            assert np.isfinite(f1)
    
    def test_roc_curve_calculation(self):
        """Test ROC curve calculation."""
        from sklearn.metrics import roc_curve, auc
        
        # Create dummy scores and labels
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.6, 0.4])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        assert len(fpr) == len(tpr) == len(thresholds)
        assert 0 <= roc_auc <= 1
        assert np.isfinite(roc_auc)
        assert np.all(fpr >= 0) and np.all(fpr <= 1)
        assert np.all(tpr >= 0) and np.all(tpr <= 1)
    
    def test_precision_recall_curve_calculation(self):
        """Test precision-recall curve calculation."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Create dummy scores and labels
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.6, 0.4])
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        assert len(precision) == len(recall)
        assert len(thresholds) == len(precision) - 1
        assert 0 <= avg_precision <= 1
        assert np.isfinite(avg_precision)
        assert np.all(precision >= 0) and np.all(precision <= 1)
        assert np.all(recall >= 0) and np.all(recall <= 1)
    
    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation."""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert np.all(cm >= 0)
        assert np.sum(cm) == len(y_true)
    
    def test_performance_metrics_consistency(self):
        """Test that performance metrics are consistent."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # F1 should be harmonic mean of precision and recall
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        np.testing.assert_almost_equal(f1, expected_f1, decimal=5)
        
        # All metrics should be finite
        assert all(np.isfinite([precision, recall, f1, accuracy]))
    
    def test_imbalanced_dataset_metrics(self):
        """Test metrics on imbalanced dataset."""
        # Create imbalanced dataset (more noise than signals)
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 noise, 2 signals
        y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])  # 6 correct noise, 1 correct signal
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # All metrics should be finite and in valid ranges
        assert all(0 <= metric <= 1 for metric in [precision, recall, f1, accuracy])
        assert all(np.isfinite([precision, recall, f1, accuracy]))
    
    def test_model_inference_time(self):
        """Test model inference time measurement."""
        import time
        
        model = SimpleCWTAutoencoder(height=64, width=128)
        model.eval()
        
        batch_size = 4
        channels = 1
        height = 64
        width = 128
        x = torch.randn(batch_size, channels, height, width)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            reconstructed, latent = model(x)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        assert inference_time >= 0
        assert np.isfinite(inference_time)
        assert reconstructed.shape == x.shape
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        model = SimpleCWTAutoencoder(height=64, width=128)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate memory usage (rough approximation)
        # Assuming float32 (4 bytes per parameter)
        estimated_memory_mb = (total_params * 4) / (1024 * 1024)
        
        assert total_params > 0
        assert estimated_memory_mb > 0
        assert np.isfinite(estimated_memory_mb)
    
    @pytest.mark.slow
    def test_cross_validation_metrics(self):
        """Test cross-validation metrics calculation."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        # Create dummy dataset for cross-validation
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Use simple classifier for testing
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Test different metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in metrics:
            scores = cross_val_score(clf, X, y, cv=5, scoring=metric)
            
            assert len(scores) == 5
            assert all(0 <= score <= 1 for score in scores)
            assert all(np.isfinite(score) for score in scores)
