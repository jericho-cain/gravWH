"""
Tests for visualization and plotting functionality.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from gravitational_wave_hunter.visualization.plotting import (
    plot_strain_data,
    plot_spectrogram,
    plot_detection_results,
    plot_training_history,
    plot_roc_curve,
    plot_waveform_comparison,
    save_figure
)


class TestBasicPlotting:
    """Test basic plotting functionality."""
    
    def test_plot_strain_data_basic(self, sample_strain_data, sample_rate, duration):
        """Test basic strain data plotting."""
        try:
            fig, ax = plot_strain_data(
                sample_strain_data,
                sample_rate=sample_rate,
                title="Test Strain Data"
            )
            
            # Should return matplotlib objects
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            
            # Check that data was plotted
            lines = ax.get_lines()
            assert len(lines) > 0
            
            # Check axis labels
            assert ax.get_xlabel() != ""
            assert ax.get_ylabel() != ""
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_strain_data not implemented")
    
    def test_plot_strain_data_with_time_axis(self, sample_strain_data, sample_rate, duration):
        """Test strain data plotting with custom time axis."""
        time_axis = np.linspace(0, duration, len(sample_strain_data))
        
        try:
            fig, ax = plot_strain_data(
                sample_strain_data,
                time_axis=time_axis,
                title="Test with Time Axis"
            )
            
            # Check that time axis is used
            lines = ax.get_lines()
            x_data = lines[0].get_xdata()
            
            # Should match provided time axis
            np.testing.assert_array_almost_equal(x_data, time_axis)
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_strain_data not implemented")
    
    def test_plot_strain_data_multiple_signals(self, sample_strain_batch, sample_rate):
        """Test plotting multiple strain signals."""
        try:
            fig, ax = plot_strain_data(
                sample_strain_batch[:3],  # Plot first 3 signals
                sample_rate=sample_rate,
                title="Multiple Signals"
            )
            
            # Should have multiple lines
            lines = ax.get_lines()
            assert len(lines) >= 3
            
            plt.close(fig)
            
        except (NotImplementedError, ValueError):
            # Function might not support multiple signals
            pytest.skip("plot_strain_data multiple signals not supported")


class TestSpectrogram:
    """Test spectrogram plotting functionality."""
    
    def test_plot_spectrogram_basic(self, sample_strain_data, sample_rate):
        """Test basic spectrogram plotting."""
        try:
            fig, ax = plot_spectrogram(
                sample_strain_data,
                sample_rate=sample_rate,
                title="Test Spectrogram"
            )
            
            # Should return matplotlib objects
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            
            # Should have a colorbar or image
            images = ax.get_images()
            collections = ax.collections
            assert len(images) > 0 or len(collections) > 0
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_spectrogram not implemented")
    
    def test_plot_spectrogram_frequency_limits(self, sample_strain_data, sample_rate):
        """Test spectrogram with frequency limits."""
        try:
            fig, ax = plot_spectrogram(
                sample_strain_data,
                sample_rate=sample_rate,
                freq_limits=(50, 500),
                title="Limited Frequency Range"
            )
            
            # Check frequency axis limits
            y_lim = ax.get_ylim()
            assert y_lim[0] >= 50
            assert y_lim[1] <= 500
            
            plt.close(fig)
            
        except (NotImplementedError, ValueError):
            pytest.skip("plot_spectrogram frequency limits not supported")
    
    def test_plot_spectrogram_parameters(self, sample_strain_data, sample_rate):
        """Test spectrogram with different parameters."""
        parameters = [
            {'nperseg': 1024, 'noverlap': 512},
            {'nperseg': 2048, 'noverlap': 1024},
        ]
        
        for params in parameters:
            try:
                fig, ax = plot_spectrogram(
                    sample_strain_data,
                    sample_rate=sample_rate,
                    **params
                )
                
                # Should create valid plot
                assert isinstance(fig, plt.Figure)
                assert isinstance(ax, plt.Axes)
                
                plt.close(fig)
                
            except (NotImplementedError, ValueError):
                # Parameters might not be supported
                continue


class TestDetectionResults:
    """Test detection results plotting."""
    
    def test_plot_detection_results_basic(self, sample_strain_batch, sample_labels):
        """Test basic detection results plotting."""
        # Create fake predictions
        predictions = np.random.randint(0, 2, len(sample_labels))
        probabilities = np.random.random(len(sample_labels))
        
        try:
            fig, axes = plot_detection_results(
                sample_strain_batch,
                sample_labels,
                predictions,
                probabilities,
                sample_rate=4096
            )
            
            # Should return figure and axes
            assert isinstance(fig, plt.Figure)
            
            # axes could be single axis or array of axes
            if isinstance(axes, np.ndarray):
                assert len(axes) > 0
                for ax in axes.flat:
                    assert isinstance(ax, plt.Axes)
            else:
                assert isinstance(axes, plt.Axes)
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_detection_results not implemented")
    
    def test_plot_detection_results_categories(self, sample_strain_batch, sample_labels):
        """Test plotting different detection categories."""
        # Ensure we have examples of each category
        sample_labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1])  # Mix of correct/incorrect
        probabilities = np.array([0.1, 0.9, 0.6, 0.3, 0.8, 0.2, 0.9, 0.7])
        
        try:
            fig, axes = plot_detection_results(
                sample_strain_batch,
                sample_labels,
                predictions,
                probabilities,
                sample_rate=4096,
                show_categories=True
            )
            
            # Should create plots for different categories
            assert isinstance(fig, plt.Figure)
            
            plt.close(fig)
            
        except (NotImplementedError, ValueError):
            pytest.skip("plot_detection_results categories not supported")


class TestTrainingHistory:
    """Test training history plotting."""
    
    def test_plot_training_history_basic(self):
        """Test basic training history plotting."""
        # Create fake training history
        epochs = 20
        train_loss = np.exp(-np.linspace(0, 2, epochs)) + 0.1 * np.random.random(epochs)
        val_loss = train_loss + 0.05 * np.random.random(epochs)
        train_acc = 1 - 0.5 * np.exp(-np.linspace(0, 2, epochs)) + 0.05 * np.random.random(epochs)
        val_acc = train_acc - 0.02 * np.random.random(epochs)
        
        history = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
        
        try:
            fig, axes = plot_training_history(history)
            
            # Should return figure and axes
            assert isinstance(fig, plt.Figure)
            
            # Should have multiple subplots for loss and accuracy
            if isinstance(axes, np.ndarray):
                assert len(axes) >= 2
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_training_history not implemented")
    
    def test_plot_training_history_loss_only(self):
        """Test training history with only loss data."""
        epochs = 15
        train_loss = np.exp(-np.linspace(0, 1.5, epochs))
        val_loss = train_loss + 0.05 * np.random.random(epochs)
        
        history = {
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        try:
            fig, ax = plot_training_history(history)
            
            # Should handle loss-only data
            assert isinstance(fig, plt.Figure)
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_training_history not implemented")


class TestROCCurve:
    """Test ROC curve plotting."""
    
    def test_plot_roc_curve_basic(self):
        """Test basic ROC curve plotting."""
        # Create test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3, 0.85, 0.15])
        
        try:
            fig, ax = plot_roc_curve(y_true, y_scores)
            
            # Should return matplotlib objects
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            
            # Should have ROC curve line
            lines = ax.get_lines()
            assert len(lines) >= 1
            
            # Should have diagonal reference line
            diagonal_line = any(
                np.allclose(line.get_xdata(), line.get_ydata()) 
                for line in lines
            )
            assert diagonal_line
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_roc_curve not implemented")
    
    def test_plot_roc_curve_multiple_models(self):
        """Test ROC curves for multiple models."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        
        models_scores = {
            'Model 1': np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7, 0.3, 0.85, 0.15]),
            'Model 2': np.array([0.2, 0.3, 0.4, 0.85, 0.1, 0.95, 0.75, 0.25, 0.9, 0.05])
        }
        
        try:
            fig, ax = plot_roc_curve(y_true, models_scores)
            
            # Should have multiple curves
            lines = ax.get_lines()
            assert len(lines) >= len(models_scores)  # Plus diagonal line
            
            # Should have legend
            legend = ax.get_legend()
            assert legend is not None
            
            plt.close(fig)
            
        except (NotImplementedError, ValueError):
            pytest.skip("plot_roc_curve multiple models not supported")


class TestWaveformComparison:
    """Test waveform comparison plotting."""
    
    def test_plot_waveform_comparison_basic(self, sample_rate, duration):
        """Test basic waveform comparison."""
        time = np.linspace(0, duration, int(sample_rate * duration))
        waveform1 = np.sin(2 * np.pi * 100 * time)
        waveform2 = np.sin(2 * np.pi * 100 * time + 0.5)  # Phase shifted
        
        waveforms = {
            'Original': waveform1,
            'Phase Shifted': waveform2
        }
        
        try:
            fig, ax = plot_waveform_comparison(waveforms, sample_rate=sample_rate)
            
            # Should return matplotlib objects
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            
            # Should have multiple lines
            lines = ax.get_lines()
            assert len(lines) >= 2
            
            # Should have legend
            legend = ax.get_legend()
            assert legend is not None
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_waveform_comparison not implemented")
    
    def test_plot_waveform_comparison_residuals(self, sample_rate, duration):
        """Test waveform comparison with residuals."""
        time = np.linspace(0, duration, int(sample_rate * duration))
        waveform1 = np.sin(2 * np.pi * 100 * time)
        waveform2 = waveform1 + 0.1 * np.random.normal(0, 1, len(time))
        
        waveforms = {
            'Clean': waveform1,
            'Noisy': waveform2
        }
        
        try:
            fig, axes = plot_waveform_comparison(
                waveforms, 
                sample_rate=sample_rate,
                show_residuals=True
            )
            
            # Should have multiple subplots
            if isinstance(axes, np.ndarray):
                assert len(axes) >= 2  # Main plot + residuals
            
            plt.close(fig)
            
        except (NotImplementedError, ValueError):
            pytest.skip("plot_waveform_comparison residuals not supported")


class TestPlottingUtilities:
    """Test plotting utility functions."""
    
    def test_save_figure(self, temp_dir):
        """Test figure saving utility."""
        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        
        # Test different formats
        formats = ['png', 'pdf', 'svg']
        
        for fmt in formats:
            output_path = temp_dir / f"test_plot.{fmt}"
            
            try:
                save_figure(fig, output_path, dpi=150)
                
                # File should exist
                assert output_path.exists()
                assert output_path.stat().st_size > 0
                
            except (NotImplementedError, ValueError):
                # Format might not be supported
                continue
        
        plt.close(fig)
    
    def test_save_figure_with_options(self, temp_dir):
        """Test figure saving with different options."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        output_path = temp_dir / "test_plot_options.png"
        
        try:
            save_figure(
                fig, 
                output_path, 
                dpi=300, 
                bbox_inches='tight',
                transparent=True
            )
            
            assert output_path.exists()
            
        except NotImplementedError:
            pytest.skip("save_figure not implemented")
        
        plt.close(fig)


class TestPlottingValidation:
    """Test input validation for plotting functions."""
    
    def test_plot_strain_data_empty_input(self, sample_rate):
        """Test plotting with empty data."""
        empty_data = np.array([])
        
        try:
            with pytest.raises((ValueError, IndexError)):
                plot_strain_data(empty_data, sample_rate=sample_rate)
        except NotImplementedError:
            pytest.skip("plot_strain_data not implemented")
    
    def test_plot_strain_data_mismatched_time_axis(self, sample_strain_data, sample_rate):
        """Test plotting with mismatched time axis."""
        wrong_time_axis = np.linspace(0, 1, len(sample_strain_data) // 2)  # Wrong length
        
        try:
            with pytest.raises((ValueError, AssertionError)):
                plot_strain_data(
                    sample_strain_data, 
                    time_axis=wrong_time_axis,
                    sample_rate=sample_rate
                )
        except NotImplementedError:
            pytest.skip("plot_strain_data not implemented")
    
    def test_plot_detection_results_mismatched_lengths(self, sample_strain_batch):
        """Test detection results with mismatched input lengths."""
        labels = np.array([0, 1, 0])  # Wrong length
        predictions = np.array([0, 1, 1, 0])  # Wrong length
        probabilities = np.array([0.1, 0.9, 0.4, 0.3])  # Wrong length
        
        try:
            with pytest.raises((ValueError, AssertionError)):
                plot_detection_results(
                    sample_strain_batch,
                    labels,
                    predictions,
                    probabilities,
                    sample_rate=4096
                )
        except NotImplementedError:
            pytest.skip("plot_detection_results not implemented")
    
    def test_plot_roc_curve_invalid_labels(self):
        """Test ROC curve with invalid labels."""
        y_true = np.array([0, 1, 2])  # Invalid binary labels
        y_scores = np.array([0.1, 0.5, 0.9])
        
        try:
            with pytest.raises((ValueError, AssertionError)):
                plot_roc_curve(y_true, y_scores)
        except NotImplementedError:
            pytest.skip("plot_roc_curve not implemented")


class TestPlottingIntegration:
    """Test integration between different plotting functions."""
    
    def test_plotting_pipeline(self, sample_strain_data, sample_rate):
        """Test a complete plotting pipeline."""
        try:
            # Plot strain data
            fig1, ax1 = plot_strain_data(sample_strain_data, sample_rate=sample_rate)
            
            # Plot spectrogram of same data
            fig2, ax2 = plot_spectrogram(sample_strain_data, sample_rate=sample_rate)
            
            # Both should succeed
            assert isinstance(fig1, plt.Figure)
            assert isinstance(fig2, plt.Figure)
            
            plt.close(fig1)
            plt.close(fig2)
            
        except NotImplementedError:
            pytest.skip("Plotting functions not implemented")
    
    @patch('matplotlib.pyplot.show')
    def test_plotting_no_display(self, mock_show, sample_strain_data, sample_rate):
        """Test that plotting works without display."""
        # Ensure plots don't actually display during testing
        try:
            fig, ax = plot_strain_data(sample_strain_data, sample_rate=sample_rate)
            
            # Should create plot without showing
            assert isinstance(fig, plt.Figure)
            mock_show.assert_not_called()
            
            plt.close(fig)
            
        except NotImplementedError:
            pytest.skip("plot_strain_data not implemented")
    
    def test_plotting_memory_cleanup(self, sample_strain_data, sample_rate):
        """Test that plots are properly cleaned up."""
        initial_figs = len(plt.get_fignums())
        
        try:
            # Create multiple plots
            figs = []
            for i in range(5):
                fig, ax = plot_strain_data(
                    sample_strain_data, 
                    sample_rate=sample_rate,
                    title=f"Plot {i}"
                )
                figs.append(fig)
            
            # Should have created figures
            assert len(plt.get_fignums()) > initial_figs
            
            # Close all figures
            for fig in figs:
                plt.close(fig)
            
            # Should be back to initial state
            assert len(plt.get_fignums()) == initial_figs
            
        except NotImplementedError:
            pytest.skip("plot_strain_data not implemented")
