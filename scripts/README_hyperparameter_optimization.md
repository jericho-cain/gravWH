# 🔍 Hyperparameter Optimization for CWT-LSTM Autoencoder

This directory contains scripts for systematically optimizing the hyperparameters of the CWT-LSTM Autoencoder to achieve the best possible performance for gravitational wave detection.

## 📁 Files Overview

- **`hyperparameter_grid_search.py`** - Main grid search implementation
- **`analyze_hyperparameter_results.py`** - Results analysis and visualization
- **`run_hyperparameter_optimization.py`** - Complete optimization pipeline
- **`quick_hyperparameter_test.py`** - Quick test with fewer combinations

## 🚀 Quick Start

### Option 1: Quick Test (Recommended First)
```bash
python scripts/quick_hyperparameter_test.py
```
- Tests 32 combinations (2×2×2×2×2)
- Takes ~10-15 minutes
- Good for initial validation

### Option 2: Full Grid Search
```bash
python scripts/run_hyperparameter_optimization.py
```
- Tests 960 combinations (4×4×3×3×3)
- Takes ~4-6 hours
- Comprehensive optimization

## 📊 Hyperparameters Tested

| Parameter | Values | Description |
|-----------|--------|-------------|
| `latent_dim` | [8, 16, 32, 64] | Latent space dimension |
| `lstm_hidden` | [16, 32, 64, 128] | LSTM hidden layer size |
| `learning_rate` | [0.0001, 0.001, 0.01] | Adam optimizer learning rate |
| `batch_size` | [4, 8, 16] | Training batch size |
| `epochs` | [20, 30, 50] | Number of training epochs |

## 📈 Metrics Tracked

The optimization tracks all 8 key metrics from the paper:

1. **`opt_precision`** - Best ≥90% precision
2. **`opt_recall`** - Recall at optimal precision
3. **`max_precision`** - Maximum precision achieved
4. **`max_recall`** - Recall at max precision
5. **`f1_precision`** - F1-optimal precision
6. **`f1_recall`** - F1-optimal recall
7. **`auc`** - AUC-ROC score
8. **`avg_precision`** - Average precision (primary metric)

## 📁 Output Structure

```
hyperparameter_results/
├── results_summary.csv          # Main results table
├── detailed_results.json        # Complete results with all data
├── best_config.json            # Best configuration found
└── analysis/                   # Visualization plots
    ├── avg_precision_heatmap.png
    ├── learning_rate_analysis.png
    ├── batch_size_analysis.png
    └── epochs_analysis.png
```

## 🎯 Finding the Best Configuration

### Method 1: CSV Table
1. Open `hyperparameter_results/results_summary.csv`
2. Sort by `avg_precision` column (descending)
3. The top row contains your optimal hyperparameters!

### Method 2: JSON File
```bash
cat hyperparameter_results/best_config.json
```

### Method 3: Analysis Script
```bash
python scripts/analyze_hyperparameter_results.py
```

## 🔧 Customizing the Grid Search

To modify the hyperparameter ranges, edit the `hyperparameter_grid` in `hyperparameter_grid_search.py`:

```python
self.hyperparameter_grid = {
    'latent_dim': [8, 16, 32, 64],        # Add/remove values
    'lstm_hidden': [16, 32, 64, 128],     # Add/remove values
    'learning_rate': [0.0001, 0.001, 0.01], # Add/remove values
    'batch_size': [4, 8, 16],             # Add/remove values
    'epochs': [20, 30, 50]                # Add/remove values
}
```

## ⚡ Performance Tips

- **Start with quick test** to validate the setup
- **Use fewer samples** for faster testing (modify `num_samples` parameter)
- **Run overnight** for full grid search
- **Monitor progress** - results are saved every 10 experiments

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `num_samples`
2. **Slow Training**: Reduce `epochs` for testing
3. **Import Errors**: Make sure you're in the project root directory

### Debug Mode
Add debug prints in `_run_single_experiment()` method to track progress.

## 📊 Expected Results

Based on initial testing, you should expect:
- **Best avg_precision**: 0.75-0.85
- **Best opt_precision**: 0.90-0.95
- **Best auc**: 0.80-0.90
- **Training time per experiment**: 2-5 minutes

## 🎉 Next Steps

After finding the optimal hyperparameters:

1. **Update the main model** with best configuration
2. **Re-run the paper results** with optimized parameters
3. **Update the paper** with new performance metrics
4. **Consider additional optimizations** (architecture changes, data augmentation, etc.)

## 📝 Notes

- Each experiment uses the same random seed for reproducibility
- Results are automatically saved to prevent data loss
- The grid search can be interrupted and resumed (though not implemented yet)
- All figures and metrics are generated for each configuration
