#!/usr/bin/env python3
"""
Automated paper results updater
Updates LaTeX placeholders with latest model performance metrics
"""

import json
import re
import os
import shutil
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def update_paper_results(results):
    """
    Update LaTeX paper with new results using named placeholders
    
    Args:
        results (dict): Dictionary containing performance metrics
    """
    
    # Save results to JSON for tracking
    results_file = "paper/data/results.json"
    
    # Load existing results for comparison
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            old_results = json.load(f)
        
        # Check if we beat previous results
        if results['opt_precision'] > old_results.get('opt_precision', 0):
            logger.info(f"NEW RECORD! Precision: {old_results.get('opt_precision', 0):.1%} → {results['opt_precision']:.1%}")
        else:
            logger.info(f"Updating results: {results['opt_precision']:.1%} precision")
    
    # Save new results with timestamp
    results['timestamp'] = datetime.now().isoformat()
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Update LaTeX placeholders
    update_latex_placeholders(results)
    
    # Copy latest plots
    copy_latest_plots()
    
    logger.info("✅ Paper updated successfully!")

def update_latex_placeholders(results):
    """
    Replace named placeholders in results.tex
    """
    
    results_tex = "paper/sections/results.tex"
    
    # Read the existing LaTeX file
    with open(results_tex, 'r') as f:
        latex_content = f.read()
    
    # Define placeholder mappings with proper escaping
    # Format percentages first to avoid regex issues
    opt_prec = f"{results['opt_precision']:.1%}".replace('%', '\\%')
    opt_rec = f"{results['opt_recall']:.1%}".replace('%', '\\%')
    max_prec = f"{results['max_precision']:.1%}".replace('%', '\\%')
    max_rec = f"{results['max_recall']:.1%}".replace('%', '\\%')
    f1_prec = f"{results['f1_precision']:.1%}".replace('%', '\\%')
    f1_rec = f"{results['f1_recall']:.1%}".replace('%', '\\%')
    
    replacements = {
        r'\\newcommand\{\\OPTPRECISION\}\{[^}]*\}': f'\\\\newcommand{{\\\\OPTPRECISION}}{{{opt_prec}}}',
        r'\\newcommand\{\\OPTRECALL\}\{[^}]*\}': f'\\\\newcommand{{\\\\OPTRECALL}}{{{opt_rec}}}',
        r'\\newcommand\{\\MAXPRECISION\}\{[^}]*\}': f'\\\\newcommand{{\\\\MAXPRECISION}}{{{max_prec}}}',
        r'\\newcommand\{\\MAXRECALL\}\{[^}]*\}': f'\\\\newcommand{{\\\\MAXRECALL}}{{{max_rec}}}',
        r'\\newcommand\{\\FONEPREC\}\{[^}]*\}': f'\\\\newcommand{{\\\\FONEPREC}}{{{f1_prec}}}',
        r'\\newcommand\{\\FONERECALL\}\{[^}]*\}': f'\\\\newcommand{{\\\\FONERECALL}}{{{f1_rec}}}',
        r'\\newcommand\{\\AUCVALUE\}\{[^}]*\}': f'\\\\newcommand{{\\\\AUCVALUE}}{{{results["auc"]:.3f}}}',
        r'\\newcommand\{\\AVGPRECISION\}\{[^}]*\}': f'\\\\newcommand{{\\\\AVGPRECISION}}{{{results["avg_precision"]:.3f}}}'
    }
    
    # Apply replacements
    for pattern, replacement in replacements.items():
        latex_content = re.sub(pattern, replacement, latex_content)
    
    # Add timestamp comment
    timestamp_comment = f"% Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    latex_content = re.sub(r'% Last updated:.*', timestamp_comment, latex_content)
    
    # Write back the updated content
    with open(results_tex, 'w') as f:
        f.write(latex_content)
    
    logger.info("✅ Updated LaTeX placeholders")

def copy_latest_plots():
    """
    Copy latest plots from results/ to paper/figures/
    """
    
    plot_mappings = {
        "results/precision_recall_curve.png": "paper/figures/precision_recall_curve.png",
        "results/roc_curve.png": "paper/figures/roc_curve.png", 
        "results/snr_performance_standalone.png": "paper/figures/snr_performance.png"
    }
    
    os.makedirs("paper/figures", exist_ok=True)
    
    for src, dst in plot_mappings.items():
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"✅ Copied {src} → {dst}")
        else:
            logger.warning(f"⚠️ Plot not found: {src}")

def compile_paper():
    """
    Compile LaTeX paper to PDF (requires LaTeX installation)
    """
    
    if not os.path.exists("paper/main.tex"):
        logger.error("❌ main.tex not found")
        return
    
    original_dir = os.getcwd()
    
    try:
        os.chdir("paper")
        
        # Run LaTeX compilation sequence
        commands = [
            "pdflatex main.tex",
            "pdflatex main.tex"  # Second pass for references
        ]
        
        for cmd in commands:
            result = os.system(cmd)
            if result != 0:
                logger.warning(f"⚠️ LaTeX compilation warning for: {cmd}")
        
        if os.path.exists("main.pdf"):
            logger.info("✅ Paper compiled to main.pdf")
        else:
            logger.warning("⚠️ PDF not generated - check LaTeX installation")
            
    finally:
        os.chdir(original_dir)

# Example usage and integration point
def update_from_model_results(precision, recall, auc, max_precision, max_precision_recall, 
                             f1_precision, f1_recall, avg_precision):
    """
    Convenience function to update paper from model output
    """
    
    results = {
        'opt_precision': precision,
        'opt_recall': recall,
        'max_precision': max_precision,
        'max_recall': max_precision_recall,
        'f1_precision': f1_precision,
        'f1_recall': f1_recall,
        'auc': auc,
        'avg_precision': avg_precision
    }
    
    update_paper_results(results)

if __name__ == "__main__":
    # Example with your current results
    results = {
        'opt_precision': 0.923,    # Best ≥90% precision
        'opt_recall': 0.676,       # Recall at optimal
        'max_precision': 1.000,    # Maximum precision  
        'max_recall': 0.014,       # Recall at max
        'f1_precision': 0.893,     # F1-optimal precision
        'f1_recall': 0.704,        # F1-optimal recall
        'auc': 0.821,              # AUC score
        'avg_precision': 0.788     # Average precision
    }
    
    logger.info("Updating paper with results...")
    update_paper_results(results)
    
    # Optionally compile paper
    # compile_paper()
