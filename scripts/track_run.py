#!/usr/bin/env python3
"""
Run tracking system for gravitational wave detection experiments.

This script automatically logs successful and failed runs with their
hyperparameters and results to a JSON file for analysis and comparison.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RunTracker:
    """Tracks experimental runs and their results."""
    
    def __init__(self, history_file: str = "results/run_history.json"):
        self.history_file = history_file
        self.ensure_history_file()
    
    def ensure_history_file(self):
        """Ensure the history file exists with proper structure."""
        if not os.path.exists(self.history_file):
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            initial_data = {
                "run_history": [],
                "best_performance": {},
                "failed_runs": []
            }
            with open(self.history_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def log_successful_run(
        self,
        hyperparameters: Dict[str, Any],
        results: Dict[str, Any],
        notes: str = ""
    ) -> str:
        """Log a successful run and return the run ID."""
        run_id = self._generate_run_id()
        
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "hyperparameters": hyperparameters,
            "results": results,
            "notes": notes
        }
        
        self._add_to_history(run_data)
        self._update_best_performance(run_data)
        
        logger.info(f"Logged successful run {run_id}: AUC={results.get('auc', 'N/A')}, AP={results.get('ap', 'N/A')}")
        return run_id
    
    def log_failed_run(
        self,
        hyperparameters: Dict[str, Any],
        error: str,
        notes: str = ""
    ) -> str:
        """Log a failed run and return the run ID."""
        run_id = self._generate_run_id()
        
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "hyperparameters": hyperparameters,
            "error": error,
            "notes": notes
        }
        
        self._add_to_failed_runs(run_data)
        
        logger.warning(f"Logged failed run {run_id}: {error}")
        return run_id
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        with open(self.history_file, 'r') as f:
            data = json.load(f)
        
        # Count existing runs
        total_runs = len(data.get("run_history", [])) + len(data.get("failed_runs", []))
        return f"run_{total_runs + 1:03d}"
    
    def _add_to_history(self, run_data: Dict[str, Any]):
        """Add run data to the history."""
        with open(self.history_file, 'r') as f:
            data = json.load(f)
        
        # Ensure run_history exists
        if "run_history" not in data:
            data["run_history"] = []
        
        # Append new run (never overwrites)
        data["run_history"].append(run_data)
        
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _add_to_failed_runs(self, run_data: Dict[str, Any]):
        """Add failed run data to the failed runs list."""
        with open(self.history_file, 'r') as f:
            data = json.load(f)
        
        # Ensure failed_runs exists
        if "failed_runs" not in data:
            data["failed_runs"] = []
        
        # Append new failed run (never overwrites)
        data["failed_runs"].append(run_data)
        
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_best_performance(self, run_data: Dict[str, Any]):
        """Update the best performance record if this run is better."""
        with open(self.history_file, 'r') as f:
            data = json.load(f)
        
        current_best = data.get("best_performance", {})
        current_auc = current_best.get("auc", 0)
        current_ap = current_best.get("ap", 0)
        
        new_auc = run_data["results"].get("auc", 0)
        new_ap = run_data["results"].get("ap", 0)
        
        # Update if this run has better AUC or AP
        if new_auc > current_auc or (new_auc == current_auc and new_ap > current_ap):
            data["best_performance"] = {
                "run_id": run_data["run_id"],
                "auc": new_auc,
                "ap": new_ap,
                "samples": run_data["hyperparameters"].get("training_samples", 0),
                "key_optimizations": self._extract_key_optimizations(run_data)
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"🏆 New best performance! AUC={new_auc:.3f}, AP={new_ap:.3f}")
    
    def _extract_key_optimizations(self, run_data: Dict[str, Any]) -> list:
        """Extract key optimizations from hyperparameters."""
        optimizations = []
        hyperparams = run_data["hyperparameters"]
        
        if hyperparams.get("cwt_height") == 8:
            optimizations.append("Reduced CWT dimensions: 16x4096 → 8x4096")
        
        if hyperparams.get("optimizer") == "SGD":
            optimizations.append("SGD optimizer instead of Adam")
        
        if hyperparams.get("mixed_precision"):
            optimizations.append("Mixed Precision Training (AMP)")
        
        if hyperparams.get("memory_cleanup"):
            optimizations.append("Comprehensive memory cleanup")
        
        return optimizations
    
    def get_best_performance(self) -> Dict[str, Any]:
        """Get the current best performance."""
        with open(self.history_file, 'r') as f:
            data = json.load(f)
        return data.get("best_performance", {})
    
    def get_run_summary(self) -> str:
        """Get a summary of all runs."""
        with open(self.history_file, 'r') as f:
            data = json.load(f)
        
        successful_runs = len(data.get("run_history", []))
        failed_runs = len(data.get("failed_runs", []))
        best = data.get("best_performance", {})
        
        summary = f"""
Run Summary:
- Successful runs: {successful_runs}
- Failed runs: {failed_runs}
- Best performance: AUC={best.get('auc', 'N/A')}, AP={best.get('ap', 'N/A')}
- Best run: {best.get('run_id', 'N/A')} with {best.get('samples', 'N/A')} samples
"""
        return summary
    
    def get_complete_history(self) -> Dict[str, Any]:
        """Get the complete run history."""
        with open(self.history_file, 'r') as f:
            data = json.load(f)
        return data
    
    def print_recent_runs(self, n: int = 5):
        """Print the most recent runs."""
        with open(self.history_file, 'r') as f:
            data = json.load(f)
        
        successful_runs = data.get("run_history", [])
        failed_runs = data.get("failed_runs", [])
        
        # Combine and sort by timestamp
        all_runs = successful_runs + failed_runs
        all_runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        print(f"\nRecent {min(n, len(all_runs))} runs:")
        for i, run in enumerate(all_runs[:n]):
            status_text = "SUCCESS" if run["status"] == "success" else "FAILED"
            run_id = run["run_id"]
            timestamp = run["timestamp"]
            
            if run["status"] == "success":
                auc = run["results"].get("auc", "N/A")
                ap = run["results"].get("ap", "N/A")
                samples = run["hyperparameters"].get("training_samples", "N/A")
                print(f"  {status_text} {run_id}: AUC={auc}, AP={ap}, Samples={samples} ({timestamp})")
            else:
                error = run.get("error", "Unknown error")
                samples = run["hyperparameters"].get("training_samples", "N/A")
                print(f"  {status_text} {run_id}: FAILED - {error}, Samples={samples} ({timestamp})")

# Example usage
if __name__ == "__main__":
    tracker = RunTracker()
    
    # Example successful run
    hyperparams = {
        "training_samples": 100,
        "test_samples": 25,
        "cwt_height": 8,
        "cwt_width": 4096,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "epochs": 20,
        "batch_size": 1,
        "latent_dim": 32,
        "mixed_precision": True,
        "memory_cleanup": True
    }
    
    results = {
        "auc": 0.921,
        "ap": 0.854,
        "training_time": "~6 seconds",
        "memory_usage": "stable",
        "crashes": 0
    }
    
    run_id = tracker.log_successful_run(
        hyperparams, 
        results, 
        "Major breakthrough with reduced CWT dimensions"
    )
    
    print(tracker.get_run_summary())
