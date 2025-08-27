#!/usr/bin/env python3
"""
Detection script for gravitational wave signals.

This script provides a command-line interface for running gravitational wave
detection on data files using pre-trained models.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

from ..detector import GWDetector
from ..data.loader import load_ligo_data, load_virgo_data, load_event_data
from ..utils.config import Config
from ..utils.helpers import setup_logging, ensure_directory, save_results, format_duration
from ..visualization.plotting import plot_detection_results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run gravitational wave detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input data arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--data-file', type=str,
        help='Path to data file (.npy, .h5, etc.)'
    )
    input_group.add_argument(
        '--ligo-data', nargs=3, metavar=('DETECTOR', 'START_TIME', 'DURATION'),
        help='Load LIGO data: detector (H1/L1/V1), GPS start time, duration in seconds'
    )
    input_group.add_argument(
        '--event-data', nargs=2, metavar=('EVENT_NAME', 'DETECTORS'),
        help='Load event data: event name (e.g., GW150914), comma-separated detectors'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--model-type', type=str, default='cnn_lstm',
        choices=['cnn_lstm', 'wavenet', 'transformer', 'autoencoder'],
        help='Type of model (if not specified in model file)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Detection threshold'
    )
    parser.add_argument(
        '--overlap', type=float, default=0.5,
        help='Overlap between detection segments'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=4096,
        help='Sample rate of input data'
    )
    parser.add_argument(
        '--preprocess', action='store_true', default=True,
        help='Apply preprocessing to input data'
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for detection'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default='detection_results',
        help='Directory to save detection results'
    )
    parser.add_argument(
        '--save-plots', action='store_true',
        help='Save detection plots'
    )
    parser.add_argument(
        '--save-data', action='store_true',
        help='Save processed data and detection results'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress non-essential output'
    )
    
    return parser.parse_args()


def load_input_data(args: argparse.Namespace, config: Config = None) -> tuple:
    """Load input data based on command line arguments."""
    
    if args.data_file:
        # Load from file
        data_file = Path(args.data_file)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        if data_file.suffix == '.npy':
            data = np.load(data_file)
        elif data_file.suffix in ['.h5', '.hdf5']:
            import h5py
            with h5py.File(data_file, 'r') as f:
                # Assume data is in 'strain' dataset
                data = f['strain'][:]
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")
        
        data_info = {
            'source': 'file',
            'file_path': str(data_file),
            'duration': len(data) / args.sample_rate,
        }
        
    elif args.ligo_data:
        # Load LIGO data
        detector, start_time_str, duration_str = args.ligo_data
        start_time = int(start_time_str)
        duration = int(duration_str)
        
        if detector.upper() == 'V1':
            data = load_virgo_data(
                start_time=start_time,
                duration=duration,
                sample_rate=args.sample_rate,
                preprocess=args.preprocess,
                config=config
            )
        else:
            data = load_ligo_data(
                detector=detector.upper(),
                start_time=start_time,
                duration=duration,
                sample_rate=args.sample_rate,
                preprocess=args.preprocess,
                config=config
            )
        
        data_info = {
            'source': 'ligo_open_data',
            'detector': detector.upper(),
            'start_time': start_time,
            'duration': duration,
        }
        
    elif args.event_data:
        # Load event data
        event_name, detectors_str = args.event_data
        detectors = [d.strip().upper() for d in detectors_str.split(',')]
        
        event_data = load_event_data(
            event_name=event_name,
            detectors=detectors,
            sample_rate=args.sample_rate,
            preprocess=args.preprocess,
            config=config
        )
        
        # For simplicity, use the first detector's data
        # In practice, you might want to analyze all detectors
        detector = detectors[0]
        data = event_data[detector]
        
        data_info = {
            'source': 'event_data',
            'event_name': event_name,
            'detector': detector,
            'available_detectors': list(event_data.keys()),
            'duration': len(data) / args.sample_rate,
        }
    
    else:
        raise ValueError("No input data specified")
    
    return data, data_info


def setup_detector(args: argparse.Namespace, config: Config = None) -> GWDetector:
    """Set up the gravitational wave detector."""
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create detector
    detector = GWDetector(
        model_type=args.model_type,
        sample_rate=args.sample_rate,
        device=args.device,
        config=config
    )
    
    # Load pre-trained model
    detector.load_pretrained(model_path)
    
    return detector


def run_detection(
    detector: GWDetector,
    data: np.ndarray,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Run gravitational wave detection."""
    
    detection_results = detector.detect(
        strain_data=data,
        threshold=args.threshold,
        overlap=args.overlap,
        preprocess=args.preprocess
    )
    
    return detection_results


def format_detection_results(
    detection_results: Dict[str, Any],
    data_info: Dict[str, Any],
    threshold: float
) -> Dict[str, Any]:
    """Format detection results for output."""
    
    detections = detection_results['detections']
    scores = detection_results['scores']
    times = detection_results['times']
    
    # Calculate summary statistics
    n_detections = len(detections)
    detection_rate = n_detections / data_info['duration'] if data_info['duration'] > 0 else 0
    
    if len(scores) > 0:
        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))
        score_std = float(np.std(scores))
    else:
        max_score = mean_score = score_std = 0.0
    
    # Format individual detections
    formatted_detections = []
    for i, (start_time, end_time) in enumerate(detections):
        # Find the score at this detection time
        time_idx = np.argmin(np.abs(times - start_time))
        score = float(scores[time_idx])
        
        formatted_detections.append({
            'detection_id': i + 1,
            'start_time': float(start_time),
            'end_time': float(end_time),
            'duration': float(end_time - start_time),
            'confidence_score': score,
            'snr_estimate': score * 10,  # Rough SNR estimate
        })
    
    return {
        'data_info': data_info,
        'detection_parameters': {
            'threshold': threshold,
            'overlap': float(detection_results.get('overlap', 0.5)),
            'model_type': 'unknown',  # Would need to get from detector
        },
        'summary': {
            'total_detections': n_detections,
            'detection_rate_per_hour': detection_rate * 3600,
            'data_duration_seconds': data_info['duration'],
            'max_confidence_score': max_score,
            'mean_confidence_score': mean_score,
            'confidence_score_std': score_std,
        },
        'detections': formatted_detections,
        'raw_results': {
            'scores': scores.tolist() if hasattr(scores, 'tolist') else scores,
            'times': times.tolist() if hasattr(times, 'tolist') else times,
        }
    }


def main() -> int:
    """Main detection function."""
    args = parse_arguments()
    
    try:
        # Set up output directory
        output_dir = ensure_directory(args.output_dir)
        
        # Set up logging
        log_file = output_dir / 'detection.log' if not args.quiet else None
        logger = setup_logging(
            level=args.log_level,
            log_file=log_file
        )
        
        logger.info("Starting gravitational wave detection")
        logger.info(f"Output directory: {output_dir}")
        
        # Load configuration
        config = None
        if args.config:
            config = Config.load(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        
        # Load input data
        logger.info("Loading input data...")
        start_time = time.time()
        
        data, data_info = load_input_data(args, config)
        
        load_time = time.time() - start_time
        logger.info(f"Data loaded in {format_duration(load_time)}")
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data duration: {data_info['duration']:.2f} seconds")
        logger.info(f"Data source: {data_info['source']}")
        
        # Set up detector
        logger.info("Setting up detector...")
        detector = setup_detector(args, config)
        logger.info(f"Model loaded from {args.model_path}")
        
        # Run detection
        logger.info("Running detection...")
        start_time = time.time()
        
        detection_results = run_detection(detector, data, args)
        
        detection_time = time.time() - start_time
        logger.info(f"Detection completed in {format_duration(detection_time)}")
        
        # Format results
        formatted_results = format_detection_results(
            detection_results, data_info, args.threshold
        )
        
        # Log summary
        summary = formatted_results['summary']
        logger.info(f"Detection summary:")
        logger.info(f"  Total detections: {summary['total_detections']}")
        logger.info(f"  Detection rate: {summary['detection_rate_per_hour']:.2f} per hour")
        logger.info(f"  Max confidence: {summary['max_confidence_score']:.3f}")
        logger.info(f"  Mean confidence: {summary['mean_confidence_score']:.3f}")
        
        # Save results
        if args.save_data:
            logger.info("Saving detection results...")
            
            # Save formatted results
            save_results(
                formatted_results,
                output_dir / 'detection_results.json'
            )
            
            # Save raw data if requested
            np.save(output_dir / 'input_data.npy', data)
            
            logger.info("Results saved")
        
        # Create plots
        if args.save_plots:
            logger.info("Creating detection plots...")
            
            plot_detection_results(
                strain_data=data,
                detection_results=detection_results,
                sample_rate=args.sample_rate,
                title=f"Detection Results - {data_info.get('detector', 'Unknown')}",
                save_path=output_dir / 'detection_plot.png',
                show_plot=False
            )
            
            logger.info("Detection plots saved")
        
        # Print results summary
        if not args.quiet:
            print("\n" + "="*60)
            print("GRAVITATIONAL WAVE DETECTION RESULTS")
            print("="*60)
            print(f"Data Source: {data_info['source']}")
            print(f"Duration: {data_info['duration']:.2f} seconds")
            print(f"Total Detections: {summary['total_detections']}")
            print(f"Detection Rate: {summary['detection_rate_per_hour']:.2f} per hour")
            
            if summary['total_detections'] > 0:
                print(f"Max Confidence: {summary['max_confidence_score']:.3f}")
                print(f"Mean Confidence: {summary['mean_confidence_score']:.3f}")
                print("\nDetections:")
                for det in formatted_results['detections'][:10]:  # Show first 10
                    print(f"  {det['detection_id']:2d}: "
                          f"{det['start_time']:8.2f} - {det['end_time']:8.2f} s "
                          f"(confidence: {det['confidence_score']:.3f})")
                
                if len(formatted_results['detections']) > 10:
                    print(f"  ... and {len(formatted_results['detections']) - 10} more")
            
            print("="*60)
        
        logger.info("Detection session completed successfully")
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Detection failed: {str(e)}")
        else:
            print(f"Error: {str(e)}", file=sys.stderr)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
