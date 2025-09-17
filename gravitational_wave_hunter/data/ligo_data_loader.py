#!/usr/bin/env python3
"""
LIGO Data Loader for Gravitational Wave Detection

Downloads real gravitational wave strain data from GWOSC (Gravitational Wave Open Science Center)
for training deep learning models.

This loader uses the correct GWOSC data structure:
- Observing runs (O1, O2, O3a, O3b, O4a)
- HDF5 file format
- 4 kHz and 16 kHz sample rates
- Proper file naming conventions

References:
- GWOSC: https://gwosc.org/
- Data Access Guide: https://gwosc.org/data/
- Observing Runs: https://gwosc.org/data/
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import json

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import GWpy, fall back to gwosc if not available
try:
    from gwpy.timeseries import TimeSeries
    GWPY_AVAILABLE = True
    logger.info("GWpy available - using full functionality")
except ImportError as e:
    GWPY_AVAILABLE = False
    logger.error("GWpy not available. Install with: pip install gwpy")
    logger.error("Cannot download real LIGO data without GWpy.")

class LIGODataLoader:
    """
    Downloads and manages LIGO gravitational wave strain data from GWOSC.
    
    This class handles:
    - Downloading HDF5 files from observing runs (O1, O2, O3a, O3b, O4a)
    - Extracting strain data for specific time periods
    - Managing data for known gravitational wave events
    - Providing clean data periods for training
    """
    
    def __init__(self, cache_dir: str = "ligo_data_cache"):
        """
        Initialize the LIGO data loader.
        
        Parameters
        ----------
        cache_dir : str
            Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if GWpy is available
        if not GWPY_AVAILABLE:
            logger.warning("GWpy not available. Install with: pip install gwpy")
            logger.warning("Falling back to manual URL construction (may not work)")
        
        # GWOSC base URL for data downloads
        self.gwosc_base_url = "https://gwosc.org/data/"
        
        # Observing runs and their time ranges
        self.observing_runs = {
            'O1': {
                'time_range': (1126051217, 1137254417),  # Sep 12, 2015 - Jan 19, 2016
                'detectors': ['H1', 'L1'],
                'sample_rates': [4096],  # 4 kHz only
                'data_url': 'https://gwosc.org/data/O1/'
            },
            'O2': {
                'time_range': (1164556817, 1187733617),  # Nov 30, 2016 - Aug 25, 2017
                'detectors': ['H1', 'L1', 'V1'],
                'sample_rates': [4096, 16384],  # 4 kHz and 16 kHz
                'data_url': 'https://gwosc.org/data/O2/'
            },
            'O3a': {
                'time_range': (1238166017, 1253977217),  # Apr 1, 2019 - Oct 1, 2019
                'detectors': ['H1', 'L1', 'V1'],
                'sample_rates': [4096, 16384],
                'data_url': 'https://gwosc.org/data/O3a/'
            },
            'O3b': {
                'time_range': (1253977217, 1269363617),  # Nov 1, 2019 - Mar 27, 2020
                'detectors': ['H1', 'L1', 'V1'],
                'sample_rates': [4096, 16384],
                'data_url': 'https://gwosc.org/data/O3b/'
            },
            'O4a': {
                'time_range': (1684896017, 1705449617),  # May 24, 2023 - Jan 16, 2024
                'detectors': ['H1', 'L1'],
                'sample_rates': [4096, 16384],
                'data_url': 'https://gwosc.org/data/O4a/'
            }
        }
        
        # Known gravitational wave events mapped to observing runs
        self.known_events = {
            'GW150914': {'time': 1126259446, 'run': 'O1', 'detectors': ['H1', 'L1']},
            'GW151226': {'time': 1135136334, 'run': 'O1', 'detectors': ['H1', 'L1']},
            'GW170104': {'time': 1167559920, 'run': 'O2', 'detectors': ['H1', 'L1']},
            'GW170608': {'time': 1180922478, 'run': 'O2', 'detectors': ['H1', 'L1']},
            'GW170814': {'time': 1186741845, 'run': 'O2', 'detectors': ['H1', 'L1', 'V1']},
            'GW170817': {'time': 1187008866, 'run': 'O2', 'detectors': ['H1', 'L1', 'V1']},
        }
        
        # GWpy channel names for different observing runs and sample rates
        self.channel_names = {
            'O1': {
                'H1': 'H1:GWOSC-4KHZ_R1_STRAIN',
                'L1': 'L1:GWOSC-4KHZ_R1_STRAIN'
            },
            'O2': {
                'H1': 'H1:GWOSC-4KHZ_R1_STRAIN',
                'L1': 'L1:GWOSC-4KHZ_R1_STRAIN',
                'V1': 'V1:GWOSC-4KHZ_R1_STRAIN'
            },
            'O3a': {
                'H1': 'H1:GWOSC-4KHZ_R1_STRAIN',
                'L1': 'L1:GWOSC-4KHZ_R1_STRAIN',
                'V1': 'V1:GWOSC-4KHZ_R1_STRAIN'
            },
            'O3b': {
                'H1': 'H1:GWOSC-4KHZ_R1_STRAIN',
                'L1': 'L1:GWOSC-4KHZ_R1_STRAIN',
                'V1': 'V1:GWOSC-4KHZ_R1_STRAIN'
            },
            'O4a': {
                'H1': 'H1:GWOSC-4KHZ_R1_STRAIN',
                'L1': 'L1:GWOSC-4KHZ_R1_STRAIN'
            }
        }
        
        logger.info(f"LIGO Data Loader initialized with cache: {cache_dir}")
        logger.info(f"Available observing runs: {list(self.observing_runs.keys())}")
    
    def test_connection(self) -> bool:
        """
        Test connection to GWOSC data portal.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        if GWPY_AVAILABLE:
            try:
                logger.info("Testing GWOSC connection via GWpy...")
                # Try to download a small amount of data to test connection
                test_data = TimeSeries.get('H1:GWOSC-4KHZ_R1_STRAIN', 1126259446, 1126259447, verbose=False)
                logger.info("GWOSC connection successful via GWpy")
                return True
                
            except Exception as e:
                logger.error(f"Failed to connect to GWOSC via GWpy: {e}")
                return False
        
        else:
            logger.error("GWpy not available. Cannot test connection.")
            return False
    
    def get_available_runs(self) -> List[str]:
        """
        Get list of available observing runs.
        
        Returns
        -------
        List[str]
            List of available observing runs
        """
        logger.info("Available observing runs:")
        for run, info in self.observing_runs.items():
            start_date = datetime.fromtimestamp(info['time_range'][0]).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(info['time_range'][1]).strftime('%Y-%m-%d')
            detectors = ', '.join(info['detectors'])
            logger.info(f"   {run}: {start_date} to {end_date} ({detectors})")
        
        return list(self.observing_runs.keys())
    
    def download_strain_data(self, detector: str, start_time: int, duration: int = 32, sample_rate: int = 4096) -> Optional[Dict]:
        """
        Download strain data for a specific detector and time period.
        
        Parameters
        ----------
        detector : str
            Detector name (H1, L1, V1)
        start_time : int
            GPS start time
        duration : int
            Duration in seconds (default: 32)
        sample_rate : int
            Sample rate in Hz (4096 or 16384, default: 4096)
            
        Returns
        -------
        Optional[Dict]
            Dictionary containing strain data and metadata, or None if failed
        """
        end_time = start_time + duration
        
        # Find which observing run contains this time
        run_info = self._find_observing_run(start_time)
        if not run_info:
            logger.error(f"Time {start_time} not in any observing run")
            return None
        
        run_name = run_info['run']
        logger.info(f"Time {start_time} is in {run_name} observing run")
        
        # Check if detector is available in this run
        if detector not in run_info['detectors']:
            logger.error(f"Detector {detector} not available in {run_name}")
            return None
        
        # Check if sample rate is available
        if sample_rate not in run_info['sample_rates']:
            logger.error(f"Sample rate {sample_rate} not available in {run_name}")
            return None
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{run_name}_{detector}_{start_time}_{duration}_{sample_rate}.npz")
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data: {cache_file}")
            try:
                data = np.load(cache_file)
                return {
                    'strain': data['strain'],
                    'times': data['times'],
                    'sample_rate': int(data['sample_rate']),
                    'detector': detector,
                    'start_time': start_time,
                    'end_time': end_time,
                    'run': run_name,
                    'cached': True
                }
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Download using available method
        if GWPY_AVAILABLE:
            return self._download_with_gwpy(detector, start_time, end_time, run_name, cache_file)
        else:
            logger.error("GWpy not available. Cannot download real LIGO data.")
            logger.error("Install GWpy with: pip install gwpy")
            return None
    
    def _download_with_gwpy(self, detector: str, start_time: int, end_time: int, run_name: str, cache_file: str) -> Optional[Dict]:
        """
        Download strain data using GWpy fetch_open_data.
        
        Parameters
        ----------
        detector : str
            Detector name (H1, L1, V1)
        start_time : int
            GPS start time
        end_time : int
            GPS end time
        run_name : str
            Observing run name
        cache_file : str
            Path to cache file
            
        Returns
        -------
        Optional[Dict]
            Dictionary containing strain data and metadata, or None if failed
        """
        try:
            logger.info(f"Downloading {detector} data using GWpy fetch_open_data...")
            
            # Use fetch_open_data for public LIGO data
            strain = TimeSeries.fetch_open_data(detector, start_time, end_time)
            
            # Extract data
            strain_data = strain.value
            times_data = strain.times.value
            
            logger.info(f"Downloaded {len(strain_data)} samples")
            logger.info(f"   Sample rate: {strain.sample_rate.value} Hz")
            logger.info(f"   Duration: {len(strain_data) / strain.sample_rate.value:.2f} seconds")
            
            # Cache the data
            np.savez(cache_file, 
                   strain=strain_data,
                   times=times_data,
                   sample_rate=strain.sample_rate.value)
            logger.info(f"Data cached: {cache_file}")
            
            return {
                'strain': strain_data,
                'times': times_data,
                'sample_rate': strain.sample_rate.value,
                'detector': detector,
                'start_time': start_time,
                'end_time': end_time,
                'run': run_name,
                'cached': False,
                'source': 'GWOSC open data'
            }
            
        except Exception as e:
            logger.error(f"Error downloading with GWpy: {e}")
            return None
    
    
    def _find_observing_run(self, gps_time: int) -> Optional[Dict]:
        """
        Find which observing run contains the given GPS time.
        
        Parameters
        ----------
        gps_time : int
            GPS time to find
            
        Returns
        -------
        Optional[Dict]
            Observing run info or None if not found
        """
        for run_name, run_info in self.observing_runs.items():
            start_time, end_time = run_info['time_range']
            if start_time <= gps_time <= end_time:
                return {'run': run_name, **run_info}
        return None
    
    
    def get_event_data(self, event_name: str, duration: int = 32, sample_rate: int = 4096) -> Dict[str, Optional[Dict]]:
        """
        Download data for a known gravitational wave event.
        
        Parameters
        ----------
        event_name : str
            Name of the event (e.g., 'GW150914')
        duration : int
            Duration in seconds around the event
        sample_rate : int
            Sample rate in Hz (4096 or 16384, default: 4096)
            
        Returns
        -------
        Dict[str, Optional[Dict]]
            Dictionary mapping detector names to their data
        """
        if event_name not in self.known_events:
            logger.error(f"Unknown event: {event_name}")
            return {}
        
        event_info = self.known_events[event_name]
        event_time = event_info['time']
        detectors = event_info['detectors']
        run_name = event_info['run']
        
        logger.info(f"Downloading data for {event_name} at GPS time {event_time} from {run_name}")
        
        event_data = {}
        for detector in detectors:
            logger.info(f"   Downloading {detector} data...")
            data = self.download_strain_data(detector, event_time, duration, sample_rate)
            event_data[detector] = data
            
            if data:
                logger.info(f"   {detector}: {len(data['strain'])} samples")
            else:
                logger.error(f"   {detector}: Failed to download")
        
        return event_data
    
    def get_clean_data_periods(self, num_periods: int = 5, duration: int = 32, sample_rate: int = 4096) -> List[Dict]:
        """
        Download clean data periods (no gravitational waves) for training.
        
        Parameters
        ----------
        num_periods : int
            Number of clean periods to download
        duration : int
            Duration of each period in seconds
        sample_rate : int
            Sample rate in Hz (4096 or 16384, default: 4096)
            
        Returns
        -------
        List[Dict]
            List of clean data periods
        """
        logger.info(f"Downloading {num_periods} clean data periods...")
        
        # Use different time periods from different observing runs
        clean_periods = [
            {'start': 1126052000, 'run': 'O1'},  # Early O1
            {'start': 1127000000, 'run': 'O1'},  # Mid O1
            {'start': 1165000000, 'run': 'O2'},  # Early O2
            {'start': 1170000000, 'run': 'O2'},  # Mid O2
            {'start': 1239000000, 'run': 'O3a'}, # Early O3a
        ]
        
        all_clean_data = []
        for i, period in enumerate(clean_periods[:num_periods]):
            logger.info(f"   Period {i+1}: {period['start']} from {period['run']}")
            
            # Get detectors for this run
            run_info = self.observing_runs[period['run']]
            detectors = run_info['detectors']
            
            period_data = {}
            for detector in detectors:
                data = self.download_strain_data(detector, period['start'], duration, sample_rate)
                period_data[detector] = data
                
                if data:
                    logger.info(f"   {detector}: {len(data['strain'])} samples")
                else:
                    logger.error(f"   {detector}: Failed to download")
            
            all_clean_data.append(period_data)
        
        return all_clean_data


def main():
    """Test the LIGO data loader"""
    logger.info("Testing LIGO Data Loader...")
    
    # Initialize loader
    loader = LIGODataLoader()
    
    # Test connection
    if not loader.test_connection():
        logger.error("Cannot connect to GWOSC. Exiting.")
        return
    
    # Show available observing runs
    runs = loader.get_available_runs()
    logger.info(f"Found {len(runs)} observing runs")
    
    # Test downloading a small amount of data from GW150914
    logger.info("Testing data download for GW150914...")
    test_data = loader.download_strain_data('H1', 1126259446, 4, 4096)  # 4 seconds around GW150914
    
    if test_data:
        logger.info("Data download successful!")
        logger.info(f"   Samples: {len(test_data['strain'])}")
        logger.info(f"   Sample rate: {test_data['sample_rate']} Hz")
        logger.info(f"   Duration: {len(test_data['strain']) / test_data['sample_rate']:.2f} seconds")
        logger.info(f"   Observing run: {test_data['run']}")
    else:
        logger.error("Data download failed")
    
    # Test downloading event data
    logger.info("Testing event data download...")
    event_data = loader.get_event_data('GW150914', duration=8, sample_rate=4096)
    
    if event_data:
        logger.info("Event data download successful!")
        for detector, data in event_data.items():
            if data:
                logger.info(f"   {detector}: {len(data['strain'])} samples")
            else:
                logger.error(f"   {detector}: Failed to download")
    else:
        logger.error("Event data download failed")
    
    logger.info("Test completed")


if __name__ == "__main__":
    main()
