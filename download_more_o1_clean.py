#!/usr/bin/env python3
"""
Download more O1 clean data (guaranteed clean noise)
"""

import os
import numpy as np
import logging
from gwpy.timeseries import TimeSeries

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_o1_clean_data(num_samples=500):
    """Download more O1 clean data."""
    
    # O1 observing run: Sep 12, 2015 - Jan 19, 2016
    o1_start = 1126051217
    o1_end = 1137254417
    
    # Known GW events in O1 (avoid these times)
    o1_gw_events = [
        1126259446,  # GW150914
        1135136334,  # GW151226  
        1128678900,  # GW151012
    ]
    
    # Create buffer around GW events (avoid ±1 hour)
    buffer = 3600  # 1 hour in seconds
    avoid_times = []
    for event_time in o1_gw_events:
        avoid_times.extend(range(event_time - buffer, event_time + buffer))
    
    downloaded_samples = 0
    skipped_samples = 0
    
    for i in range(num_samples):
        try:
            # Generate random GPS time in O1
            random_gps = np.random.randint(o1_start, o1_end - 4)
            
            # Skip if too close to known GW events
            if random_gps in avoid_times:
                skipped_samples += 1
                continue
            
            # Check if file already exists
            cache_filename = f"ligo_data_cache/O1_H1_{random_gps}_4_4096.npz"
            if os.path.exists(cache_filename):
                skipped_samples += 1
                continue
            
            logger.info(f"Downloading O1 clean sample {i+1}/{num_samples} at GPS {random_gps}")
            
            # Download O1 data (should be clean)
            data = TimeSeries.fetch_open_data('H1', random_gps, random_gps + 4, cache=True)
            
            # Save to cache
            np.savez(cache_filename, strain=data.value, times=random_gps, sample_rate=4096)
            
            downloaded_samples += 1
            logger.info(f"Successfully downloaded O1 clean sample {i+1}")
            
            if downloaded_samples % 50 == 0:
                logger.info(f"Downloaded {downloaded_samples} new O1 clean samples")
                
        except Exception as e:
            logger.warning(f"Failed to download sample {i+1}: {e}")
            continue
    
    logger.info(f"Download complete! Downloaded {downloaded_samples} new O1 clean samples, skipped {skipped_samples}")
    return downloaded_samples

if __name__ == "__main__":
    print("Downloading 500 more O1 clean samples...")
    download_o1_clean_data(500)
    print("O1 clean data download complete!")
