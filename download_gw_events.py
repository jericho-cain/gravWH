#!/usr/bin/env python3
"""
GW Events Data Downloader

This script downloads gravitational wave events from GWOSC and saves them locally.
Run this once to populate the cache, then use the training pipeline.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add the project root to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gravitational_wave_hunter', 'data'))
from ligo_data_loader import LIGODataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_gw_events():
    """Download gravitational wave events and save them locally."""
    
    logger.info("🌌 Starting GW events download...")
    
    # Initialize data loader
    data_loader = LIGODataLoader()
    
    # Create cache directory for GW events
    gw_cache_dir = "gw_events_cache"
    os.makedirs(gw_cache_dir, exist_ok=True)
    
    # Known gravitational wave events - ONLY the most reliable ones that we know work
    # Starting with just a few to test the pipeline
    gw_events = [
        # O1 events (2015-2017) - All confirmed and available
        'GW150914', 'GW151226', 'GW151012',
        
        # O2 events (2016-2017) - All confirmed and available
        'GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170817', 'GW170818', 'GW170823',
        
        # O3a events (2019-2020) - Only the most reliable ones
        'GW190408_181802', 'GW190412', 'GW190413_052954', 'GW190413_134308', 'GW190421_213856',
        'GW190426_152155', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428', 'GW190514_065416',
        'GW190517_055101', 'GW190519_153544', 'GW190521', 'GW190527_092243', 'GW190602_175927',
        'GW190630_185205', 'GW190701_203306', 'GW190706_222641', 'GW190707_093326',
        'GW190708_232457', 'GW190719_215514', 'GW190720_000836', 'GW190727_060333', 'GW190728_064510',
        'GW190731_140936', 'GW190803_022701', 'GW190805_211137', 'GW190828_063405', 'GW190828_065509',
        'GW190910_112807', 'GW190915_235702', 'GW190924_021846', 'GW190925_232845', 'GW190926_050336',
        'GW190930_133541', 'GW191103_012549', 'GW191105_143521', 'GW191109_010717', 'GW191113_071529',
        'GW191126_115259', 'GW191127_050227', 'GW191129_012715', 'GW191204_171526', 'GW191215_223052',
        'GW191216_213338', 'GW191222_033537', 'GW191230_180458', 'GW200112_155838', 'GW200128_022011',
        'GW200129_065458', 'GW200202_154313', 'GW200208_130117', 'GW200208_222617', 'GW200209_085452',
        'GW200216_220804', 'GW200219_094415', 'GW200220_124850', 'GW200224_222234', 'GW200225_060421',
        'GW200302_015811', 'GW200311_115853', 'GW200316_215756', 'GW200322_091133'
    ]
    
    # Limit to first 20 events for testing
    gw_events = gw_events[:20]
    
    successful_downloads = []
    failed_downloads = []
    
    logger.info(f"📊 Attempting to download {len(gw_events)} GW events...")
    
    for i, event_name in enumerate(gw_events):
        try:
            logger.info(f"📥 Downloading {event_name} ({i+1}/{len(gw_events)})...")
            
            # Download event data with timeout
            event_data = data_loader.get_event_data(event_name, duration=4)
            
            if event_data and 'H1' in event_data:
                # Save to cache
                cache_file = os.path.join(gw_cache_dir, f"{event_name}_H1.npz")
                np.savez(cache_file, 
                        strain=event_data['H1']['strain'],
                        event_name=event_name,
                        duration=4,
                        sample_rate=4096)
                
                successful_downloads.append(event_name)
                logger.info(f"✅ {event_name} downloaded successfully")
            else:
                failed_downloads.append(event_name)
                logger.warning(f"❌ {event_name} - No H1 data available")
                
        except KeyboardInterrupt:
            logger.info("🛑 Download interrupted by user")
            break
        except Exception as e:
            failed_downloads.append(event_name)
            logger.warning(f"❌ {event_name} - Download failed: {e}")
            
            # If we get too many failures in a row, stop
            if len(failed_downloads) >= 5 and len(successful_downloads) == 0:
                logger.error("❌ Too many consecutive failures, stopping download")
                break
    
    # Summary
    logger.info(f"\n📈 Download Summary:")
    logger.info(f"   ✅ Successful: {len(successful_downloads)}")
    logger.info(f"   ❌ Failed: {len(failed_downloads)}")
    logger.info(f"   📁 Cache directory: {gw_cache_dir}")
    
    if successful_downloads:
        logger.info(f"\n🎉 Successfully downloaded events:")
        for event in successful_downloads[:10]:  # Show first 10
            logger.info(f"   - {event}")
        if len(successful_downloads) > 10:
            logger.info(f"   ... and {len(successful_downloads) - 10} more")
    
    if failed_downloads:
        logger.info(f"\n⚠️  Failed downloads:")
        for event in failed_downloads[:5]:  # Show first 5
            logger.info(f"   - {event}")
        if len(failed_downloads) > 5:
            logger.info(f"   ... and {len(failed_downloads) - 5} more")
    
    return successful_downloads, failed_downloads

if __name__ == "__main__":
    download_gw_events()
