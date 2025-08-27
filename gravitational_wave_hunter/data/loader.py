"""
Data loading utilities for gravitational wave open data.

This module provides functions to load and access gravitational wave data
from LIGO, Virgo, and other open data sources.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

try:
    from gwpy.timeseries import TimeSeries
    from gwpy.segments import DataQualityFlag
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False
    warnings.warn(
        "gwpy not available. Some functionality will be limited. "
        "Install with: pip install gwpy"
    )

try:
    from pycbc.catalog import Catalog
    from pycbc.frame import read_frame
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False
    warnings.warn(
        "PyCBC not available. Some functionality will be limited. "
        "Install with: pip install pycbc"
    )

from ..utils.config import Config
from .preprocessing import preprocess_strain_data

# Set up logging
logger = logging.getLogger(__name__)


def load_ligo_data(
    detector: str,
    start_time: int,
    duration: int,
    sample_rate: int = 4096,
    cache_dir: Optional[Union[str, Path]] = None,
    preprocess: bool = True,
    config: Optional[Config] = None,
) -> np.ndarray:
    """
    Load LIGO strain data for a specified time period.
    
    Args:
        detector: Detector name ('H1', 'L1', 'V1')
        start_time: GPS start time
        duration: Duration in seconds
        sample_rate: Desired sample rate in Hz
        cache_dir: Directory to cache downloaded data
        preprocess: Whether to apply standard preprocessing
        config: Configuration object for preprocessing parameters
        
    Returns:
        Strain data as numpy array
        
    Raises:
        ImportError: If gwpy is not available
        ValueError: If detector name is invalid
        RuntimeError: If data loading fails
        
    Example:
        >>> data = load_ligo_data('H1', 1126259446, 4096, sample_rate=4096)
        >>> print(f"Loaded {len(data)} samples")
    """
    if not GWPY_AVAILABLE:
        raise ImportError(
            "gwpy is required for LIGO data loading. "
            "Install with: pip install gwpy"
        )
    
    valid_detectors = ['H1', 'L1', 'V1']
    if detector not in valid_detectors:
        raise ValueError(
            f"Invalid detector: {detector}. "
            f"Valid options: {valid_detectors}"
        )
    
    try:
        # Load data using gwpy
        end_time = start_time + duration
        
        logger.info(
            f"Loading {detector} data from {start_time} to {end_time} "
            f"(duration: {duration}s)"
        )
        
        # Construct channel name
        channel = f"{detector}:GDS-CALIB_STRAIN"
        
        # Load the data
        strain = TimeSeries.fetch_open_data(
            channel,
            start_time,
            end_time,
            sample_rate=sample_rate,
            cache=cache_dir is not None,
        )
        
        # Convert to numpy array
        data = strain.data
        
        # Apply preprocessing if requested
        if preprocess:
            data = preprocess_strain_data(
                data, 
                sample_rate=sample_rate, 
                config=config
            )
        
        logger.info(f"Successfully loaded {len(data)} samples from {detector}")
        return data
        
    except Exception as e:
        raise RuntimeError(f"Failed to load LIGO data: {str(e)}")


def load_virgo_data(
    start_time: int,
    duration: int,
    sample_rate: int = 4096,
    cache_dir: Optional[Union[str, Path]] = None,
    preprocess: bool = True,
    config: Optional[Config] = None,
) -> np.ndarray:
    """
    Load Virgo strain data for a specified time period.
    
    Args:
        start_time: GPS start time
        duration: Duration in seconds
        sample_rate: Desired sample rate in Hz
        cache_dir: Directory to cache downloaded data
        preprocess: Whether to apply standard preprocessing
        config: Configuration object for preprocessing parameters
        
    Returns:
        Strain data as numpy array
        
    Example:
        >>> data = load_virgo_data(1126259446, 4096, sample_rate=4096)
    """
    return load_ligo_data(
        detector='V1',
        start_time=start_time,
        duration=duration,
        sample_rate=sample_rate,
        cache_dir=cache_dir,
        preprocess=preprocess,
        config=config,
    )


def load_event_data(
    event_name: str,
    detectors: Optional[List[str]] = None,
    duration: int = 32,
    sample_rate: int = 4096,
    preprocess: bool = True,
    config: Optional[Config] = None,
) -> Dict[str, np.ndarray]:
    """
    Load gravitational wave event data from known detections.
    
    Args:
        event_name: Name of the event (e.g., 'GW150914', 'GW170817')
        detectors: List of detectors to load data from
        duration: Duration around event time in seconds
        sample_rate: Desired sample rate in Hz
        preprocess: Whether to apply standard preprocessing
        config: Configuration object
        
    Returns:
        Dictionary mapping detector names to strain data arrays
        
    Example:
        >>> event_data = load_event_data('GW150914', ['H1', 'L1'])
        >>> h1_data = event_data['H1']
        >>> l1_data = event_data['L1']
    """
    if not PYCBC_AVAILABLE:
        raise ImportError(
            "PyCBC is required for event data loading. "
            "Install with: pip install pycbc"
        )
    
    if detectors is None:
        detectors = ['H1', 'L1']
    
    try:
        # Get event information from catalog
        catalog = Catalog()
        
        if event_name not in catalog.names:
            available_events = list(catalog.names)[:10]  # Show first 10
            raise ValueError(
                f"Event {event_name} not found in catalog. "
                f"Available events include: {available_events}"
            )
        
        event_time = catalog[event_name].time
        start_time = int(event_time - duration // 2)
        
        logger.info(f"Loading event {event_name} at GPS time {event_time}")
        
        # Load data for each detector
        data_dict = {}
        for detector in detectors:
            try:
                data = load_ligo_data(
                    detector=detector,
                    start_time=start_time,
                    duration=duration,
                    sample_rate=sample_rate,
                    preprocess=preprocess,
                    config=config,
                )
                data_dict[detector] = data
                
            except Exception as e:
                logger.warning(f"Failed to load {detector} data for {event_name}: {e}")
                continue
        
        if not data_dict:
            raise RuntimeError(f"Failed to load data from any detector for {event_name}")
        
        return data_dict
        
    except Exception as e:
        raise RuntimeError(f"Failed to load event data: {str(e)}")


def download_open_data(
    detector: str,
    start_time: int,
    end_time: int,
    output_dir: Union[str, Path],
    data_type: str = "strain",
    sample_rate: int = 4096,
) -> List[Path]:
    """
    Download gravitational wave open data files.
    
    Args:
        detector: Detector name ('H1', 'L1', 'V1')
        start_time: GPS start time
        end_time: GPS end time
        output_dir: Directory to save downloaded files
        data_type: Type of data to download ('strain', 'raw')
        sample_rate: Sample rate for the data
        
    Returns:
        List of paths to downloaded files
        
    Example:
        >>> files = download_open_data('H1', 1126259446, 1126263542, './data/')
    """
    if not GWPY_AVAILABLE:
        raise ImportError("gwpy is required for data downloading")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(
        f"Downloading {detector} data from {start_time} to {end_time}"
    )
    
    # This is a simplified implementation
    # In practice, you might want to use gwpy's data finding capabilities
    downloaded_files = []
    
    # Calculate number of segments (typically 4096s segments)
    segment_length = 4096
    current_time = start_time
    
    while current_time < end_time:
        segment_end = min(current_time + segment_length, end_time)
        
        try:
            # Load segment
            data = load_ligo_data(
                detector=detector,
                start_time=current_time,
                duration=segment_end - current_time,
                sample_rate=sample_rate,
                preprocess=False,
            )
            
            # Save to file
            filename = f"{detector}_{current_time}_{segment_end}.npy"
            file_path = output_dir / filename
            np.save(file_path, data)
            downloaded_files.append(file_path)
            
            logger.info(f"Downloaded segment: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to download segment {current_time}: {e}")
        
        current_time = segment_end
    
    logger.info(f"Downloaded {len(downloaded_files)} files to {output_dir}")
    return downloaded_files


class GWDataset(Dataset):
    """
    PyTorch Dataset for gravitational wave data.
    
    This dataset handles loading, preprocessing, and augmentation of
    gravitational wave data for training machine learning models.
    
    Args:
        data_files: List of paths to data files or numpy arrays
        labels: Optional labels for supervised learning
        segment_length: Length of each training segment in seconds
        sample_rate: Sample rate of the data
        overlap: Overlap between consecutive segments
        augment: Whether to apply data augmentation
        config: Configuration object for preprocessing
        
    Example:
        >>> dataset = GWDataset(
        ...     data_files=['data1.npy', 'data2.npy'],
        ...     labels=[0, 1],  # 0=noise, 1=signal
        ...     segment_length=8.0,
        ...     sample_rate=4096
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        data_files: Union[List[Union[str, Path]], List[np.ndarray]],
        labels: Optional[List[int]] = None,
        segment_length: float = 8.0,
        sample_rate: int = 4096,
        overlap: float = 0.5,
        augment: bool = False,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize the gravitational wave dataset."""
        self.data_files = data_files
        self.labels = labels
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.augment = augment
        self.config = config or Config()
        
        # Calculate segment parameters
        self.segment_samples = int(segment_length * sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - overlap))
        
        # Create index mapping
        self._create_index_mapping()
        
    def _create_index_mapping(self) -> None:
        """Create mapping from dataset indices to file and segment indices."""
        self.index_mapping = []
        
        for file_idx, data_file in enumerate(self.data_files):
            # Load data to determine length
            if isinstance(data_file, (str, Path)):
                data = np.load(data_file)
            else:
                data = data_file
            
            # Calculate number of segments
            num_segments = max(
                1, 
                (len(data) - self.segment_samples) // self.hop_samples + 1
            )
            
            # Add mapping entries
            for seg_idx in range(num_segments):
                self.index_mapping.append((file_idx, seg_idx))
    
    def __len__(self) -> int:
        """Return the number of segments in the dataset."""
        return len(self.index_mapping)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a data segment and optional label.
        
        Args:
            idx: Index of the segment to retrieve
            
        Returns:
            Tensor containing the data segment, and optionally the label
        """
        file_idx, seg_idx = self.index_mapping[idx]
        
        # Load data
        if isinstance(self.data_files[file_idx], (str, Path)):
            data = np.load(self.data_files[file_idx])
        else:
            data = self.data_files[file_idx]
        
        # Extract segment
        start_idx = seg_idx * self.hop_samples
        end_idx = start_idx + self.segment_samples
        
        # Handle edge cases
        if end_idx > len(data):
            # Pad with zeros if necessary
            segment = np.zeros(self.segment_samples)
            available_samples = len(data) - start_idx
            segment[:available_samples] = data[start_idx:]
        else:
            segment = data[start_idx:end_idx]
        
        # Apply preprocessing
        segment = preprocess_strain_data(
            segment,
            sample_rate=self.sample_rate,
            config=self.config
        )
        
        # Apply augmentation if enabled
        if self.augment:
            segment = self._augment_segment(segment)
        
        # Convert to tensor
        segment_tensor = torch.FloatTensor(segment)
        
        # Return with or without label
        if self.labels is not None:
            label = self.labels[file_idx]
            return segment_tensor, torch.LongTensor([label])
        else:
            return segment_tensor
    
    def _augment_segment(self, segment: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to a segment.
        
        Args:
            segment: Input data segment
            
        Returns:
            Augmented segment
        """
        # Simple augmentation examples
        # In practice, you might want more sophisticated augmentation
        
        augmented = segment.copy()
        
        # Random amplitude scaling
        if np.random.random() < 0.3:
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented *= scale_factor
        
        # Random time shift
        if np.random.random() < 0.3:
            shift_samples = np.random.randint(-100, 100)
            augmented = np.roll(augmented, shift_samples)
        
        # Add small amount of noise
        if np.random.random() < 0.2:
            noise_level = np.std(augmented) * 0.01
            noise = np.random.normal(0, noise_level, len(augmented))
            augmented += noise
        
        return augmented


def create_dataloader(
    dataset: GWDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a GWDataset.
    
    Args:
        dataset: GWDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        PyTorch DataLoader
        
    Example:
        >>> dataset = GWDataset(data_files, labels)
        >>> loader = create_dataloader(dataset, batch_size=64)
        >>> for batch in loader:
        ...     # Process batch
        ...     pass
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
