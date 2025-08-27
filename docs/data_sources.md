# Data Sources and Formats

## Table of Contents

1. [Introduction](#introduction)
2. [LIGO Open Science Center](#ligo-open-science-center)
3. [Virgo Open Data](#virgo-open-data)
4. [Data Formats](#data-formats)
5. [Data Quality](#data-quality)
6. [Preprocessing Pipeline](#preprocessing-pipeline)
7. [Simulated Data](#simulated-data)
8. [Training Datasets](#training-datasets)
9. [Data Access Examples](#data-access-examples)
10. [Best Practices](#best-practices)
11. [References](#references)

## Introduction

This document provides comprehensive information about gravitational wave data sources, formats, and preprocessing procedures used in the Gravitational Wave Hunter framework. The data primarily comes from the LIGO and Virgo interferometric detectors, which are made freely available through open science initiatives.

### Data Overview

Gravitational wave detectors produce continuous streams of strain data that capture tiny distortions in spacetime. This data contains:

- **Gravitational wave signals**: Extremely weak signals from cosmic events
- **Detector noise**: Various noise sources that mask the signals
- **Environmental disturbances**: Seismic, thermal, and electromagnetic interference
- **Data quality information**: Flags indicating data reliability

## LIGO Open Science Center

The LIGO Open Science Center (LOSC) provides free access to gravitational wave data and related materials for research and education.

### Available Data

#### Observation Runs

- **O1 (September 2015 - January 2016)**: First observing run
- **O2 (November 2016 - August 2017)**: Second observing run
- **O3a (April 2019 - October 2019)**: Third observing run, first half
- **O3b (November 2019 - March 2020)**: Third observing run, second half

#### Data Types

1. **Strain Data**: Calibrated gravitational wave strain
2. **Raw Data**: Uncalibrated detector output
3. **Auxiliary Channels**: Environmental and instrumental monitoring
4. **Data Quality Flags**: Information about data reliability

#### Detectors

- **LIGO Hanford (H1)**: 4 km arms, Washington State, USA
- **LIGO Livingston (L1)**: 4 km arms, Louisiana, USA
- **Virgo (V1)**: 3 km arms, Cascina, Italy (from O2 onwards)

### Data Access Methods

#### 1. Direct Download

Download specific data files from the LIGO Open Science Center website:

```
https://www.gw-openscience.org/data/
```

#### 2. GWpy Library

Use the GWpy library for programmatic access:

```python
from gwpy.timeseries import TimeSeries

# Load strain data
strain = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
```

#### 3. REST API

Access data through RESTful web services:

```python
import requests

url = "https://www.gw-openscience.org/archive/data/S6/967824014/H-H1_LOSC_4_V1-967824014-4096.txt"
response = requests.get(url)
```

## Virgo Open Data

Virgo data is available through the European Gravitational Observatory (EGO) and integrated with LIGO data releases.

### Virgo Specifics

- **Location**: Cascina, near Pisa, Italy
- **Arm Length**: 3 km
- **Sensitivity**: Optimized for 10 Hz - 10 kHz frequency range
- **Unique Features**: Different suspension system and environment

### Data Characteristics

- **Sample Rate**: Typically 16384 Hz (downsampled to 4096 Hz for analysis)
- **Data Format**: Same as LIGO (HDF5, frame files)
- **Coordinate System**: Different orientation affects signal reconstruction

## Data Formats

### HDF5 Format

The primary format for strain data storage:

```python
import h5py

def read_hdf5_strain(filename):
    """
    Read strain data from HDF5 file.
    
    Args:
        filename: Path to HDF5 file
        
    Returns:
        strain: Time series data
        meta: Metadata dictionary
    """
    with h5py.File(filename, 'r') as f:
        strain = f['strain']['Strain'][:]
        ts = f['strain']['Strain'].attrs['Xspacing']  # Time spacing
        t0 = f['strain']['Strain'].attrs['Xstart']    # Start time
        
        meta = {
            'dt': ts,
            't0': t0,
            'duration': len(strain) * ts,
            'sample_rate': 1.0 / ts,
        }
        
    return strain, meta
```

### Frame Files

Frame files contain time-series data and metadata:

```python
from pycbc.frame import read_frame

def read_frame_data(filename, channel, start_time, duration):
    """
    Read data from frame file.
    
    Args:
        filename: Path to frame file
        channel: Channel name (e.g., 'H1:GDS-CALIB_STRAIN')
        start_time: GPS start time
        duration: Duration in seconds
        
    Returns:
        TimeSeries object
    """
    strain = read_frame(filename, channel, start_time, start_time + duration)
    return strain
```

### Text Files

Simple ASCII format for educational purposes:

```python
import numpy as np

def read_txt_strain(filename):
    """
    Read strain data from text file.
    
    Args:
        filename: Path to text file
        
    Returns:
        time: Time array
        strain: Strain values
    """
    data = np.loadtxt(filename)
    time = data[:, 0]
    strain = data[:, 1]
    return time, strain
```

### JSON Metadata

Event parameters and detection information:

```python
import json

def read_event_metadata(filename):
    """
    Read event metadata from JSON file.
    
    Args:
        filename: Path to JSON file
        
    Returns:
        Dictionary with event parameters
    """
    with open(filename, 'r') as f:
        metadata = json.load(f)
    
    return metadata
```

## Data Quality

### Data Quality Flags

Data quality flags indicate periods of poor data quality:

#### Flag Categories

1. **CAT1**: Most severe issues (data unusable)
2. **CAT2**: Moderate issues (caution advised)
3. **CAT3**: Minor issues (data generally usable)

#### Common Flags

- **NO_OMC_DCPD_ADC_OVERFLOW**: ADC overflow in output mode cleaner
- **NO_STOCH_HW_INJ**: No stochastic hardware injections
- **NO_CBC_HW_INJ**: No compact binary coalescence hardware injections

#### Using Data Quality Flags

```python
from gwpy.segments import DataQualityFlag

def get_good_data_segments(detector, start_time, end_time):
    """
    Get segments of good quality data.
    
    Args:
        detector: Detector name ('H1', 'L1', 'V1')
        start_time: GPS start time
        end_time: GPS end time
        
    Returns:
        SegmentList of good data periods
    """
    # Define data quality flags
    flags = [
        f'{detector}:DMT-ANALYSIS_READY:1',
        f'{detector}:DCH-CLEAN_STRAIN_C02:1',
    ]
    
    good_segments = DataQualityFlag.query_dqsegdb(
        flags, start_time, end_time
    ).active
    
    return good_segments
```

### Noise Characterization

Understanding detector noise is crucial for signal detection:

#### Noise Sources

1. **Seismic Noise**: Ground vibrations at low frequencies (< 40 Hz)
2. **Thermal Noise**: Brownian motion in mirrors and suspensions
3. **Shot Noise**: Quantum noise from photon counting
4. **Technical Noise**: Electronics, laser instabilities

#### Power Spectral Density

```python
from gwpy.frequencyseries import FrequencySeries

def estimate_psd(strain, sample_rate, fftlength=4):
    """
    Estimate power spectral density of strain data.
    
    Args:
        strain: Time series strain data
        sample_rate: Sample rate in Hz
        fftlength: FFT length in seconds
        
    Returns:
        FrequencySeries with PSD
    """
    from gwpy.timeseries import TimeSeries
    
    # Create TimeSeries object
    ts = TimeSeries(strain, sample_rate=sample_rate)
    
    # Compute PSD using Welch's method
    psd = ts.psd(fftlength=fftlength, overlap=fftlength/2)
    
    return psd
```

## Preprocessing Pipeline

### Standard Preprocessing Steps

1. **Bandpass Filtering**: Remove frequencies outside sensitive band
2. **Whitening**: Flatten the noise spectrum
3. **Glitch Removal**: Remove transient artifacts
4. **Normalization**: Standardize amplitude distribution

### Implementation

```python
from gravitational_wave_hunter.signal_processing import preprocess_strain_data
from gravitational_wave_hunter.utils.config import Config

def preprocess_data(strain, sample_rate=4096):
    """
    Apply standard preprocessing to strain data.
    
    Args:
        strain: Raw strain data
        sample_rate: Sample rate in Hz
        
    Returns:
        Preprocessed strain data
    """
    # Create configuration
    config = Config()
    config.bandpass_low = 20.0      # Hz
    config.bandpass_high = 2000.0   # Hz
    config.apply_whitening = True
    config.remove_glitches = True
    config.normalize = True
    
    # Apply preprocessing
    processed_strain = preprocess_strain_data(
        strain, 
        sample_rate=sample_rate, 
        config=config
    )
    
    return processed_strain
```

### Advanced Preprocessing

#### Gating

Remove loud glitches by zeroing data segments:

```python
def gate_data(strain, times, sample_rate, gate_width=0.25):
    """
    Gate loud glitches in strain data.
    
    Args:
        strain: Input strain data
        times: Times of glitches to gate
        sample_rate: Sample rate in Hz
        gate_width: Width of gate in seconds
        
    Returns:
        Gated strain data
    """
    gated_strain = strain.copy()
    gate_samples = int(gate_width * sample_rate)
    
    for gate_time in times:
        # Convert time to sample index
        gate_idx = int(gate_time * sample_rate)
        
        # Apply gate
        start_idx = max(0, gate_idx - gate_samples // 2)
        end_idx = min(len(strain), gate_idx + gate_samples // 2)
        
        # Apply Tukey window for smooth gating
        gate_length = end_idx - start_idx
        window = np.ones(gate_length)
        taper_length = gate_length // 4
        
        # Taper edges
        window[:taper_length] = np.sin(np.pi/2 * np.linspace(0, 1, taper_length))**2
        window[-taper_length:] = np.cos(np.pi/2 * np.linspace(0, 1, taper_length))**2
        
        gated_strain[start_idx:end_idx] *= (1 - window)
    
    return gated_strain
```

#### Spectral Subtraction

Remove line noise artifacts:

```python
def remove_line_noise(strain, sample_rate, line_frequencies):
    """
    Remove spectral lines from strain data.
    
    Args:
        strain: Input strain data
        sample_rate: Sample rate in Hz
        line_frequencies: List of line frequencies to remove
        
    Returns:
        Strain data with lines removed
    """
    # Transform to frequency domain
    strain_fft = np.fft.rfft(strain)
    freqs = np.fft.rfftfreq(len(strain), 1/sample_rate)
    
    # Remove lines
    for line_freq in line_frequencies:
        # Find frequency bin
        freq_idx = np.argmin(np.abs(freqs - line_freq))
        
        # Notch filter around line
        notch_width = 5  # Hz
        width_bins = int(notch_width / (freqs[1] - freqs[0]))
        
        start_bin = max(0, freq_idx - width_bins)
        end_bin = min(len(freqs), freq_idx + width_bins)
        
        # Zero out the line
        strain_fft[start_bin:end_bin] = 0
    
    # Transform back to time domain
    cleaned_strain = np.fft.irfft(strain_fft, n=len(strain))
    
    return cleaned_strain
```

## Simulated Data

### Waveform Generation

Generate synthetic gravitational wave signals:

```python
from pycbc.waveform import get_td_waveform

def generate_bbh_waveform(mass1, mass2, distance, sample_rate=4096, duration=4):
    """
    Generate binary black hole waveform.
    
    Args:
        mass1: Primary mass in solar masses
        mass2: Secondary mass in solar masses
        distance: Distance in Mpc
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        
    Returns:
        hp, hc: Plus and cross polarizations
    """
    # Generate waveform
    hp, hc = get_td_waveform(
        approximant='IMRPhenomPv2',
        mass1=mass1,
        mass2=mass2,
        distance=distance,
        delta_t=1.0/sample_rate,
        f_lower=20.0
    )
    
    # Resize to desired duration
    target_length = int(duration * sample_rate)
    
    if len(hp) < target_length:
        # Pad with zeros
        hp.resize(target_length)
        hc.resize(target_length)
    else:
        # Trim to duration
        hp = hp[-target_length:]
        hc = hc[-target_length:]
    
    return hp.data, hc.data
```

### Noise Generation

Generate realistic detector noise:

```python
def generate_colored_noise(length, sample_rate, psd):
    """
    Generate colored noise with specified PSD.
    
    Args:
        length: Length of noise in samples
        sample_rate: Sample rate in Hz
        psd: Power spectral density
        
    Returns:
        Colored noise time series
    """
    # Generate white noise
    white_noise = np.random.normal(0, 1, length)
    
    # Transform to frequency domain
    white_fft = np.fft.rfft(white_noise)
    freqs = np.fft.rfftfreq(length, 1/sample_rate)
    
    # Color the noise
    psd_interp = np.interp(freqs, psd.frequencies, psd.value)
    colored_fft = white_fft * np.sqrt(psd_interp * sample_rate / 2)
    
    # Transform back to time domain
    colored_noise = np.fft.irfft(colored_fft, n=length)
    
    return colored_noise
```

## Training Datasets

### Dataset Structure

Organize training data efficiently:

```
training_data/
├── noise/
│   ├── H1_noise_001.npy
│   ├── H1_noise_002.npy
│   └── ...
├── signals/
│   ├── BBH_001.npy
│   ├── BBH_002.npy
│   └── ...
├── mixed/
│   ├── signal_in_noise_001.npy
│   ├── signal_in_noise_002.npy
│   └── ...
└── labels.json
```

### Label Format

```json
{
    "H1_noise_001": 0,
    "H1_noise_002": 0,
    "BBH_001": 1,
    "BBH_002": 1,
    "signal_in_noise_001": 1
}
```

### Creating Training Sets

```python
def create_training_dataset(noise_files, signal_files, output_dir, 
                          signal_probability=0.5, snr_range=(8, 25)):
    """
    Create balanced training dataset.
    
    Args:
        noise_files: List of noise-only files
        signal_files: List of signal templates
        output_dir: Output directory
        signal_probability: Fraction of samples with signals
        snr_range: Range of signal-to-noise ratios
        
    Returns:
        Dictionary with file paths and labels
    """
    import random
    import os
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = {}
    file_counter = 0
    
    for noise_file in noise_files:
        # Load noise
        noise = np.load(noise_file)
        
        # Decide whether to inject signal
        if random.random() < signal_probability:
            # Inject signal
            signal_file = random.choice(signal_files)
            signal = np.load(signal_file)
            
            # Random SNR
            snr = random.uniform(*snr_range)
            
            # Scale signal to desired SNR
            signal_energy = np.sqrt(np.sum(signal**2))
            noise_energy = np.sqrt(np.sum(noise**2))
            scale_factor = (snr * noise_energy) / signal_energy
            
            # Add signal to noise
            combined = noise + scale_factor * signal
            
            # Save
            output_file = output_dir / f"data_{file_counter:06d}.npy"
            np.save(output_file, combined)
            labels[f"data_{file_counter:06d}"] = 1
            
        else:
            # Noise only
            output_file = output_dir / f"data_{file_counter:06d}.npy"
            np.save(output_file, noise)
            labels[f"data_{file_counter:06d}"] = 0
        
        file_counter += 1
    
    # Save labels
    with open(output_dir / "labels.json", 'w') as f:
        json.dump(labels, f, indent=2)
    
    return labels
```

## Data Access Examples

### Loading LIGO Event Data

```python
from gravitational_wave_hunter.data.loader import load_event_data

# Load GW150914 data
event_data = load_event_data(
    event_name='GW150914',
    detectors=['H1', 'L1'],
    duration=32,
    sample_rate=4096,
    preprocess=True
)

h1_strain = event_data['H1']
l1_strain = event_data['L1']
```

### Custom Data Loading

```python
from gravitational_wave_hunter.data.loader import GWDataset, create_dataloader

# Create dataset
dataset = GWDataset(
    data_files=['data1.npy', 'data2.npy'],
    labels=[0, 1],
    segment_length=8.0,
    sample_rate=4096,
    overlap=0.5,
    augment=True
)

# Create dataloader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Use in training loop
for batch_data, batch_labels in dataloader:
    # Training code here
    pass
```

### Streaming Data Access

```python
def stream_ligo_data(detector, start_time, duration, chunk_size=4096):
    """
    Stream LIGO data in chunks.
    
    Args:
        detector: Detector name
        start_time: GPS start time
        duration: Total duration
        chunk_size: Chunk size in samples
        
    Yields:
        Data chunks
    """
    current_time = start_time
    end_time = start_time + duration
    chunk_duration = chunk_size / 4096  # Assuming 4096 Hz
    
    while current_time < end_time:
        # Load chunk
        try:
            chunk = load_ligo_data(
                detector=detector,
                start_time=current_time,
                duration=chunk_duration,
                sample_rate=4096
            )
            yield chunk, current_time
            
        except Exception as e:
            print(f"Failed to load chunk at {current_time}: {e}")
            
        current_time += chunk_duration
```

## Best Practices

### Data Management

1. **Version Control**: Track data versions and preprocessing parameters
2. **Metadata**: Store comprehensive metadata with each dataset
3. **Backup**: Maintain backup copies of important datasets
4. **Documentation**: Document data sources and processing steps

### Performance Optimization

1. **Caching**: Cache frequently accessed data
2. **Parallel Loading**: Use multiple workers for data loading
3. **Memory Management**: Monitor memory usage for large datasets
4. **Storage Format**: Choose efficient storage formats (HDF5 vs NumPy)

### Quality Assurance

1. **Validation**: Validate data integrity and format
2. **Sanity Checks**: Perform basic statistical checks
3. **Visualization**: Plot data to identify anomalies
4. **Testing**: Test preprocessing pipeline thoroughly

### Example Quality Check

```python
def validate_strain_data(strain, sample_rate, expected_duration=None):
    """
    Validate strain data quality.
    
    Args:
        strain: Strain time series
        sample_rate: Sample rate in Hz
        expected_duration: Expected duration in seconds
        
    Returns:
        Dictionary with validation results
    """
    results = {'valid': True, 'warnings': [], 'errors': []}
    
    # Check data type
    if not isinstance(strain, np.ndarray):
        results['errors'].append("Data is not a numpy array")
        results['valid'] = False
        return results
    
    # Check dimensions
    if strain.ndim != 1:
        results['errors'].append(f"Data has {strain.ndim} dimensions, expected 1")
        results['valid'] = False
    
    # Check duration
    actual_duration = len(strain) / sample_rate
    if expected_duration and abs(actual_duration - expected_duration) > 0.1:
        results['warnings'].append(
            f"Duration mismatch: expected {expected_duration}s, got {actual_duration}s"
        )
    
    # Check for NaN or infinite values
    if np.any(np.isnan(strain)):
        results['errors'].append("Data contains NaN values")
        results['valid'] = False
    
    if np.any(np.isinf(strain)):
        results['errors'].append("Data contains infinite values")
        results['valid'] = False
    
    # Check amplitude range
    rms = np.sqrt(np.mean(strain**2))
    if rms > 1e-18:  # Typical strain amplitude
        results['warnings'].append(f"Large RMS amplitude: {rms:.2e}")
    
    # Check for constant values
    if np.std(strain) == 0:
        results['errors'].append("Data contains only constant values")
        results['valid'] = False
    
    return results
```

## References

### Data Sources

1. LIGO Open Science Center: https://www.gw-openscience.org/
2. Virgo Open Data: https://www.ego-gw.it/
3. Gravitational Wave Open Science Center: https://www.gw-openscience.org/

### Software Libraries

4. GWpy: https://gwpy.github.io/
5. PyCBC: https://pycbc.org/
6. LALSuite: https://lscsoft.docs.ligo.org/lalsuite/

### Data Formats and Standards

7. LIGO Algorithm Library (LAL): https://lscsoft.docs.ligo.org/lalsuite/lal/
8. Gravitational Wave Frame Format: https://dcc.ligo.org/LIGO-T970130/public

### Publications

9. Abbott, B.P., et al. (2019). "GWTC-1: A Gravitational-Wave Transient Catalog of Compact Binary Mergers Observed by LIGO and Virgo during the First and Second Observing Runs." Physical Review X 9, 031040.

10. Vallisneri, M., et al. (2015). "The LIGO Open Science Center." Journal of Physics: Conference Series 610, 012021.

---

*This document provides comprehensive information about gravitational wave data sources and formats. For the latest data releases and technical updates, please refer to the official LIGO and Virgo documentation.*
