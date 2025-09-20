#!/usr/bin/env python3
"""
Create CWT comparison visualizations using real cached LIGO data
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import glob
import sys

# Add the project root to the sys.path
sys.path.append('..')

def load_cached_gw_data(num_samples=3):
    """Load cached gravitational wave data."""
    gw_files = glob.glob('gw_events_cache/*.npz')
    
    if len(gw_files) == 0:
        print("No cached GW data found")
        return []
    
    # Load up to num_samples
    samples = []
    for i, file_path in enumerate(gw_files[:num_samples]):
        try:
            data = np.load(file_path)
            samples.append({
                'strain': data['strain'],
                'times': data['times'] if 'times' in data else np.linspace(0, 4, len(data['strain'])),
                'sample_rate': data['sample_rate'] if 'sample_rate' in data else 4096
            })
            print(f"Loaded GW sample {i+1}: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return samples

def load_cached_noise_data(num_samples=3):
    """Load cached noise data."""
    noise_files = glob.glob('ligo_data_cache/O1_H1_*.npz')
    
    if len(noise_files) == 0:
        print("No cached noise data found")
        return []
    
    # Load up to num_samples
    samples = []
    for i, file_path in enumerate(noise_files[:num_samples]):
        try:
            data = np.load(file_path)
            samples.append({
                'strain': data['strain'],
                'times': data['times'] if 'times' in data else np.linspace(0, 4, len(data['strain'])),
                'sample_rate': data['sample_rate'] if 'sample_rate' in data else 4096
            })
            print(f"Loaded noise sample {i+1}: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return samples

def create_time_to_cwt_figure():
    """Create figure showing time domain → CWT domain transformation."""
    
    print("Loading cached gravitational wave data...")
    gw_samples = load_cached_gw_data(1)
    
    if len(gw_samples) == 0:
        print("No GW data available")
        return
    
    # Extract the first GW sample
    gw_sample = gw_samples[0]
    strain = gw_sample['strain']
    times = gw_sample['times']
    
    print(f"GW signal shape: {strain.shape}, times shape: {times.shape}")
    
    # Apply CWT
    scales = np.logspace(0, 2, 64)  # 64 scales
    frequencies = pywt.scale2frequency('morl', scales) / (1/4096)  # Convert to Hz
    
    # Perform CWT
    coefficients, _ = pywt.cwt(strain, scales, 'morl')
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left panel: Time domain
    ax1.plot(times, strain, 'b-', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Strain')
    ax1.set_title('Gravitational Wave Signal (Time Domain)')
    ax1.grid(True, alpha=0.3)
    
    # Right panel: CWT domain
    # Scale color map to data range for better visibility
    coeff_abs = np.abs(coefficients)
    vmin = np.percentile(coeff_abs, 5)  # 5th percentile
    vmax = np.percentile(coeff_abs, 95)  # 95th percentile
    
    im = ax2.imshow(coeff_abs, aspect='auto', origin='lower', 
                    extent=[times[0], times[-1], frequencies[-1], frequencies[0]],
                    cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('CWT Spectrogram (Time-Frequency Domain)')
    ax2.set_yscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('|CWT Coefficient|')
    
    plt.tight_layout()
    plt.savefig('results/time_to_cwt_transformation_real.png', dpi=300, bbox_inches='tight')
    print("Saved: results/time_to_cwt_transformation_real.png")
    plt.close()
    
    return coefficients, frequencies, times

def create_frequency_band_comparison():
    """Create figure comparing frequency bands between noise and gravitational waves."""
    
    print("Loading cached data for comparison...")
    
    # Load cached data
    noise_samples = load_cached_noise_data(3)
    gw_samples = load_cached_gw_data(3)
    
    if len(noise_samples) == 0 or len(gw_samples) == 0:
        print("Insufficient cached data available")
        return
    
    # Process samples
    noise_coeffs = []
    gw_coeffs = []
    
    # Apply CWT to noise samples
    scales = np.logspace(0, 2, 64)
    frequencies = pywt.scale2frequency('morl', scales) / (1/4096)
    
    for sample in noise_samples:
        coefficients, _ = pywt.cwt(sample['strain'], scales, 'morl')
        noise_coeffs.append(coefficients)
    
    for sample in gw_samples:
        coefficients, _ = pywt.cwt(sample['strain'], scales, 'morl')
        gw_coeffs.append(coefficients)
    
    # Define frequency bands
    freq_ranges = [
        (20, 50),    # Low frequency band
        (50, 100),   # Mid frequency band  
        (100, 200)   # High frequency band
    ]
    
    # Create the comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot noise samples (top row)
    for i, (low_freq, high_freq) in enumerate(freq_ranges):
        ax = axes[0, i]
        
        # Find frequency band
        mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        band_data = noise_coeffs[0][mask, :]  # Use first noise sample
        band_freqs = frequencies[mask]
        
        # Plot frequency vs time with scaled color map
        band_abs = np.abs(band_data)
        vmin = np.percentile(band_abs, 5)  # 5th percentile
        vmax = np.percentile(band_abs, 95)  # 95th percentile
        
        im = ax.imshow(band_abs, aspect='auto', origin='lower',
                       extent=[0, 4, band_freqs[0], band_freqs[-1]],
                       cmap='viridis', vmin=vmin, vmax=vmax)
        
        ax.set_title(f'Noise - {low_freq}-{high_freq} Hz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
    
    # Plot GW samples (bottom row)
    for i, (low_freq, high_freq) in enumerate(freq_ranges):
        ax = axes[1, i]
        
        # Find frequency band
        mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        band_data = gw_coeffs[0][mask, :]  # Use first GW sample
        band_freqs = frequencies[mask]
        
        # Plot frequency vs time with scaled color map
        band_abs = np.abs(band_data)
        vmin = np.percentile(band_abs, 5)  # 5th percentile
        vmax = np.percentile(band_abs, 95)  # 95th percentile
        
        im = ax.imshow(band_abs, aspect='auto', origin='lower',
                       extent=[0, 4, band_freqs[0], band_freqs[-1]],
                       cmap='viridis', vmin=vmin, vmax=vmax)
        
        ax.set_title(f'Gravitational Wave - {low_freq}-{high_freq} Hz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
    
    # Add overall title
    fig.suptitle('Frequency Band Comparison: Noise vs Gravitational Waves (Real LIGO Data)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('results/frequency_band_comparison_real.png', dpi=300, bbox_inches='tight')
    print("Saved: results/frequency_band_comparison_real.png")
    plt.close()

def main():
    """Main function to create both visualizations."""
    print("Creating CWT comparison visualizations with real cached LIGO data...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create time domain → CWT domain figure
    print("Creating time domain → CWT domain transformation figure...")
    coefficients, frequencies, times = create_time_to_cwt_figure()
    
    # Create frequency band comparison figure
    print("Creating frequency band comparison figure...")
    create_frequency_band_comparison()
    
    print("Visualizations complete! Check results/ directory for saved figures.")

if __name__ == "__main__":
    main()
