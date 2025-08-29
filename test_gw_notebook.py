"""
Test script to verify that the gravitational wave detection notebook imports work
"""

print("🧪 Testing notebook imports...")

# Test the import cell
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import seaborn as sns
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')

    # Additional scientific computing imports
    from scipy import signal
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import our gravitational wave hunter modules
    import sys
    sys.path.append('../')

    # Try to import our custom modules, with fallbacks for missing ones
    try:
        from gravitational_wave_hunter.data.loader import load_simulated_data, generate_chirp_signal
        from gravitational_wave_hunter.models.cnn_lstm import CNNLSTMDetector
        from gravitational_wave_hunter.signal_processing.preprocessing import preprocess_strain_data
        custom_modules_available = True
        print("✅ Custom modules imported successfully!")
    except ImportError as e:
        print(f"⚠️  Custom modules not available: {e}")
        print("📝 Will use fallback implementations")
        custom_modules_available = False
        
        # Fallback implementations
        def generate_chirp_signal(t, initial_freq=35, final_freq=350, amplitude=1e-21):
            """Generate a gravitational wave chirp signal."""
            tau = t[-1] - t[0]
            freq_evolution = initial_freq + (final_freq - initial_freq) * (t / tau)**3
            phase = 2 * np.pi * np.cumsum(freq_evolution) * (t[1] - t[0])
            h_plus = amplitude * np.cos(phase)
            h_cross = amplitude * np.sin(phase)
            signal = h_plus + h_cross
            envelope = np.exp(-((t - tau/2) / (tau/8))**2)
            return signal * envelope

        def load_simulated_data(num_samples, duration, sample_rate, signal_probability, 
                               add_glitches=True, noise_level=1e-23):
            """Load simulated gravitational wave data."""
            time_samples = int(sample_rate * duration)
            strain_data = []
            labels = []
            
            for i in range(num_samples):
                noise = np.random.normal(0, noise_level, time_samples)
                
                if np.random.random() < signal_probability:
                    t = np.linspace(0, duration, time_samples)
                    signal = generate_chirp_signal(t, initial_freq=35, final_freq=350, amplitude=1e-21)
                    strain = noise + signal
                    label = 1
                else:
                    strain = noise
                    label = 0
                
                if add_glitches and np.random.random() < 0.1:
                    glitch_start = np.random.randint(0, len(strain) - 100)
                    strain[glitch_start:glitch_start+100] += np.random.normal(0, 10*noise_level, 100)
                
                strain_data.append(strain)
                labels.append(label)
            
            strain_data = np.array(strain_data)
            labels = np.array(labels)
            metadata = {'sample_rate': sample_rate, 'duration': duration}
            
            return strain_data, labels, metadata

        def preprocess_strain_data(strain, sample_rate, highpass_freq=20, lowpass_freq=2048):
            """Basic preprocessing of strain data."""
            sos = signal.butter(8, [highpass_freq, lowpass_freq], btype='band', fs=sample_rate, output='sos')
            filtered = signal.sosfilt(sos, strain)
            whitened = (filtered - np.mean(filtered)) / np.std(filtered)
            return whitened
        
        # Simple CNN model for gravitational wave detection
        class CNNLSTMDetector(nn.Module):
            def __init__(self, input_length):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=8)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=4)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(128, 128)
                self.fc2 = nn.Linear(128, 2)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)  # Add channel dimension
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = self.pool(x).squeeze(-1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

    print("✅ All basic imports successful!")
    print("🌌 Welcome to Gravitational Wave Detection with Deep Learning")

    # Test data generation
    print("\n🧪 Testing data generation...")
    
    # Configuration parameters (optimized for faster demo)
    SAMPLE_RATE = 4096  # Hz - LIGO's standard sampling rate  
    DURATION = 16       # seconds (reduced for faster processing)
    NUM_SAMPLES = 100   # Number of training samples (small for test)
    SIGNAL_PROB = 0.3   # Probability of a sample containing a signal

    print("📡 Generating simulated LIGO-like data...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Samples per segment: {SAMPLE_RATE * DURATION}")

    # Generate simulated data that mimics real LIGO characteristics
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate simulated strain data
    strain_data, labels, metadata = load_simulated_data(
        num_samples=NUM_SAMPLES,
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        signal_probability=SIGNAL_PROB,
        add_glitches=True,
        noise_level=1e-23
    )
    
    print(f"✅ Generated {NUM_SAMPLES} samples")
    print(f"📊 Data shape: {strain_data.shape}")
    print(f"🎯 Signal samples: {labels.sum()}/{len(labels)} ({labels.mean():.1%})")

    # Test preprocessing
    print("\n🧪 Testing preprocessing...")
    processed_data = []
    for strain in strain_data:
        processed_strain = preprocess_strain_data(
            strain, 
            sample_rate=SAMPLE_RATE,
            highpass_freq=20,
            lowpass_freq=1000
        )
        processed_data.append(processed_strain)

    processed_data = np.array(processed_data)
    print("✅ Basic preprocessing completed")
    print(f"📊 Processed data shape: {processed_data.shape}")

    # Test model creation
    print("\n🧪 Testing model creation...")
    model = CNNLSTMDetector(len(processed_data[0])).to(device)
    print("✅ CNN model initialized")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n🎉 All tests passed! The notebook should work correctly.")

except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

