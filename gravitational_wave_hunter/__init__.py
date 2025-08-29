"""
Gravitational Wave Hunter - Deep Learning Framework for Gravitational Wave Detection

A comprehensive framework for detecting gravitational waves using deep learning
approaches including CNN-LSTM, Transformer, WaveNet, and Autoencoder architectures.
"""

__version__ = "0.1.0"
__author__ = "Gravitational Wave Hunter Team"
__email__ = "info@gravitationalwavehunter.org"
__description__ = "Deep learning framework for gravitational wave detection"
__url__ = "https://github.com/gravitationalwavehunter/gw-hunter"
__license__ = "MIT"

# Essential imports that the notebook needs  
try:
    from .data.loader import load_simulated_data, generate_chirp_signal
    from .signal_processing.preprocessing import preprocess_strain_data
except ImportError:
    # Functions don't exist yet, will use fallback in notebook
    pass

# Main detector class will be imported when fully implemented
# from .detector import GWDetector

__all__ = [
    "load_simulated_data",
    "generate_chirp_signal", 
    "preprocess_strain_data",
]