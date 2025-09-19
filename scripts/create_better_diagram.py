#!/usr/bin/env python3
"""
Create a better model architecture diagram using matplotlib
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    """Create a clean architecture diagram using matplotlib."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define colors
    input_color = '#E3F2FD'
    encoder_color = '#BBDEFB'
    latent_color = '#90CAF9'
    decoder_color = '#BBDEFB'
    output_color = '#E3F2FD'
    
    # Input
    input_box = FancyBboxPatch((0.5, 6.5), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 7, 'CWT Input\n(1, 1, 8, 4096)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Encoder
    encoder_box = FancyBboxPatch((3.5, 6.5), 2, 1, boxstyle="round,pad=0.1", 
                                facecolor=encoder_color, edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(4.5, 7, 'Encoder\n(Bidirectional LSTM)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Linear Encoder
    linear_enc_box = FancyBboxPatch((3.5, 4.5), 2, 1, boxstyle="round,pad=0.1", 
                                   facecolor=encoder_color, edgecolor='black', linewidth=2)
    ax.add_patch(linear_enc_box)
    ax.text(4.5, 5, 'Linear Encoder\n(64→32)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Latent Space
    latent_box = FancyBboxPatch((3.5, 2.5), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor=latent_color, edgecolor='black', linewidth=2)
    ax.add_patch(latent_box)
    ax.text(4.5, 3, 'Latent Space\n(32 dims)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Linear Decoder
    linear_dec_box = FancyBboxPatch((6.5, 2.5), 2, 1, boxstyle="round,pad=0.1", 
                                   facecolor=decoder_color, edgecolor='black', linewidth=2)
    ax.add_patch(linear_dec_box)
    ax.text(7.5, 3, 'Linear Decoder\n(32→64)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Decoder
    decoder_box = FancyBboxPatch((6.5, 4.5), 2, 1, boxstyle="round,pad=0.1", 
                                facecolor=decoder_color, edgecolor='black', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(7.5, 5, 'Decoder\n(Bidirectional LSTM)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Output
    output_box = FancyBboxPatch((6.5, 6.5), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7.5, 7, 'Reconstruction\n(1, 1, 8, 4096)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Anomaly Score
    anomaly_box = FancyBboxPatch((3.5, 0.5), 2, 1, boxstyle="round,pad=0.1", 
                                facecolor='#FFCDD2', edgecolor='black', linewidth=2)
    ax.add_patch(anomaly_box)
    ax.text(4.5, 1, 'Anomaly Score\nMSE(Input, Output)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Arrows
    # Input to Encoder
    ax.annotate('', xy=(3.5, 7), xytext=(2.5, 7), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Encoder to Linear Encoder
    ax.annotate('', xy=(4.5, 5.5), xytext=(4.5, 6.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Linear Encoder to Latent
    ax.annotate('', xy=(4.5, 3.5), xytext=(4.5, 4.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Latent to Linear Decoder
    ax.annotate('', xy=(6.5, 3), xytext=(5.5, 3), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Linear Decoder to Decoder
    ax.annotate('', xy=(7.5, 4.5), xytext=(7.5, 3.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Decoder to Output
    ax.annotate('', xy=(7.5, 6.5), xytext=(7.5, 5.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Output to Anomaly Score
    ax.annotate('', xy=(4.5, 1.5), xytext=(7.5, 6.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Title
    ax.text(5, 7.8, 'CWT-LSTM Autoencoder Architecture', ha='center', va='center', 
            fontsize=16, weight='bold')
    
    # Add some details
    ax.text(0.2, 5.5, 'Encoder', ha='center', va='center', fontsize=12, weight='bold', 
            rotation=90, color='blue')
    ax.text(9.8, 5.5, 'Decoder', ha='center', va='center', fontsize=12, weight='bold', 
            rotation=90, color='green')
    
    plt.tight_layout()
    plt.savefig('model_architecture_clean.png', dpi=300, bbox_inches='tight')
    print("Clean architecture diagram saved as model_architecture_clean.png")
    
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()
