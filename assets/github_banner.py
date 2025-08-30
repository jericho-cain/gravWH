#!/usr/bin/env python3
"""
Generate a professional GitHub repository banner for Gravitational Wave Hunter
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Create figure with specific GitHub banner dimensions (1280x640 recommended)
fig, ax = plt.subplots(figsize=(16, 8), dpi=80)

# Set dark space background
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Create gradient background effect
x = np.linspace(0, 10, 100)
y = np.linspace(0, 5, 50)
X, Y = np.meshgrid(x, y)

# Gravitational wave pattern (chirp signal)
def gravitational_wave(x, y, t=0):
    # Frequency increases over time (chirp)
    freq = 2 + 0.5 * x  # Frequency sweep
    amplitude = np.exp(-0.1 * x) * np.exp(-0.5 * (y - 2.5)**2)  # Gaussian envelope
    return amplitude * np.sin(2 * np.pi * freq * x + t)

# Generate wave pattern
wave = gravitational_wave(X, Y)

# Plot the gravitational wave as a subtle background
contour = ax.contourf(X, Y, wave, levels=20, cmap='plasma', alpha=0.3)

# Add main title
ax.text(5, 4.2, '🌌 GRAVITATIONAL WAVE HUNTER', 
        fontsize=48, fontweight='bold', ha='center', va='center',
        color='white', family='monospace')

# Add subtitle
ax.text(5, 3.6, 'CWT-LSTM Autoencoder for Advanced Signal Detection', 
        fontsize=24, ha='center', va='center',
        color='#7c3aed', family='sans-serif')

# Add performance metrics in boxes
metrics = [
    ('🎯 90.6%', 'PRECISION', '#10b981'),
    ('📈 67.6%', 'RECALL', '#3b82f6'), 
    ('⚡ 0.821', 'AUC SCORE', '#f59e0b'),
    ('🏆 0.788', 'AVG PRECISION', '#ef4444')
]

start_x = 1.5
for i, (value, label, color) in enumerate(metrics):
    x_pos = start_x + i * 2
    
    # Background box
    rect = Rectangle((x_pos - 0.7, 1.8), 1.4, 1.2, 
                    facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    
    # Metric value
    ax.text(x_pos, 2.6, value, fontsize=20, fontweight='bold', 
           ha='center', va='center', color=color)
    
    # Metric label
    ax.text(x_pos, 2.2, label, fontsize=12, fontweight='bold',
           ha='center', va='center', color='white')

# Add technology badges
technologies = ['PyTorch', 'CWT', 'LSTM', 'Anomaly Detection', 'LIGO Standard']
badge_colors = ['#ee4c2c', '#7c3aed', '#10b981', '#f59e0b', '#3b82f6']

for i, (tech, color) in enumerate(zip(technologies, badge_colors)):
    x_pos = 1 + i * 1.6
    y_pos = 0.8
    
    # Badge background
    rect = Rectangle((x_pos - 0.4, y_pos - 0.2), 0.8, 0.4,
                    facecolor=color, alpha=0.8, edgecolor='white', linewidth=1)
    ax.add_patch(rect)
    
    # Badge text
    ax.text(x_pos, y_pos, tech, fontsize=10, fontweight='bold',
           ha='center', va='center', color='white')

# Add wave equation (decorative)
ax.text(8.5, 1.2, r'$h(t) = A \cos(2\pi f(t) t + \phi)$', 
        fontsize=16, ha='center', va='center', color='#7c3aed',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#0d1117', edgecolor='#7c3aed'))

# Add GitHub link hint
ax.text(5, 0.3, 'github.com/jericho-cain/gravWH', 
        fontsize=14, ha='center', va='center', color='#6b7280',
        style='italic')

# Clean up axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Tight layout
plt.tight_layout()

# Save as high-quality PNG
plt.savefig('github_banner.png', dpi=200, bbox_inches='tight', 
           facecolor='#0d1117', edgecolor='none')

print("🎨 GitHub banner created: github_banner.png")
print("📏 Dimensions: 1280x640 (GitHub recommended)")
print("🎯 Features: Dark theme, metrics, tech stack, wave pattern")
