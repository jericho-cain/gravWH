#!/usr/bin/env python3
"""
Generate a clean, professional GitHub banner for Gravitational Wave Hunter
Fixed version with proper text placement and no random equations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# Set up the figure with proper GitHub banner dimensions
fig, ax = plt.subplots(1, 1, figsize=(12.8, 6.4), dpi=100)
ax.set_xlim(0, 1280)
ax.set_ylim(0, 640)
ax.axis('off')

# Background gradient (dark space theme)
background = np.linspace(0, 1, 640)
for i, alpha in enumerate(background):
    ax.axhline(y=i, color='#0d1117', alpha=0.1 + 0.9 * alpha)

# Add subtle star field
np.random.seed(42)  # For reproducible star positions
star_x = np.random.uniform(0, 1280, 150)
star_y = np.random.uniform(0, 640, 150)
star_sizes = np.random.uniform(0.5, 2, 150)
star_alphas = np.random.uniform(0.3, 0.8, 150)

for x, y, size, alpha in zip(star_x, star_y, star_sizes, star_alphas):
    ax.scatter(x, y, s=size, c='white', alpha=alpha, marker='*')

# Main title
ax.text(640, 520, '🌌 Gravitational Wave Hunter', 
        fontsize=48, fontweight='bold', ha='center', va='center',
        color='white', fontfamily='sans-serif')

# Subtitle
ax.text(640, 460, 'CWT-LSTM Autoencoder for Signal Detection', 
        fontsize=28, ha='center', va='center',
        color='#58a6ff', fontweight='medium', fontfamily='sans-serif')

# Key features
ax.text(640, 380, 'Unsupervised Detection • Template-Free Discovery', 
        fontsize=20, ha='center', va='center',
        color='#7c3aed', fontweight='medium', fontfamily='sans-serif')

# Bottom tagline
ax.text(640, 280, 'Advanced Machine Learning for Gravitational Wave Astronomy', 
        fontsize=18, ha='center', va='center',
        color='#8b949e', fontweight='medium', fontfamily='sans-serif')

# Save the banner
plt.tight_layout()
plt.savefig('assets/github_banner.png', 
            bbox_inches='tight', 
            dpi=100, 
            facecolor='#0d1117',
            edgecolor='none')
plt.close()

print("SUCCESS: New clean GitHub banner generated: assets/github_banner.png")
print("INFO: Dimensions: 1280x640 pixels (GitHub standard)")
print("INFO: Minimalist design with no clutter")
print("INFO: Professional space theme with clean text only")
