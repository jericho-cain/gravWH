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

# Performance metrics box
metrics_box = FancyBboxPatch((200, 280), 880, 120, 
                            boxstyle="round,pad=0.02", 
                            facecolor='#21262d', 
                            edgecolor='#30363d', 
                            linewidth=2)
ax.add_patch(metrics_box)

# Metrics text
ax.text(640, 340, '90.6% Precision • 67.6% Recall • AUC: 0.821', 
        fontsize=24, ha='center', va='center',
        color='white', fontweight='bold', fontfamily='sans-serif')

# Key features
ax.text(640, 300, 'Unsupervised Detection • Template-Free Discovery', 
        fontsize=20, ha='center', va='center',
        color='#7c3aed', fontweight='medium', fontfamily='sans-serif')

# LIGO-inspired detector visualization (simplified)
detector_x = 1000
detector_y = 150
detector_size = 80

# Detector arms
arm_length = 60
ax.plot([detector_x-arm_length, detector_x+arm_length], [detector_y, detector_y], 
        color='#58a6ff', linewidth=4, solid_capstyle='round')
ax.plot([detector_x, detector_x], [detector_y-arm_length, detector_y+arm_length], 
        color='#58a6ff', linewidth=4, solid_capstyle='round')

# Central detector
detector_circle = plt.Circle((detector_x, detector_y), 15, 
                           facecolor='#21262d', edgecolor='#58a6ff', linewidth=3)
ax.add_patch(detector_circle)

# Performance graph (simplified)
graph_x = 200
graph_y = 150
graph_width = 200
graph_height = 80

# Graph background
graph_bg = Rectangle((graph_x, graph_y), graph_width, graph_height, 
                    facecolor='#21262d', edgecolor='#30363d', linewidth=2)
ax.add_patch(graph_bg)

# Simple precision-recall curve
x_curve = np.linspace(0, 1, 50)
y_curve = 0.9 * (1 - np.exp(-3 * x_curve))  # Approximate PR curve
curve_x = graph_x + 20 + x_curve * (graph_width - 40)
curve_y = graph_y + 20 + y_curve * (graph_height - 40)
ax.plot(curve_x, curve_y, color='#7c3aed', linewidth=3)

# Graph labels
ax.text(graph_x + graph_width//2, graph_y - 20, 'Performance', 
        fontsize=16, ha='center', va='center',
        color='#58a6ff', fontweight='medium', fontfamily='sans-serif')

# Bottom tagline
ax.text(640, 80, 'Advanced Machine Learning for Gravitational Wave Astronomy', 
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

print("✅ New GitHub banner generated: assets/github_banner.png")
print("📏 Dimensions: 1280x640 pixels (GitHub standard)")
print("🎨 Clean design with no text overflow or random equations")
print("🌌 Professional space theme with proper text placement")
