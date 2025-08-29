"""
Simple, Fast Gravitational Wave Detection Demo

This script demonstrates a minimal version that won't freeze or get stuck.
Estimated runtime: < 1 minute
"""

print("🌌 Fast Gravitational Wave Detection Demo")
print("=" * 50)

# Setup and Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("✅ All imports successful!")

# Fast Data Generation
print("\n📡 Generating minimal simulated data...")

# Minimal configuration for speed
SAMPLE_RATE = 512   # Reduced from 4096
DURATION = 2        # Reduced from 16  
NUM_SAMPLES = 60    # Reduced from 300
SIGNAL_PROB = 0.5

print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"Duration: {DURATION} seconds")
print(f"Total samples: {NUM_SAMPLES}")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_simple_chirp(t, f0=50, f1=150):
    """Generate a simple chirp signal."""
    frequency = f0 + (f1 - f0) * (t / t[-1])
    phase = 2 * np.pi * np.cumsum(frequency) * (t[1] - t[0])
    envelope = np.exp(-((t - t[-1]/2) / (t[-1]/3))**2)
    return 1e-21 * np.sin(phase) * envelope

# Generate data quickly
time_samples = SAMPLE_RATE * DURATION
strain_data = []
labels = []

print("Generating samples... ", end="")
for i in range(NUM_SAMPLES):
    if i % 20 == 0:
        print(f"{i}", end=" ")
    
    # Simple noise
    noise = np.random.normal(0, 1e-23, time_samples)
    
    if np.random.random() < SIGNAL_PROB:
        # Add simple chirp
        t = np.linspace(0, DURATION, time_samples)
        signal = generate_simple_chirp(t)
        strain = noise + signal
        label = 1
    else:
        strain = noise
        label = 0
    
    strain_data.append(strain)
    labels.append(label)

strain_data = np.array(strain_data)
labels = np.array(labels)

print(f"\n✅ Generated {NUM_SAMPLES} samples")
print(f"📊 Data shape: {strain_data.shape}")
print(f"🎯 Signal samples: {labels.sum()}/{len(labels)} ({labels.mean():.1%})")

# Simple Preprocessing
print("\n🔧 Basic preprocessing...")

# Simple normalization only (no complex filtering)
processed_data = []
for strain in strain_data:
    # Just normalize
    normalized = (strain - np.mean(strain)) / (np.std(strain) + 1e-10)
    processed_data.append(normalized)

processed_data = np.array(processed_data)
print("✅ Preprocessing completed")
print(f"📊 Processed data shape: {processed_data.shape}")

# Simple Model and Training
print("\n🧠 Creating simple model...")

# Convert to tensors
X = torch.FloatTensor(processed_data)
y = torch.LongTensor(labels)

# Simple train/test split
train_size = int(0.8 * len(X))
indices = torch.randperm(len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Minimal model
class SimpleCNN(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=16, stride=2)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=8, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 2)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

model = SimpleCNN(len(processed_data[0])).to(device)
print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Quick training (only 3 epochs)
print("\n🚀 Quick training (3 epochs only)...")
model.train()

for epoch in range(3):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    print(f'Epoch {epoch+1}: Loss={avg_loss:.3f}, Accuracy={accuracy:.3f}')

print("✅ Training completed!")

# Quick Evaluation
print("\n🔍 Evaluating model...")

model.eval()
with torch.no_grad():
    X_test_gpu = X_test.to(device)
    outputs = model(X_test_gpu)
    predictions = outputs.argmax(dim=1).cpu().numpy()
    probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

# Calculate basic metrics
y_test_np = y_test.numpy()
accuracy = np.mean(y_test_np == predictions)

# Basic confusion matrix
tp = np.sum((predictions == 1) & (y_test_np == 1))
fp = np.sum((predictions == 1) & (y_test_np == 0))
tn = np.sum((predictions == 0) & (y_test_np == 0))
fn = np.sum((predictions == 0) & (y_test_np == 1))

print(f"📊 Test Accuracy: {accuracy:.3f}")
print(f"\n📈 Results:")
print(f"True Positives: {tp}, False Positives: {fp}")
print(f"True Negatives: {tn}, False Negatives: {fn}")

if tp + fp > 0:
    precision = tp / (tp + fp)
    print(f"Precision: {precision:.3f}")
if tp + fn > 0:
    recall = tp / (tp + fn)
    print(f"Recall: {recall:.3f}")

print(f"\n🎉 Fast Gravitational Wave Detection Demo Completed!")
print(f"📝 Summary:")
print(f"• Processed {NUM_SAMPLES} samples in {DURATION}s segments")
print(f"• Trained minimal CNN with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"• Achieved {accuracy:.1%} accuracy on test data")
print(f"• Demo completed successfully without freezing! 🚀")

# Save model for notebook use
torch.save(model.state_dict(), 'simple_gw_model.pth')
print(f"💾 Model saved as 'simple_gw_model.pth'")
