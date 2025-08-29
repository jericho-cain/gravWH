"""
Gravitational Wave Detection Demo Script

A complete workflow for detecting gravitational waves using deep learning.
This script demonstrates data generation, preprocessing, model training, and detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("✅ All imports successful!")
print("🌌 Welcome to Gravitational Wave Detection with Deep Learning")

# Configuration parameters
SAMPLE_RATE = 4096  # Hz - LIGO's standard sampling rate
DURATION = 32       # seconds
NUM_SAMPLES = 1000  # Number of training samples
SIGNAL_PROB = 0.3   # Probability of a sample containing a signal

print("📡 Generating simulated LIGO-like data...")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"Duration: {DURATION} seconds")
print(f"Samples per segment: {SAMPLE_RATE * DURATION}")

# Generate simulated data that mimics real LIGO characteristics
np.random.seed(42)
torch.manual_seed(42)

def generate_chirp_signal(t, initial_freq=35, final_freq=350, amplitude=1e-21):
    """Generate a gravitational wave chirp signal."""
    # Frequency evolution for a chirp
    tau = t[-1] - t[0]
    freq_evolution = initial_freq + (final_freq - initial_freq) * (t / tau)**3
    
    # Phase evolution
    phase = 2 * np.pi * np.cumsum(freq_evolution) * (t[1] - t[0])
    
    # Generate the waveform (simplified)
    h_plus = amplitude * np.cos(phase)
    h_cross = amplitude * np.sin(phase)
    
    # Combine polarizations
    signal = h_plus + h_cross
    
    # Apply envelope to make it more realistic
    envelope = np.exp(-((t - tau/2) / (tau/8))**2)
    
    return signal * envelope

def load_simulated_data(num_samples, duration, sample_rate, signal_probability, 
                       add_glitches=True, noise_level=1e-23):
    """Load simulated gravitational wave data."""
    time_samples = int(sample_rate * duration)
    strain_data = []
    labels = []
    
    for i in range(num_samples):
        # Generate background noise
        noise = np.random.normal(0, noise_level, time_samples)
        
        # Add gravitational wave signal with some probability
        if np.random.random() < signal_probability:
            # Generate a chirp signal
            t = np.linspace(0, duration, time_samples)
            signal = generate_chirp_signal(t, 
                                         initial_freq=35, 
                                         final_freq=350, 
                                         amplitude=1e-21)
            strain = noise + signal
            label = 1
        else:
            strain = noise
            label = 0
        
        # Add occasional glitches if requested
        if add_glitches and np.random.random() < 0.1:
            glitch_start = np.random.randint(0, len(strain) - 100)
            strain[glitch_start:glitch_start+100] += np.random.normal(0, 10*noise_level, 100)
        
        strain_data.append(strain)
        labels.append(label)
    
    strain_data = np.array(strain_data)
    labels = np.array(labels)
    metadata = {'sample_rate': sample_rate, 'duration': duration}
    
    return strain_data, labels, metadata

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

# Data Preprocessing
print("🔧 Preprocessing strain data...")

def preprocess_strain_data(strain, sample_rate, highpass_freq=20, lowpass_freq=2048):
    """Basic preprocessing of strain data."""
    from scipy import signal
    
    # Basic bandpass filter
    sos = signal.butter(8, [highpass_freq, lowpass_freq], btype='band', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, strain)
    
    # Simple whitening (normalize)
    whitened = (filtered - np.mean(filtered)) / np.std(filtered)
    
    return whitened

# Visualize raw data first
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot examples of signals and noise
signal_idx = np.where(labels == 1)[0][0]
noise_idx = np.where(labels == 0)[0][0]

time_axis = np.linspace(0, DURATION, len(strain_data[0]))

# Raw signal example
axes[0, 0].plot(time_axis, strain_data[signal_idx])
axes[0, 0].set_title(f'Raw Strain Data (with GW signal)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Strain')

# Raw noise example
axes[0, 1].plot(time_axis, strain_data[noise_idx])
axes[0, 1].set_title(f'Raw Strain Data (noise only)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Strain')

# Preprocess the data
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

# Plot processed data
axes[1, 0].plot(time_axis, processed_data[signal_idx])
axes[1, 0].set_title('Processed Strain Data (with GW signal)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Normalized Strain')

axes[1, 1].plot(time_axis, processed_data[noise_idx])
axes[1, 1].set_title('Processed Strain Data (noise only)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Normalized Strain')

plt.tight_layout()
plt.show()

print(f"📊 Processed data shape: {processed_data.shape}")
print(f"📈 Data range: [{processed_data.min():.3f}, {processed_data.max():.3f}]")
print(f"📊 Data std: {processed_data.std():.3f}")

# Model Training
print("🧠 Setting up neural network models...")

# Prepare data for training
X = torch.FloatTensor(processed_data)
y = torch.LongTensor(labels)

# Split data
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

# Create train/val/test splits
indices = torch.randperm(len(X))
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]
X_test, y_test = X[test_indices], y[test_indices]

print(f"📊 Training set: {len(X_train)} samples")
print(f"📊 Validation set: {len(X_val)} samples") 
print(f"📊 Test set: {len(X_test)} samples")

# Create data loaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Simple CNN model for gravitational wave detection
class SimpleCNN(nn.Module):
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
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN(len(processed_data[0])).to(device)
print("✅ CNN model initialized")
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print("🚀 Starting training...")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
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
    
    return total_loss / len(train_loader), correct / total

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), correct / total

# Training loop
num_epochs = 15
best_val_acc = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"🏃‍♂️ Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_gw_detector.pth')
    
    if (epoch + 1) % 3 == 0:
        print(f'Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, '
              f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}')

print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.3f}")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss', alpha=0.8)
ax1.plot(val_losses, label='Validation Loss', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label='Train Accuracy', alpha=0.8)
ax2.plot(val_accs, label='Validation Accuracy', alpha=0.8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Model Evaluation
print("🔍 Testing the trained model on new data...")

# Load the best model
model.load_state_dict(torch.load('best_gw_detector.pth'))
model.eval()

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of signal class
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

# Get predictions
predictions, true_labels, signal_probs = evaluate_model(model, test_loader, device)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
auc = roc_auc_score(true_labels, signal_probs)

print("📊 Model Performance Metrics:")
print(f"   Accuracy:  {accuracy:.3f}")
print(f"   Precision: {precision:.3f}")
print(f"   Recall:    {recall:.3f}")
print(f"   F1-Score:  {f1:.3f}")
print(f"   AUC-ROC:   {auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print(f"\n📈 Confusion Matrix:")
print(f"   True Negatives:  {cm[0,0]:3d}")
print(f"   False Positives: {cm[0,1]:3d}")
print(f"   False Negatives: {cm[1,0]:3d}")
print(f"   True Positives:  {cm[1,1]:3d}")

# Plot ROC curve and probability distributions
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(true_labels, signal_probs)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(signal_probs[true_labels == 0], bins=30, alpha=0.7, label='Noise', density=True)
plt.hist(signal_probs[true_labels == 1], bins=30, alpha=0.7, label='Signal', density=True)
plt.xlabel('Signal Probability')
plt.ylabel('Density')
plt.title('Signal Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot detection examples
plt.subplot(1, 3, 3)
# Find a true positive example
tp_indices = np.where((predictions == 1) & (true_labels == 1))[0]
if len(tp_indices) > 0:
    example_idx = tp_indices[0]
    time_axis = np.linspace(0, DURATION, len(X_test[0]))
    plt.plot(time_axis, X_test[example_idx].numpy())
    plt.title(f'Detected Signal (Confidence: {signal_probs[example_idx]:.3f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("🎯 Key Insights from the Analysis:")
print(f"• The model achieved {accuracy:.1%} accuracy on unseen data")
print(f"• Detection sensitivity (recall): {recall:.1%}")
print(f"• False alarm rate: {1-precision:.1%}")
print(f"• The model can distinguish signals from noise with AUC = {auc:.3f}")
print("• This demonstrates the potential for deep learning in gravitational wave astronomy")

print("\n🎉 Gravitational Wave Detection Demo Completed Successfully!")
print("This script demonstrated:")
print("• Simulated gravitational wave data generation")
print("• Signal preprocessing and filtering") 
print("• Deep learning model training")
print("• Performance evaluation and visualization")
print("• Real-world applicability for LIGO-like data")

