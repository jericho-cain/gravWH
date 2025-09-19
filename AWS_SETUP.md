# AWS EC2 Setup Guide for Gravitational Wave Analysis

This guide provides step-by-step instructions for setting up an AWS EC2 instance to run the gravitational wave detection pipeline.

## 🚀 Instance Requirements

### Recommended Configuration
- **Instance Type**: m5.xlarge or larger
- **vCPUs**: 4+ cores
- **Memory**: 16+ GB RAM
- **Storage**: 50+ GB EBS volume
- **Network**: Standard VPC with internet access

### Minimum Configuration
- **Instance Type**: t3.large
- **vCPUs**: 2 cores
- **Memory**: 8 GB RAM
- **Storage**: 30+ GB EBS volume

## 📋 Step-by-Step Setup

### 1. Launch EC2 Instance

1. **Log into AWS Console**
   - Navigate to EC2 service
   - Click "Launch Instance"

2. **Choose AMI**
   - **Recommended**: Amazon Linux 2 AMI (HVM)
   - **Alternative**: Ubuntu Server 20.04 LTS

3. **Select Instance Type**
   - Choose m5.xlarge or larger
   - Ensure sufficient vCPUs and memory

4. **Configure Instance**
   - **Storage**: Add 50+ GB EBS volume
   - **Security Group**: Allow SSH (port 22) from your IP
   - **Key Pair**: Create or select existing key pair

5. **Launch Instance**
   - Review configuration
   - Launch instance
   - Download key pair (.pem file)

### 2. Connect to Instance

#### Using SSH (Linux/Mac)
```bash
chmod 400 your-key-pair.pem
ssh -i your-key-pair.pem ec2-user@your-instance-ip
```

#### Using PuTTY (Windows)
1. Convert .pem to .ppk using PuTTYgen
2. Use PuTTY to connect with .ppk file

### 3. System Setup

#### For Amazon Linux 2
```bash
# Update system
sudo yum update -y

# Install development tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git wget curl

# Install Python 3.9
sudo yum install -y python39 python39-pip python39-devel
```

#### For Ubuntu
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install development tools
sudo apt install -y build-essential git wget curl

# Install Python 3.9
sudo apt install -y python3.9 python3.9-pip python3.9-dev
```

### 4. Install Miniconda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# Reload shell
source ~/.bashrc

# Verify installation
conda --version
```

### 5. Create Python Environment

```bash
# Create conda environment
conda create -n gwn-py39 python=3.9 -y

# Activate environment
conda activate gwn-py39

# Verify Python version
python --version
```

### 6. Install Required Packages

```bash
# Install core packages
pip install -r requirements.txt

# Install additional dependencies
conda install -c conda-forge python-nds2-client -y
pip install torchviz

# Install system Graphviz (required for torchviz)
sudo yum install -y graphviz  # Amazon Linux
# or
sudo apt install -y graphviz  # Ubuntu
```

### 7. Clone Repository

```bash
# Clone the repository
git clone <repository-url>
cd gravitational-wave-hunter

# Verify structure
ls -la
```

### 8. Test Installation

```bash
# Test Python imports
python -c "import torch; import gwpy; print('Installation successful!')"

# Test data download
python scripts/check_available_data.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Add to ~/.bashrc
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage if needed
```

### Disk Space Management
```bash
# Check disk usage
df -h

# Clean up if needed
rm -rf ~/.astropy/cache
rm -rf ~/.cache/pip
```

## 📊 Performance Optimization

### Instance Scaling
- **Small datasets**: t3.large (2 vCPUs, 8 GB RAM)
- **Medium datasets**: m5.xlarge (4 vCPUs, 16 GB RAM)
- **Large datasets**: m5.2xlarge (8 vCPUs, 32 GB RAM)

### Storage Optimization
- Use EBS GP3 for better performance
- Consider EBS optimization for large data processing
- Monitor disk usage during data downloads

### Network Optimization
- Use instances in same region as data sources
- Consider VPC endpoints for S3 access
- Monitor network usage during downloads

## 🛠️ Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod 400 your-key-pair.pem
   ```

2. **Disk Space Full**
   ```bash
   # Clean cache directories
   rm -rf ~/.astropy/cache
   rm -rf ~/.cache/pip
   
   # Check disk usage
   df -h
   ```

3. **Package Installation Fails**
   ```bash
   # Update pip
   pip install --upgrade pip
   
   # Install with verbose output
   pip install -v package-name
   ```

4. **NDS2 Authentication Error**
   ```bash
   # Install NDS2 client
   conda install -c conda-forge python-nds2-client
   ```

5. **Graphviz Not Found**
   ```bash
   # Install system Graphviz
   sudo yum install -y graphviz  # Amazon Linux
   sudo apt install -y graphviz  # Ubuntu
   ```

### Monitoring Commands

```bash
# Check system resources
htop
df -h
free -h

# Check network connectivity
ping google.com
curl -I https://gwosc.org

# Check Python environment
conda list
pip list
```

## 💰 Cost Optimization

### Instance Management
- **Stop instances** when not in use
- **Use spot instances** for long-running tasks
- **Monitor usage** with AWS Cost Explorer

### Storage Management
- **Delete temporary files** regularly
- **Use lifecycle policies** for S3 storage
- **Monitor EBS usage** and resize as needed

### Data Management
- **Cache downloads** to avoid re-downloading
- **Use compression** for large datasets
- **Clean up** old model checkpoints

## 🔒 Security Best Practices

1. **Key Pair Management**
   - Store .pem files securely
   - Use different keys for different projects
   - Rotate keys regularly

2. **Network Security**
   - Use security groups to restrict access
   - Enable VPC flow logs
   - Use private subnets when possible

3. **Data Security**
   - Encrypt EBS volumes
   - Use IAM roles for AWS services
   - Enable CloudTrail logging

## 📞 Support

### AWS Support
- **Basic Support**: Included with AWS account
- **Developer Support**: $29/month
- **Business Support**: $100/month

### Community Resources
- AWS EC2 Documentation
- Gravitational Wave Open Science Center
- GWpy Documentation

---

**Note**: This guide assumes basic familiarity with AWS EC2. For advanced configurations or troubleshooting, consult AWS documentation or support.
