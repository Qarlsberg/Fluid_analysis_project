# Deployment Guide - Fluid Analysis Application

This guide provides step-by-step instructions for data engineers to deploy and operate the Fluid Analysis Application in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation Steps](#installation-steps)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Testing the Deployment](#testing-the-deployment)
7. [Production Considerations](#production-considerations)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Verification

Before beginning deployment, verify your system meets these requirements:

```bash
# Check Python version (requires 3.8+)
python --version

# Check available memory (requires 8GB+)
# Windows:
systeminfo | findstr "Total Physical Memory"
# Linux/macOS:
free -h

# Check disk space (requires 2GB+)
# Windows:
dir
# Linux/macOS:
df -h
```

### Required Software

1. **Python 3.8 or higher**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure pip is included in installation

2. **Git** (optional, for cloning repository)
   - Download from [git-scm.com](https://git-scm.com/)

3. **Modern web browser**
   - Chrome, Firefox, Safari, or Edge

### Optional but Recommended

**NVIDIA GPU with CUDA support** for faster processing:
- Check GPU compatibility: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
- Install NVIDIA drivers: [NVIDIA Driver Downloads](https://www.nvidia.com/drivers/)

## Environment Setup

### Step 1: Create Project Directory

Choose a deployment location and create the project directory:

```bash
# Windows
mkdir C:\fluid-analysis
cd C:\fluid-analysis

# Linux/macOS
mkdir ~/fluid-analysis
cd ~/fluid-analysis
```

### Step 2: Clone or Extract Application

**Option A: Using Git**
```bash
git clone <repository-url> .
```

**Option B: Manual extraction**
1. Download the project archive
2. Extract to your project directory
3. Ensure the following structure exists:
```
fluid-analysis/
├── src/
│   ├── inference_app.py
│   ├── refine_net.py
│   └── Example.jpg
├── config/
│   └── config.json
├── models/
│   └── model002.pt
├── requirements/
│   └── requirements.txt
└── README.md
```

### Step 3: Verify File Structure

```bash
# List files to verify structure
# Windows
dir /s

# Linux/macOS
find . -type f -name "*.py" -o -name "*.json" -o -name "*.pt" -o -name "*.txt"
```

Expected output should include:
- `src/inference_app.py`
- `src/refine_net.py`
- `config/config.json`
- `models/model002.pt`
- `requirements/requirements.txt`

## Installation Steps

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv fluid-analysis-env

# Activate virtual environment
# Windows
fluid-analysis-env\Scripts\activate

# Linux/macOS
source fluid-analysis-env/bin/activate
```

**Verification**: Your command prompt should now show `(fluid-analysis-env)` at the beginning.

### Step 2: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
pip install -r requirements/requirements.txt
```

**Expected installation time**: 2-5 minutes depending on internet speed.

### Step 4: Verify Installation

```bash
# Test key imports
python -c "import streamlit, torch, cv2, numpy as np; print('All dependencies installed successfully')"
```

**Success indicator**: Should print "All dependencies installed successfully" without errors.

## Configuration

### Application Configuration

The application uses [`config/config.json`](config/config.json) for model settings. Default configuration should work for most deployments.

**Verify configuration file**:
```bash
# Windows
type config\config.json

# Linux/macOS
cat config/config.json
```

### Environment Variables (Optional)

For production deployments, you may want to set:

```bash
# Windows
set STREAMLIT_SERVER_PORT=8501
set STREAMLIT_SERVER_ADDRESS=localhost

# Linux/macOS
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
```

## Running the Application

### Step 1: Start the Application

From the project root directory:

```bash
streamlit run src/inference_app.py
```

**Expected output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.xxx:8501
```

### Step 2: Access the Interface

1. **Automatic**: Browser should open automatically
2. **Manual**: Navigate to `http://localhost:8501` in your browser

### Step 3: Verify Model Loading

Look for these success indicators in the interface:
- Green "Model loaded successfully" message
- Model selection dropdown populated
- No error messages in the sidebar

## Testing the Deployment

### Basic Functionality Test

1. **Upload Test Image**
   - Use the provided `src/Example.jpg` file
   - Click "Browse files" and select the example image
   - Verify image displays correctly

2. **Model Inference Test**
   - Click "Run Analysis" button
   - Wait for processing to complete
   - Verify results display:
     - Colored overlays (blue for fluid, red for slag)
     - Quality assessment text
     - Confidence scores

3. **ROI Configuration Test**
   - Adjust ROI sliders in the sidebar
   - Verify ROI rectangle updates on the image
   - Re-run analysis with different ROI settings

### Expected Results from Example Image

The test should produce:
- **Quality Classification**: Likely "Clear" or "Steam"
- **Visual Overlays**: Blue regions where fluid is detected
- **Confidence Scores**: Numerical values between 0 and 1
- **Processing Time**: Typically 1-10 seconds depending on hardware

### Performance Verification

Monitor system resources during testing:

**Windows Task Manager**:
- CPU usage should be moderate (30-70% during processing)
- Memory usage should be under 4GB for single image processing

**Linux/macOS htop**:
```bash
htop
```

## Production Considerations

### Scaling for Production

1. **Batch Processing**
   - For multiple files, process sequentially to avoid memory issues
   - Consider breaking large videos into smaller segments

2. **Resource Management**
   - Monitor memory usage regularly
   - Restart application if memory usage exceeds 6GB
   - Consider upgrading to 16GB+ RAM for heavy workloads

3. **Network Deployment**
   - For remote access, modify server address:
   ```bash
   streamlit run src/inference_app.py --server.address 0.0.0.0
   ```
   - **Security Warning**: Only do this on trusted networks

### GPU Acceleration Setup

If you have an NVIDIA GPU:

1. **Install CUDA Toolkit** (if not already installed)
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

2. **Install PyTorch with CUDA**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU Usage**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Data Management

1. **Input Data Organization**
   ```
   data/
   ├── input/
   │   ├── videos/
   │   └── images/
   └── output/
       ├── results/
       └── reports/
   ```

2. **Output Data Handling**
   - CSV exports go to browser downloads by default
   - Consider setting up automated data collection
   - Implement data archival strategy for long-term operations

## Monitoring and Maintenance

### Health Checks

Create a simple health check script:

```python
# health_check.py
import requests
import subprocess
import time

def check_application_health():
    try:
        response = requests.get('http://localhost:8501', timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

if __name__ == "__main__":
    if check_application_health():
        print("✅ Application is healthy")
    else:
        print("❌ Application is not responding")
```

Run periodically:
```bash
python health_check.py
```

### Log Monitoring

1. **Application Logs**
   - Check terminal output where streamlit is running
   - Look for memory warnings or processing errors

2. **System Resource Monitoring**
   ```bash
   # Monitor CPU and memory usage
   # Windows
   wmic cpu get loadpercentage /value
   wmic computersystem get TotalPhysicalMemory /value
   
   # Linux/macOS
   top -p $(pgrep -f streamlit)
   ```

### Regular Maintenance

1. **Weekly Tasks**
   - Check disk space usage
   - Archive old output files
   - Verify model file integrity

2. **Monthly Tasks**
   - Update dependencies (if needed): `pip install -r requirements/requirements.txt --upgrade`
   - Review processing performance metrics
   - Clean temporary files

3. **Model File Verification**
   ```bash
   # Check model file exists and has correct size
   # Windows
   dir models\model002.pt
   
   # Linux/macOS
   ls -lh models/model002.pt
   ```
   Expected size: ~100-200MB

## Troubleshooting

### Installation Issues

#### Problem: "pip install" fails with permission errors

**Windows Solution**:
```bash
pip install --user -r requirements/requirements.txt
```

**Linux/macOS Solution**:
```bash
sudo pip install -r requirements/requirements.txt
# OR (preferred)
pip install --user -r requirements/requirements.txt
```

#### Problem: "python command not found"

**Solution**: Add Python to system PATH
- Windows: Reinstall Python with "Add to PATH" option
- Linux/macOS: Use package manager or update `.bashrc`

### Runtime Issues

#### Problem: "Address already in use" error

**Solution**: Kill existing processes
```bash
# Windows
netstat -ano | findstr 8501
taskkill /PID <process_id> /F

# Linux/macOS
lsof -ti:8501 | xargs kill -9
```

#### Problem: Out of memory errors

**Solution**: Reduce batch size in config
1. Edit `config/config.json`
2. Change `"batch_size"` from 32 to 16 or 8
3. Restart application

#### Problem: Model loading fails

**Symptoms**:
```
RuntimeError: Error(s) in loading state_dict for EnhancedRefineNet
```

**Solutions**:
1. Verify model file integrity:
   ```bash
   # Check file size (should be ~100-200MB)
   ls -lh models/model002.pt
   ```

2. Re-download model file if corrupted

3. Check available memory:
   ```bash
   # Should have at least 4GB available
   free -m  # Linux/macOS
   ```

### Performance Issues

#### Problem: Very slow processing

**Diagnostics**:
```bash
# Check if GPU is being used
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Solutions**:
1. **Enable GPU** (if available): Install CUDA-compatible PyTorch
2. **Reduce image size**: Use smaller ROI regions
3. **Close background apps**: Free up CPU and memory
4. **Use faster storage**: Move data to SSD if using HDD

#### Problem: Inconsistent results

**Diagnostics**:
- Check quality indicators (red = unreliable)
- Verify consistent lighting in input videos
- Ensure ROI is properly positioned

**Solutions**:
1. **Filter by quality**: Only use "Clear" quality results
2. **Improve lighting**: Use consistent illumination
3. **Stabilize camera**: Reduce camera movement
4. **Clean lens**: Remove dust or condensation

### Application-Specific Issues

#### Problem: No fluid detection

**Troubleshooting steps**:
1. Verify ROI covers fluid area
2. Check image contrast and lighting
3. Try different threshold values
4. Ensure model is appropriate for your use case

#### Problem: Web interface doesn't load

**Check browser console**:
- Open Developer Tools (F12)
- Look for JavaScript errors
- Try different browser
- Clear browser cache

#### Problem: File upload fails

**Solutions**:
1. Check file format (supports .mp4, .avi, .mov, .jpg, .png)
2. Verify file size (recommended < 100MB per file)
3. Check available disk space
4. Try smaller file first

### Getting Help

When reporting issues, include:
1. **System information**: OS, Python version, available RAM
2. **Error messages**: Full error text from terminal
3. **Steps to reproduce**: What exactly you were doing
4. **File sizes**: Video/image file sizes being processed
5. **Configuration**: Contents of `config/config.json`

### Emergency Recovery

If application becomes completely unresponsive:

1. **Force stop application**: Ctrl+C in terminal
2. **Clear temporary files**: Restart computer if needed
3. **Reinstall dependencies**:
   ```bash
   pip uninstall -r requirements/requirements.txt -y
   pip install -r requirements/requirements.txt
   ```
4. **Reset configuration**: Copy fresh `config/config.json`

---

**Deployment Complete!** Your Fluid Analysis Application should now be running successfully. For ongoing support, refer to the main [README.md](README.md) for usage instructions.