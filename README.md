# Fluid Analysis Application - Deployment Guide

## Overview

This application provides **automated fluid and slag level detection** in industrial video footage using computer vision and machine learning. It's designed for data engineers who need to deploy and operate the system for production monitoring and quality control.

### What This Application Does

The system analyzes industrial video footage to:
- **Detect fluid levels** in real-time with pixel-level precision
- **Identify slag presence** for quality assessment
- **Classify image quality** (Clear, Steam, Fuzzy) to assess measurement reliability
- **Generate measurement reports** with exportable data for downstream analysis
- **Provide visual feedback** with color-coded overlays and confidence scores

### Key Benefits for Operations

- **Non-intrusive monitoring**: Uses existing camera feeds
- **Real-time analysis**: Processes video frames as they come in
- **Quality awareness**: Automatically flags unreliable measurements
- **Data export**: CSV output for integration with existing data pipelines
- **Visual validation**: Clear overlays to verify system performance

## System Requirements

### Hardware Requirements
- **CPU**: Modern multi-core processor (Intel i5 or AMD equivalent)
- **RAM**: Minimum 8GB, recommended 16GB for smooth operation
- **GPU**: Optional but recommended - NVIDIA GPU with CUDA support for faster processing
- **Storage**: 2GB free space for application and model files
- **Network**: Standard internet connection for initial setup

### Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Web Browser**: Modern browser (Chrome, Firefox, Safari, Edge)

### GPU Acceleration (Optional)
If you have an NVIDIA GPU:
- Install [NVIDIA drivers](https://www.nvidia.com/drivers/)
- The application will automatically use GPU acceleration when available

## Quick Start

### 1. Download and Extract
```bash
git clone <repository-url>
cd Fluid_analysis_project-1
```

### 2. Install Dependencies
```bash
pip install -r requirements/requirements.txt
```

### 3. Launch Application
```bash
streamlit run src/inference_app.py
```

### 4. Open in Browser
- The application will automatically open at `http://localhost:8501`
- If not, manually navigate to this address in your browser

## Application Interface

### Main Components

1. **Model Selection**: Choose from available trained models (located in `models/` folder)
2. **Video Upload**: Upload video files for analysis (.mp4, .avi, .mov formats)
3. **ROI Configuration**: Set Region of Interest to focus analysis on specific areas
4. **Real-time Visualization**: See live analysis results with colored overlays
5. **Data Export**: Download measurement data as CSV files

### Understanding the Output

#### Visual Indicators
- **Blue overlay**: Detected fluid areas
- **Red overlay**: Detected slag areas
- **Text annotations**: Quality assessment and confidence scores

#### Quality Classifications
- **Clear (Green)**: High reliability measurements (Readability: 1.0)
- **Steam (Orange)**: Moderate reliability due to steam presence (Readability: 0.4)
- **Fuzzy (Red)**: Lower reliability due to poor image quality (Readability: 0.7)

#### Measurement Data
- **Height**: Topmost fluid level in pixels (y-coordinate)
- **Mass**: Total detected fluid area in pixels
- **Quality Score**: Confidence in measurement reliability (0-1 scale)

## Configuration

### Model Configuration
The system uses [`config/config.json`](config/config.json) for model settings:
- Image processing parameters
- Model architecture settings
- Performance optimization options

### Region of Interest (ROI)
Set ROI coordinates to focus analysis on specific areas:
- Improves accuracy by excluding irrelevant image areas
- Reduces processing time
- Configurable through the web interface

## Typical Workflow

### For Data Engineers

1. **Setup**: Install dependencies and launch application
2. **Configuration**: Set ROI based on your camera setup
3. **Processing**: Upload video files or connect live camera feed
4. **Monitoring**: Watch real-time analysis and quality indicators
5. **Data Collection**: Export measurement data for downstream processing
6. **Quality Control**: Review flagged poor-quality frames

### Integration with Data Pipelines

The application produces structured data suitable for:
- **Time series databases**: Fluid level measurements over time
- **Quality monitoring systems**: Image quality assessments
- **Alert systems**: Automated notifications based on measurement thresholds
- **Reporting systems**: Statistical analysis of production data

## Output Format

### CSV Export Structure
```csv
timestamp,fluid_height,fluid_mass,quality_label,quality_score,readability
2024-01-15 14:30:00,245.5,1250,Clear,0.95,1.0
2024-01-15 14:30:01,248.2,1275,Clear,0.92,1.0
2024-01-15 14:30:02,250.1,1290,Steam,0.78,0.4
```

### Understanding the Measurements

- **fluid_height**: Vertical position of topmost fluid (pixels from top)
- **fluid_mass**: Total fluid area detected (pixel count)
- **quality_label**: Visual condition classification
- **quality_score**: Confidence in the quality assessment (0-1)
- **readability**: Reliability factor for measurements (0-1)

## Troubleshooting

### Common Issues

#### Application Won't Start
**Problem**: Error when running `streamlit run src/inference_app.py`
**Solutions**:
1. Verify Python version: `python --version` (should be 3.8+)
2. Install missing dependencies: `pip install -r requirements/requirements.txt`
3. Check if ports are available: Try `streamlit run src/inference_app.py --server.port 8502`

#### Model Loading Error
**Problem**: "Failed to load model weights" error message
**Solutions**:
1. Ensure model file exists: Check `models/model002.pt`
2. Verify file permissions: Model file should be readable
3. Check available memory: Close other applications if needed

#### Poor Performance
**Problem**: Slow processing or high CPU usage
**Solutions**:
1. **Use GPU**: Install CUDA if you have an NVIDIA graphics card
2. **Reduce ROI size**: Smaller analysis areas process faster
3. **Close background apps**: Free up system resources
4. **Check video resolution**: Lower resolution videos process faster

#### Inconsistent Results
**Problem**: Measurements vary significantly between similar frames
**Solutions**:
1. **Check image quality indicators**: Red "Fuzzy" quality indicates unreliable data
2. **Adjust ROI**: Ensure ROI captures the relevant area consistently
3. **Review video quality**: Poor lighting or camera issues affect accuracy
4. **Filter by quality**: Only use measurements from "Clear" quality frames

#### No Detections
**Problem**: Application shows no fluid or slag detection
**Solutions**:
1. **Verify ROI placement**: Ensure ROI covers the area with fluid
2. **Check image contrast**: Poor contrast makes detection difficult
3. **Review model compatibility**: Ensure model was trained on similar scenarios
4. **Adjust lighting**: Consistent lighting improves detection accuracy

### Performance Optimization

#### For Better Speed
- Use NVIDIA GPU with CUDA support
- Reduce video resolution before processing
- Process shorter video segments
- Use smaller ROI areas when possible

#### For Better Accuracy
- Ensure consistent lighting conditions
- Use high-quality camera equipment
- Keep camera position stable
- Clean camera lens regularly
- Filter results by quality indicators

## Support and Maintenance

### Log Files
The application generates logs for troubleshooting:
- Check browser console for web interface issues
- Review terminal output for processing errors
- Monitor system resource usage during operation

### Regular Maintenance
- **Update dependencies** periodically: `pip install -r requirements/requirements.txt --upgrade`
- **Monitor model performance** on your specific data
- **Archive processed data** to maintain system performance
- **Clean temporary files** if disk space becomes limited

### When to Contact Support
- Consistent accuracy issues on your specific video content
- Performance problems that persist after optimization
- Integration challenges with existing data systems
- Model retraining requirements for new scenarios

## Security Considerations

- **Input validation**: Only process video files from trusted sources
- **Network access**: Application runs locally by default (port 8501)
- **Data privacy**: Processed video data remains on local system
- **File permissions**: Ensure proper access controls for model files and output data

---

**Need help?** Check the detailed [DEPLOYMENT.md](DEPLOYMENT.md) guide for step-by-step setup instructions and advanced configuration options.