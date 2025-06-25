# CVAT Annotation Pipeline Documentation

## Overview
This pipeline automates the process of generating CVAT-compatible annotations from videos using the trained fluid analysis model. It implements a sophisticated frame selection strategy to identify the most relevant frames for annotation.

## Frame Selection Process

### 1. Initial Video Analysis (1 FPS)
- Processes video at 1 frame per second
- Computes comprehensive histogram features:
  - Hue (32 bins, weight: 0.3)
  - Saturation (32 bins, weight: 0.2)
  - Value/Brightness (32 bins, weight: 0.2)
  - Intensity (32 bins, weight: 0.3)
- Handles 30-45 minute videos (1800-2700 frames at 1 FPS)

### 2. Diverse Frame Selection (200 frames)
- Uses K-means clustering on histogram features
- Selects frames closest to cluster centers
- Ensures representation of different:
  - Lighting conditions
  - Contrast levels
  - Color distributions
  - Scene compositions

### 3. Model Analysis (200 → 100 frames)
Analyzes the 200 diverse frames using:
- Model confidence scores
- Edge complexity metrics
- Quality classification
- Readability scores

### 4. Final Frame Selection (100 frames)
Selects final frames based on:
- Quality distribution balancing
- Difficulty scores combining:
  - Model uncertainty (30%)
  - Readability scores (20%)
  - Mask confidence (20%)
  - Edge complexity (30%)

## Quick Start
```bash
python src/generate_cvat_annotations.py \
    --model_path models/model001.pt \
    --config_path config/config.json \
    --output_dir output/cvat_annotations \
    --video_dir path/to/videos \
    --initial_frames 200 \
    --final_frames 100
```

## Output Structure
```
output_dir/
├── frames/
│   └── video_name/
│       ├── frame_0001.png
│       ├── frame_0002.png
│       └── ...
├── video_name_annotations.xml
├── video_name_metadata.json
└── processing_results.json
```

## Metadata and Analysis

### Frame Selection Metadata
The pipeline generates detailed metadata for each video:
```json
{
    "video_name": "example_video",
    "total_frames_analyzed": 1800,
    "diverse_frames_selected": 200,
    "final_frames_selected": 100,
    "frame_indices": [...],
    "quality_distribution": {
        "Clear": 45,
        "Steam": 30,
        "Fuzzy": 25
    },
    "average_difficulty": 0.65
}
```

### Quality Metrics
Frames are scored based on:
1. **Model Confidence**
   - Quality classification probability
   - Mask prediction confidence

2. **Visual Complexity**
   - Edge detection analysis
   - Histogram distribution
   - Contrast variations

3. **Quality Distribution**
   - Balanced selection across categories
   - Emphasis on challenging cases

## Best Practices

### 1. Video Preparation
- Ensure consistent frame rate (25 FPS recommended)
- Maintain stable video quality
- Avoid extreme compression artifacts

### 2. Frame Selection
- Default settings (200 → 100) work well for 30-45 min videos
- Adjust initial_frames for longer/shorter videos
- Consider video complexity when setting final_frames

### 3. Quality Review
- Focus on frames with high difficulty scores
- Verify quality classification accuracy
- Check edge cases and transitions

## Troubleshooting

### Common Issues
1. **Memory Management**
   - 1 FPS analysis reduces memory load
   - Batch processing of predictions
   - Efficient histogram computation

2. **Processing Time**
   - K-means clustering optimization
   - Parallel histogram computation
   - Efficient frame extraction

3. **Frame Selection**
   - Adjust clustering parameters
   - Fine-tune difficulty weights
   - Balance quality distribution

## Future Improvements
1. **Advanced Analysis**
   - Motion detection integration
   - Scene change detection
   - Temporal consistency checks

2. **Optimization**
   - GPU acceleration for histograms
   - Parallel frame processing
   - Memory-efficient clustering

3. **Quality Control**
   - Automated validation checks
   - Cross-frame consistency
   - Annotation quality metrics
