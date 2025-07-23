"""
Fluid and Slag Detection Inference Application

This Streamlit application provides a user-friendly interface for analyzing
industrial video footage to detect and measure fluid and slag levels while
assessing image quality.

Key Features for Data Engineers:
- Video Processing: Frame-by-frame analysis of industrial footage
- ROI Analysis: Focus analysis on specific rectangular regions of interest
- Real-time Visualization: Live plotting of fluid level measurements
- Quality Assessment: Automatic classification of image clarity conditions
- Data Export: CSV download of measurement data for further analysis

Technical Concepts:
- Segmentation: Pixel-level classification to identify materials (fluid/slag)
- ROI (Region of Interest): Specific area within image where analysis is focused
- Quality Classification: Determines if image conditions affect measurement reliability
- Baseline Calculation: Reference level used for relative measurements
- Mass Distribution: Analysis of material concentration in different image regions

Author: Industrial AI Team
Purpose: Production monitoring and quality control
"""

import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import albumentations as A
import plotly.graph_objects as go
from refine_net import EnhancedRefineNet, QUALITY_READABILITY

# Configuration constants for image quality classification
QUALITY_LABELS = {
    0: 'Clear',    # High quality, reliable measurements
    1: 'Steam',    # Steam present, reduced visibility
    2: 'Fuzzy'     # Poor quality, measurements less reliable
}

# Color coding for quality visualization (BGR format for OpenCV)
QUALITY_COLORS = {
    'Clear': (0, 255, 0),    # Green - good quality
    'Steam': (0, 165, 255),  # Orange - moderate quality
    'Fuzzy': (255, 0, 0)     # Red - poor quality
}

class InferenceModel:
    """
    Neural Network Model Wrapper for Fluid and Slag Detection
    
    This class handles loading, initializing, and running inference with the
    trained RefineNet model for fluid and slag segmentation and quality classification.
    
    Key Responsibilities:
    1. Load pre-trained model weights from checkpoint files
    2. Handle device placement (CPU vs GPU) for optimal performance
    3. Preprocess input images to match training format
    4. Run forward pass and return structured predictions
    
    For Data Engineers:
    - Model Checkpoint: Saved state of trained neural network weights
    - Device Placement: Where computation occurs (CPU for compatibility, GPU for speed)
    - Preprocessing: Standardizing input format (resize, normalize) to match training
    - Inference: Running the model on new data to generate predictions
    """
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize the inference model with pre-trained weights
        
        Args:
            model_path (str): Path to the saved model checkpoint (.pt file)
            config_path (str): Path to the configuration file (.json)
        """
        # Load model configuration settings
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Determine compute device (GPU if available, otherwise CPU)
        self.computation_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the neural network architecture
        self.segmentation_model = EnhancedRefineNet(self.config)
        
        # Load pre-trained weights with comprehensive error handling
        try:
            # Load checkpoint file (contains trained weights)
            model_checkpoint = torch.load(model_path, map_location=self.computation_device)
            
            # Extract state dictionary (actual weights) from checkpoint
            if 'model_state_dict' in model_checkpoint:
                model_weights = model_checkpoint['model_state_dict']
            else:
                # Assume checkpoint is the state dict directly
                model_weights = model_checkpoint
            
            # Clean up weight names (remove 'module.' prefix from distributed training)
            cleaned_weights = {}
            for weight_name, weight_tensor in model_weights.items():
                if weight_name.startswith('module.'):
                    # Remove 'module.' prefix added during multi-GPU training
                    weight_name = weight_name[7:]
                cleaned_weights[weight_name] = weight_tensor
            
            # Handle potential architecture mismatches in attention modules
            for weight_name in list(cleaned_weights.keys()):
                if 'channel_gate.1.weight' in weight_name:
                    # Get expected size from current model architecture
                    expected_size = self.segmentation_model.state_dict()[weight_name].shape[0]
                    actual_size = cleaned_weights[weight_name].shape[0]
                    
                    # Resize weight tensor if dimensions don't match
                    if actual_size != expected_size:
                        st.info(f"Adjusting {weight_name} from {actual_size} to {expected_size} channels")
                        cleaned_weights[weight_name] = torch.nn.functional.interpolate(
                            cleaned_weights[weight_name].unsqueeze(0).unsqueeze(0),
                            size=(expected_size,),
                            mode='linear',
                            align_corners=False
                        ).squeeze()
            
            # Load weights into model (strict=False allows for minor mismatches)
            missing_keys, unexpected_keys = self.segmentation_model.load_state_dict(
                cleaned_weights, strict=False
            )
            
            # Report any loading issues
            if missing_keys:
                st.warning(f"Some model weights were not found in checkpoint: {missing_keys}")
            if unexpected_keys:
                st.warning(f"Checkpoint contains extra weights not used by model: {unexpected_keys}")
                
        except Exception as loading_error:
            st.error(f"Failed to load model weights: {str(loading_error)}")
            raise
        
        # Move model to appropriate device and set to evaluation mode
        self.segmentation_model.to(self.computation_device)
        self.segmentation_model.eval()  # Disable dropout, batch norm updates
        
        # Setup image preprocessing pipeline
        # This ensures input images match the format used during training
        self.image_preprocessor = A.Compose([
            A.Resize(
                height=self.config['data_pipeline']['image_size'][0],
                width=self.config['data_pipeline']['image_size'][1]
            )
        ])
        
        st.success(f"Model loaded successfully on {self.computation_device}")
    
    @torch.no_grad()
    def predict(self, input_image: np.ndarray) -> dict:
        """
        Run inference on a single image to detect fluid, slag, and assess quality
        
        This method processes an input image through the neural network to produce:
        1. Fluid segmentation mask (binary mask showing fluid locations)
        2. Slag segmentation mask (binary mask showing slag locations)
        3. Image quality classification and reliability score
        
        Process Flow:
        1. Preprocessing: Resize and normalize image for model input
        2. Neural Network Inference: Run forward pass through model
        3. Postprocessing: Convert outputs to usable format and resize to original dimensions
        
        Args:
            input_image (np.ndarray): Input image as numpy array (H, W, 3) in RGB format
            
        Returns:
            dict: Structured predictions containing:
                - 'fluid_mask': Binary mask for fluid detection (0-1 values)
                - 'slag_mask': Binary mask for slag detection (0-1 values)
                - 'quality': Dict with quality classification results
        """
        # === PREPROCESSING STAGE ===
        # Resize image to match training input size and convert format
        preprocessed_data = self.image_preprocessor(image=input_image)
        resized_image = preprocessed_data['image']
        
        # Convert to PyTorch tensor format: [channels, height, width]
        image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        # Add batch dimension: [1, channels, height, width]
        batched_tensor = image_tensor.unsqueeze(0).to(self.computation_device)
        
        # === NEURAL NETWORK INFERENCE ===
        # Run the trained model on preprocessed image
        model_predictions = self.segmentation_model(batched_tensor)
        
        # === POSTPROCESSING STAGE ===
        # Extract segmentation masks and move to CPU for processing
        raw_fluid_mask = model_predictions['fluid_mask'].squeeze().cpu().numpy()
        raw_slag_mask = model_predictions['slag_mask'].squeeze().cpu().numpy()
        
        # Process quality classification results
        quality_logits = model_predictions['quality'].squeeze().cpu()
        quality_probabilities = torch.softmax(quality_logits, dim=0).numpy()
        predicted_quality_class = int(torch.argmax(quality_logits).item())
        
        # Get measurement reliability score based on image quality
        measurement_reliability = QUALITY_READABILITY[predicted_quality_class]
        
        # Resize masks back to original image dimensions
        original_height, original_width = input_image.shape[:2]
        final_fluid_mask = cv2.resize(raw_fluid_mask, (original_width, original_height))
        final_slag_mask = cv2.resize(raw_slag_mask, (original_width, original_height))
        
        # Structure output for downstream processing
        prediction_results = {
            'fluid_mask': final_fluid_mask,
            'slag_mask': final_slag_mask,
            'quality': {
                'label': QUALITY_LABELS[predicted_quality_class],
                'probabilities': {
                    QUALITY_LABELS[class_idx]: float(prob)
                    for class_idx, prob in enumerate(quality_probabilities)
                },
                'readability': measurement_reliability
            }
        }
        
        return prediction_results

def extract_roi_from_frame(video_frame: np.ndarray, roi_coordinates: tuple) -> np.ndarray:
    """
    Extract Region of Interest (ROI) from video frame
    
    ROI extraction focuses analysis on a specific rectangular area within the frame,
    which is useful for industrial applications where the area of interest is known
    (e.g., furnace interior, processing vessel, etc.)
    
    Args:
        video_frame (np.ndarray): Full video frame as numpy array
        roi_coordinates (tuple): ROI coordinates as (x, y, width, height)
    
    Returns:
        np.ndarray: Cropped image containing only the ROI area
    """
    x_start, y_start, roi_width, roi_height = roi_coordinates
    return video_frame[y_start:y_start+roi_height, x_start:x_start+roi_width]


def create_visualization_overlay(original_image: np.ndarray, model_predictions: dict,
                                transparency: float = 0.5) -> np.ndarray:
    """
    Create visualization overlay combining segmentation masks and quality information
    
    This function creates a visual representation of the model's predictions by:
    1. Overlaying colored masks for fluid (blue) and slag (red) detection
    2. Adding quality assessment information with color-coded text
    3. Including confidence scores for each quality class
    
    Color Coding:
    - Blue overlay: Fluid detection areas
    - Red overlay: Slag detection areas
    - Green text: Clear/good quality conditions
    - Orange text: Steam present, moderate quality
    - Red text: Fuzzy/poor quality conditions
    
    Args:
        original_image (np.ndarray): Original input image (RGB format)
        model_predictions (dict): Predictions from neural network
        transparency (float): Overlay transparency (0=invisible, 1=opaque)
    
    Returns:
        np.ndarray: Image with visual overlays and annotations
    """
    visualization_image = original_image.copy()
    image_height, image_width = visualization_image.shape[:2]
    
    # Create fluid detection overlay (blue channel)
    fluid_overlay = np.zeros_like(visualization_image)
    fluid_overlay[..., 0] = model_predictions['fluid_mask'] * 255  # Blue channel
    visualization_image = cv2.addWeighted(visualization_image, 1, fluid_overlay, transparency, 0)
    
    # Create slag detection overlay (red channel)
    slag_overlay = np.zeros_like(visualization_image)
    slag_overlay[..., 2] = model_predictions['slag_mask'] * 255  # Red channel
    visualization_image = cv2.addWeighted(visualization_image, 1, slag_overlay, transparency, 0)
    
    # Extract quality assessment information
    quality_label = model_predictions['quality']['label']
    quality_probabilities = model_predictions['quality']['probabilities']
    measurement_reliability = model_predictions['quality']['readability']
    
    # Select text color based on quality assessment
    text_color = QUALITY_COLORS[quality_label]
    
    # Add main quality assessment text
    quality_text = f"Quality: {quality_label} (Reliability: {measurement_reliability:.2f})"
    cv2.putText(visualization_image, quality_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    # Add detailed probability breakdown
    text_y_position = 60
    for quality_class, probability in quality_probabilities.items():
        probability_text = f"{quality_class}: {probability:.2f}"
        cv2.putText(visualization_image, probability_text, (10, text_y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, QUALITY_COLORS[quality_class], 1)
        text_y_position += 20
    
    return visualization_image

def calculate_fluid_level_metrics(fluid_segmentation_mask: np.ndarray) -> dict:
    """
    Calculate fluid level measurements from segmentation mask
    
    This function analyzes a binary segmentation mask to determine fluid level
    measurements. It uses statistical filtering to remove noise and spurious
    detections that could affect measurement accuracy.
    
    Key Concepts for Data Engineers:
    - Binary Mask: Image where 1.0 = fluid present, 0.0 = no fluid
    - Y-coordinate: Vertical position (0 = top of image, higher values = lower)
    - Height Measurement: Topmost fluid position (lowest y-coordinate)
    - Mass Distribution: Amount of detected fluid material per horizontal row
    - Outlier Filtering: Removing spurious detections that don't represent real fluid
    
    Statistical Filtering Process:
    1. Calculate fluid mass (pixel count) for each horizontal row
    2. Determine mean and standard deviation of non-zero rows
    3. Filter out rows with too little mass (noise) or too much mass (artifacts)
    4. Use remaining rows to determine fluid level
    
    Args:
        fluid_segmentation_mask (np.ndarray): Binary mask from neural network
    
    Returns:
        dict: Measurements containing:
            - 'height': Topmost fluid position in pixels (float)
            - 'mass': Total fluid mass in valid regions (float)
    """
    # Handle empty mask case
    if not fluid_segmentation_mask.any():
        return {'height': 0.0, 'mass': 0}
    
    # Find all pixel coordinates where fluid is detected (above 50% confidence)
    fluid_y_coords, fluid_x_coords = np.where(fluid_segmentation_mask > 0.5)
    if len(fluid_y_coords) == 0:
        return {'height': 0.0, 'mass': 0}
    
    # Calculate fluid mass (pixel count) for each horizontal row
    # This helps identify which rows contain significant fluid vs. noise
    pixels_per_row = np.sum(fluid_segmentation_mask > 0.5, axis=1)
    
    # Statistical analysis to filter out noise and artifacts
    non_zero_row_masses = pixels_per_row[pixels_per_row > 0]
    if len(non_zero_row_masses) == 0:
        return {'height': 0.0, 'mass': 0}
    
    # Calculate statistics for outlier detection
    mean_pixels_per_row = np.mean(non_zero_row_masses)
    std_pixels_per_row = np.std(non_zero_row_masses)
    
    # Define filtering thresholds
    # Minimum threshold: Remove rows with very few pixels (likely noise)
    minimum_mass_threshold = mean_pixels_per_row * 0.5
    # Maximum threshold: Remove rows with excessive pixels (likely artifacts)
    maximum_mass_threshold = mean_pixels_per_row + 2 * std_pixels_per_row
    
    # Identify rows with significant fluid content (not noise or artifacts)
    valid_rows_mask = ((pixels_per_row > minimum_mass_threshold) &
                       (pixels_per_row < maximum_mass_threshold))
    
    # Filter fluid coordinates to only include valid rows
    valid_y_coordinates = fluid_y_coords[np.isin(fluid_y_coords, np.where(valid_rows_mask)[0])]
    
    if len(valid_y_coordinates) == 0:
        return {'height': 0.0, 'mass': 0}
    
    # Determine fluid level: topmost position (minimum y-coordinate)
    # Note: y=0 is top of image, so min(y) = highest physical level
    fluid_level_height = np.min(valid_y_coordinates)
    
    # Calculate total fluid mass from valid detections
    total_fluid_mass = len(valid_y_coordinates)
    
    return {
        'height': float(fluid_level_height),
        'mass': float(total_fluid_mass)
    }

def calculate_relative_mass(mask: np.ndarray, baseline: float) -> dict:
    """Calculate mass distribution relative to baseline"""
    if baseline is None or not mask.any():
        return {'relative_mass': 0.0, 'top_mass_ratio': 0.0}
    
    height = mask.shape[0]
    
    # Calculate mass in each row
    row_masses = np.sum(mask > 0.5, axis=1)  # Sum across rows
    
    # Use higher threshold (30% of mean mass) for more robust detection
    significant_threshold = np.mean(row_masses[row_masses > 0]) * 0.3
    
    # Define top 25% region relative to baseline
    # Note: y-coordinates start from top (0) to bottom (height)
    top_region_start = max(0, int(baseline - 0.25 * (height - baseline)))
    top_region_end = int(baseline)
    
    # Calculate mass in top region and total mass, considering only significant rows
    top_region_mask = mask[top_region_start:top_region_end, :]
    top_row_masses = np.sum(top_region_mask > 0.5, axis=1)
    valid_top_rows = top_row_masses > significant_threshold
    top_mass = np.sum(top_region_mask[valid_top_rows])
    
    # Calculate total mass from significant rows
    valid_rows = row_masses > significant_threshold
    total_mass = np.sum(mask[valid_rows])
    
    # Calculate relative mass (as ratio of total mass)
    if total_mass > 0:
        top_mass_ratio = float(top_mass / total_mass)
    else:
        top_mass_ratio = 0.0
    
    return {
        'top_mass_ratio': top_mass_ratio,
        'top_region': (top_region_start, top_region_end)
    }

def calculate_baseline(levels: list) -> float:
    """Calculate baseline from first 30 frames"""
    if not levels:
        return 0.0
    # Use the highest points (minimum y-coordinates) from first 30 frames
    baseline_levels = levels[:30]
    # Take median to be robust against outliers
    return np.median(baseline_levels)

def get_available_models():
    """Get list of available .pt model files"""
    models_dir = Path("models")
    return [f.name for f in models_dir.glob("*.pt")]

def main():
    st.title("Fluid and Slag Detection with Quality Classification")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Model selection - dynamically load available models
    available_models = get_available_models()
    if not available_models:
        st.error("No model files found in the models directory!")
        return
        
    model_path = st.sidebar.selectbox(
        "Select model",
        available_models,
        index=0
    )
    model_path = str(Path("models") / model_path)
    
    # Config file selection
    config_path = str(Path("config") / "config.json")
    if not Path(config_path).exists():
        st.error(f"Config file not found at {config_path}")
        return
    
    # ROI settings
    roi_x = st.sidebar.number_input("ROI X", value=400, min_value=0)
    roi_y = st.sidebar.number_input("ROI Y", value=50, min_value=0)
    roi_w = st.sidebar.number_input("ROI Width", value=1200, min_value=1)
    roi_h = st.sidebar.number_input("ROI Height", value=950, min_value=1)
    
    # Visualization settings
    overlay_alpha = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.5)
    
    # Initialize model with detailed error handling
    try:
        ai_model = InferenceModel(model_path, config_path)
    except FileNotFoundError as file_error:
        st.error(f"❌ **Model file not found**: {str(file_error)}")
        st.info("💡 **Solution**: Please ensure the model file exists in the models directory")
        return
    except json.JSONDecodeError as json_error:
        st.error(f"❌ **Configuration file error**: Invalid JSON format in config file")
        st.info("💡 **Solution**: Please check that config.json has valid JSON syntax")
        return
    except torch.nn.modules.module.ModuleAttributeError as model_error:
        st.error(f"❌ **Model architecture mismatch**: {str(model_error)}")
        st.info("💡 **Solution**: The model file may be incompatible. Try using a different model checkpoint.")
        return
    except Exception as general_error:
        st.error(f"❌ **Unexpected error during model initialization**: {str(general_error)}")
        st.info("💡 **Troubleshooting**: Please check that:")
        st.info("   • Model file (.pt) exists and is not corrupted")
        st.info("   • Config file (config.json) exists and has valid syntax")
        st.info("   • You have sufficient system memory available")
        return
    
    # Video input
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    
    if video_file is not None:
        # Save uploaded file temporarily with error handling
        temp_video_path = Path("temp_video.mp4")
        try:
            with open(temp_video_path, 'wb') as file_handle:
                file_handle.write(video_file.read())
        except IOError as io_error:
            st.error(f"❌ **Failed to save video file**: {str(io_error)}")
            st.info("💡 **Solution**: Please check available disk space and try again")
            return
        
        # Open video with error handling
        video_capture = cv2.VideoCapture(str(temp_video_path))
        
        # Verify video opened successfully
        if not video_capture.isOpened():
            st.error("❌ **Failed to open video file**: Video format may not be supported")
            st.info("💡 **Solution**: Please try with a different video format (MP4 or AVI)")
            temp_video_path.unlink(missing_ok=True)  # Clean up temporary file
            return
        
        # Extract video properties with validation
        video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video properties
        if video_fps <= 0 or total_frame_count <= 0:
            st.error("❌ **Invalid video file**: Unable to read video properties")
            st.info("💡 **Solution**: Please try with a different video file")
            video_capture.release()
            temp_video_path.unlink(missing_ok=True)
            return
        
        st.info(f"📹 **Video loaded successfully**: {total_frame_count} frames at {video_fps} FPS")
        
        # Progress bar and metrics
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        quality_metrics = {label: {'count': 0, 'readability': []} for label in QUALITY_LABELS.values()}
        
        # Fluid level tracking
        fluid_levels = []
        baseline = None
        relative_levels = []
        processed_frames = 0
        predictions_history = []  # Store predictions for each frame
        
        # Create plots
        fig_placeholder = st.empty()
        
        # Process video
        frame_idx = 0
        start_time = time.time()
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Process 1 frame per second
            if frame_idx % video_fps == 0:
                # Extract ROI
                roi = extract_roi_from_frame(frame, (roi_x, roi_y, roi_w, roi_h))
                
                # Convert BGR to RGB
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                
                # Get predictions
                predictions = ai_model.predict(roi_rgb)
                predictions_history.append(predictions)  # Store predictions
                
                # Calculate fluid level metrics
                level_metrics = calculate_fluid_level_metrics(predictions['fluid_mask'])
                current_level = level_metrics['height']
                fluid_levels.append(current_level)
                processed_frames += 1
                
                # Calculate baseline from first 30 processed frames
                if processed_frames == 30:
                    baseline = calculate_baseline(fluid_levels)
                    # Convert previous levels to relative in pixels (positive when higher)
                    relative_levels.extend([level - baseline for level in fluid_levels])
                elif processed_frames > 30:
                    # Positive when higher than baseline
                    relative_level = baseline - current_level
                    relative_levels.append(relative_level)
                
                # Calculate mass distribution if baseline exists
                mass_metrics = None
                if baseline is not None:
                    mass_metrics = calculate_relative_mass(predictions['fluid_mask'], baseline)
                
                # Update quality metrics
                quality = predictions['quality']['label']
                readability = predictions['quality']['readability']
                quality_metrics[quality]['count'] += 1
                quality_metrics[quality]['readability'].append(readability)
                
                # Create visualization
                vis_frame = create_visualization_overlay(roi_rgb, predictions, overlay_alpha)
                
                # Add fluid level indicator if baseline exists
                if baseline is not None:
                    # Draw baseline
                    cv2.line(vis_frame, (0, int(baseline)), (roi_w, int(baseline)), 
                            (0, 255, 0), 2)  # Green line
                    # Draw current level
                    cv2.line(vis_frame, (0, int(current_level)), (roi_w, int(current_level)), 
                            (255, 0, 0), 2)  # Red line
                    
                    # Draw top 25% region
                    if mass_metrics:
                        top_start, top_end = mass_metrics['top_region']
                        # Draw region boundaries
                        cv2.line(vis_frame, (0, top_start), (roi_w, top_start),
                                (255, 255, 0), 1)  # Yellow line
                        cv2.putText(vis_frame, f"Top Mass Ratio: {mass_metrics['top_mass_ratio']:.1%}", 
                                  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Add relative level text in pixels (positive when higher)
                    relative_level = baseline - current_level
                    cv2.putText(vis_frame, f"Relative Level: {relative_level:.1f} px", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display results
                frame_placeholder.image(vis_frame, channels="RGB", use_container_width=True)

                # Update fluid level plot
                if len(relative_levels) > 0:
                    fig = go.Figure()
                    
                    # Create separate traces for each quality type
                    quality_colors = {
                        'Clear': 'blue',
                        'Steam': 'orange',
                        'Fuzzy': 'red'
                    }
                    
                    # Store data points for each quality type
                    quality_data = {
                        'Clear': {'x': [], 'y': [], 'quality': []},
                        'Steam': {'x': [], 'y': [], 'quality': []},
                        'Fuzzy': {'x': [], 'y': [], 'quality': []}
                    }
                    
                    # Group data points by quality
                    for i, (level, pred) in enumerate(zip(relative_levels, predictions_history)):
                        quality = pred['quality']['label']
                        quality_data[quality]['x'].append(i)
                        quality_data[quality]['y'].append(level)
                        quality_data[quality]['quality'].append(pred['quality']['readability'])
                    
                    # Add trace for each quality type
                    for q, color in quality_colors.items():
                        if quality_data[q]['x']:  # Only add trace if there are points
                            # Create hover text with quality info
                            hover_text = [f"Frame: {x}<br>Level: {y:.1f}px<br>Quality: {q}<br>Readability: {r:.2f}" 
                                        for x, y, r in zip(quality_data[q]['x'], 
                                                         quality_data[q]['y'],
                                                         quality_data[q]['quality'])]
                            
                            fig.add_trace(go.Scatter(
                                x=quality_data[q]['x'],
                                y=quality_data[q]['y'],
                                mode='lines+markers',
                                name=f'Quality: {q}',
                                line=dict(color=color),
                                marker=dict(color=color),
                                text=hover_text,
                                hoverinfo='text'
                            ))
                    
                    fig.update_layout(
                        title='Relative Fluid Level Over Time (by Image Quality)',
                        yaxis_title='Relative Level (pixels)',
                        xaxis_title='Processed Frame',
                        height=400,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        ),
                        hovermode='closest'
                    )
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="green", 
                                annotation_text="Baseline")
                    
                    fig_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Update stats
                elapsed_time = time.time() - start_time
                stats = f"""
                Frame: {frame_idx}/{total_frame_count}
                FPS: {frame_idx/elapsed_time:.1f}
                Time: {elapsed_time:.1f}s
                
                Quality Distribution:
                """
                for label, metrics in quality_metrics.items():
                    avg_readability = np.mean(metrics['readability']) if metrics['readability'] else 0
                    stats += f"\n{label}: {metrics['count']} frames (avg readability: {avg_readability:.2f})"
                
                if baseline is not None:
                    stats += f"\n\nCurrent Relative Level: {relative_levels[-1]:.1f} px"
                
                stats_placeholder.text(stats)
            
            # Update progress
            progress = frame_idx / total_frame_count
            progress_bar.progress(progress)
            
            frame_idx += 1
        
        # Cleanup
        video_capture.release()
        temp_video_path.unlink()
        
        st.success("Processing complete!")
        
        # Save fluid level data
        if len(relative_levels) > 0:
            df = pd.DataFrame({
                'Frame': range(len(relative_levels)),
                'Relative_Level': relative_levels
            })
            csv_path = 'fluid_levels.csv'
            df.to_csv(csv_path, index=False)
            st.download_button(
                label="Download Fluid Level Data",
                data=df.to_csv(index=False),
                file_name='fluid_levels.csv',
                mime='text/csv'
            )
        
        # Final quality distribution plot
        if sum(m['count'] for m in quality_metrics.values()) > 0:
            st.subheader("Overall Quality Distribution")
            
            # Prepare data for plotting
            plot_data = []
            for label, metrics in quality_metrics.items():
                plot_data.append({
                    'Quality': label,
                    'Count': metrics['count'],
                    'Avg Readability': np.mean(metrics['readability']) if metrics['readability'] else 0
                })
            
            df = pd.DataFrame(plot_data)
            
            # Plot counts
            st.bar_chart(df.set_index('Quality')['Count'])
            
            # Plot average readability
            st.subheader("Average Readability by Quality Type")
            st.bar_chart(df.set_index('Quality')['Avg Readability'])

if __name__ == "__main__":
    main()
