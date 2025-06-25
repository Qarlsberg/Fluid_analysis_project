import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from refine_net import EnhancedRefineNet, QUALITY_READABILITY
import albumentations as A
import plotly.graph_objects as go
from collections import deque

# Quality label mapping
QUALITY_LABELS = {
    0: 'Clear',
    1: 'Steam',
    2: 'Fuzzy'
}

# Quality color mapping
QUALITY_COLORS = {
    'Clear': (0, 255, 0),    # Green
    'Steam': (0, 165, 255),  # Orange
    'Fuzzy': (255, 0, 0)     # Red
}

class InferenceModel:
    def __init__(self, model_path: str, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EnhancedRefineNet(self.config)
        
        # Load trained weights with error handling
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint  # Assume it's the state dict directly
            
            # Remove module prefix if it exists
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # Remove 'module.' prefix
                new_state_dict[k] = v
            
            # Handle channel gate weight size mismatches
            for k in list(new_state_dict.keys()):
                if 'channel_gate.1.weight' in k:
                    # Get target size from model
                    target_size = self.model.state_dict()[k].shape[0]
                    # Resize weight tensor
                    if new_state_dict[k].shape[0] != target_size:
                        print(f"Resizing {k} from {new_state_dict[k].shape} to {target_size}")
                        new_state_dict[k] = F.interpolate(
                            new_state_dict[k].unsqueeze(0).unsqueeze(0),
                            size=(target_size,),
                            mode='linear'
                        ).squeeze()
            
            # Load state dict with strict=False to ignore missing keys
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                st.warning(f"Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                st.warning(f"Unexpected keys in state dict: {unexpected_keys}")
                
        except Exception as e:
            st.error(f"Error loading model weights: {str(e)}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = A.Compose([
            A.Resize(
                self.config['data_pipeline']['image_size'][0],
                self.config['data_pipeline']['image_size'][1]
            )
        ])
        
        st.success(f"Model loaded successfully on {self.device}")
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        # Preprocess
        transformed = self.transform(image=image)
        image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        outputs = self.model(image_tensor)
        
        # Post-process
        fluid_mask = outputs['fluid_mask'].squeeze().cpu().numpy()
        slag_mask = outputs['slag_mask'].squeeze().cpu().numpy()
        
        # Get quality prediction
        quality_logits = outputs['quality'].squeeze().cpu()
        quality_probs = torch.softmax(quality_logits, dim=0).numpy()
        quality_pred = int(torch.argmax(quality_logits).item())
        
        # Get readability score
        readability = QUALITY_READABILITY[quality_pred]
        
        # Resize masks back to original size
        h, w = image.shape[:2]
        fluid_mask = cv2.resize(fluid_mask, (w, h))
        slag_mask = cv2.resize(slag_mask, (w, h))
        
        return {
            'fluid_mask': fluid_mask,
            'slag_mask': slag_mask,
            'quality': {
                'label': QUALITY_LABELS[quality_pred],
                'probabilities': {QUALITY_LABELS[i]: float(p) for i, p in enumerate(quality_probs)},
                'readability': readability
            }
        }

def extract_roi(frame: np.ndarray, roi_coords: tuple) -> np.ndarray:
    """Extract ROI from frame using CVAT-style coordinates"""
    x, y, w, h = roi_coords
    return frame[y:y+h, x:x+w]

def overlay_masks(image: np.ndarray, predictions: dict, alpha: float = 0.5) -> np.ndarray:
    """Overlay prediction masks and quality info on image"""
    overlay = image.copy()
    h, w = image.shape[:2]
    
    # Fluid mask in blue
    fluid_vis = np.zeros_like(image)
    fluid_vis[..., 0] = predictions['fluid_mask'] * 255
    overlay = cv2.addWeighted(overlay, 1, fluid_vis, alpha, 0)
    
    # Slag mask in red
    slag_vis = np.zeros_like(image)
    slag_vis[..., 2] = predictions['slag_mask'] * 255
    overlay = cv2.addWeighted(overlay, 1, slag_vis, alpha, 0)
    
    # Add quality label
    quality = predictions['quality']['label']
    probs = predictions['quality']['probabilities']
    readability = predictions['quality']['readability']
    
    # Choose text color based on quality
    color = QUALITY_COLORS[quality]
    
    # Add quality info
    text = f"Quality: {quality} (Readability: {readability:.2f})"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add probabilities
    y_offset = 60
    for label, prob in probs.items():
        text = f"{label}: {prob:.2f}"
        cv2.putText(overlay, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, QUALITY_COLORS[label], 1)
        y_offset += 20
    
    return overlay

def calculate_fluid_level(mask: np.ndarray) -> dict:
    """Calculate fluid level metrics including height and mass distribution"""
    if not mask.any():  # If mask is empty
        return {'height': 0.0, 'mass': 0}
    
    # Find all y-coordinates where fluid is present
    y_coords, x_coords = np.where(mask > 0.5)
    if len(y_coords) == 0:
        return {'height': 0.0, 'mass': 0}
    
    # Calculate mass in each row to detect outliers
    row_masses = np.sum(mask > 0.5, axis=1)  # Sum across rows
    
    # Calculate statistics for outlier detection
    non_zero_masses = row_masses[row_masses > 0]
    if len(non_zero_masses) == 0:
        return {'height': 0.0, 'mass': 0}
    
    mean_mass = np.mean(non_zero_masses)
    std_mass = np.std(non_zero_masses)
    
    # Use higher threshold (50% of mean mass) and consider standard deviation
    significant_threshold = mean_mass * 0.5
    outlier_threshold = mean_mass + 2 * std_mass
    
    # Filter rows based on both thresholds
    significant_rows = (row_masses > significant_threshold) & (row_masses < outlier_threshold)
    
    # Filter y_coords based on significant mass
    valid_y_coords = y_coords[np.isin(y_coords, np.where(significant_rows)[0])]
    
    if len(valid_y_coords) == 0:
        return {'height': 0.0, 'mass': 0}
    
    # Get highest point (minimum y coordinate) from valid coordinates
    height = np.min(valid_y_coords)
    
    # Calculate total mass from valid coordinates
    total_mass = len(valid_y_coords)
    
    return {
        'height': float(height),
        'mass': float(total_mass)
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
    
    # Initialize model
    try:
        model = InferenceModel(model_path, config_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Video input
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    
    if video_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("temp_video.mp4")
        with open(temp_path, 'wb') as f:
            f.write(video_file.read())
        
        # Open video
        cap = cv2.VideoCapture(str(temp_path))
        
        # Video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
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
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process 1 frame per second
            if frame_idx % fps == 0:
                # Extract ROI
                roi = extract_roi(frame, (roi_x, roi_y, roi_w, roi_h))
                
                # Convert BGR to RGB
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                
                # Get predictions
                predictions = model.predict(roi_rgb)
                predictions_history.append(predictions)  # Store predictions
                
                # Calculate fluid level metrics
                level_metrics = calculate_fluid_level(predictions['fluid_mask'])
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
                vis_frame = overlay_masks(roi_rgb, predictions, overlay_alpha)
                
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
                Frame: {frame_idx}/{frame_count}
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
            progress = frame_idx / frame_count
            progress_bar.progress(progress)
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        temp_path.unlink()
        
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
