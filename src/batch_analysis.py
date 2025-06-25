import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from refine_net import EnhancedRefineNet
import plotly.graph_objects as go
from datetime import datetime
import albumentations as A

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

class BatchAnalyzer:
    def __init__(self, model_path: str, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EnhancedRefineNet(self.config)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove module prefix if it exists
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = A.Compose([
            A.Resize(
                self.config['data_pipeline']['image_size'][0],
                self.config['data_pipeline']['image_size'][1]
            )
        ])
        
        print(f"Model loaded successfully on {self.device}")

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
        
        # Resize masks back to original size
        h, w = image.shape[:2]
        fluid_mask = cv2.resize(fluid_mask, (w, h))
        slag_mask = cv2.resize(slag_mask, (w, h))
        
        return {
            'fluid_mask': fluid_mask,
            'slag_mask': slag_mask,
            'quality': {
                'label': QUALITY_LABELS[quality_pred],
                'probabilities': {QUALITY_LABELS[i]: float(p) for i, p in enumerate(quality_probs)}
            }
        }

def extract_roi(frame: np.ndarray, roi_coords: tuple) -> np.ndarray:
    """Extract ROI from frame using coordinates"""
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
    
    # Choose text color based on quality
    color = QUALITY_COLORS[quality]
    
    # Add quality info
    text = f"Quality: {quality}"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add probabilities
    y_offset = 60
    for label, prob in probs.items():
        text = f"{label}: {prob:.2f}"
        cv2.putText(overlay, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, QUALITY_COLORS[label], 1)
        y_offset += 20
    
    return overlay

def calculate_fluid_level(mask: np.ndarray) -> float:
    """Calculate fluid level from mask"""
    if not mask.any():
        return 0.0
    
    # Find all y-coordinates where fluid is present
    y_coords = np.where(mask > 0.5)[0]
    if len(y_coords) == 0:
        return 0.0
    
    # Get highest point (minimum y coordinate)
    return float(np.min(y_coords))

def process_video(input_path: str, output_path: str, model_path: str, config_path: str, roi_coords: tuple):
    """Process video file and generate analysis"""
    # Initialize analyzer
    analyzer = BatchAnalyzer(model_path, config_path)
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frames to skip (5 seconds worth)
    frames_to_skip = fps * 5
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 2, (width, height))  # 2 fps since we're taking 1 frame every 5 seconds
    
    # Check if video writer is initialized properly
    if not out.isOpened():
        print(f"Failed to initialize video writer for {output_path}")
        print(f"Codec: mp4v, FPS: 2, Size: {width}x{height}")
        return
    else:
        print(f"Video writer initialized successfully for {output_path}")
    
    # Initialize results storage
    results = []
    frame_idx = 0
    
    print(f"Processing video: {input_path}")
    print(f"Frame count: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Processing every {frames_to_skip} frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 5 seconds
        if frame_idx % frames_to_skip == 0:
            # Extract ROI
            roi = extract_roi(frame, roi_coords)
            
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            predictions = analyzer.predict(roi_rgb)
            
            # Calculate fluid level
            fluid_level = calculate_fluid_level(predictions['fluid_mask'])
            
            # Store results
            results.append({
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'fluid_level': fluid_level,
                'quality': predictions['quality']['label'],
                **predictions['quality']['probabilities']
            })
            
            # Create visualization
            vis_frame = overlay_masks(roi_rgb, predictions)
            
            # Convert back to BGR for video writing
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            
            # Place ROI back in full frame
            x, y, w, h = roi_coords
            frame[y:y+h, x:x+w] = vis_frame
            
            # Add timestamp
            timestamp = frame_idx / fps
            cv2.putText(frame, f"Time: {timestamp:.1f}s", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Print progress
            print(f"Processed frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = output_path.replace('.mp4', '_analysis.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"Analysis complete!")
    print(f"Video saved to: {output_path}")
    print(f"CSV saved to: {csv_path}")

def main():
    # Default paths
    model_path = "models/model002.pt"
    config_path = "config/config.json"
    
    # Default ROI coordinates (can be adjusted)
    roi_coords = (400, 50, 1200, 950)  # x, y, w, h
    
    # Get input video path
    input_path = input("Enter input video path: ")
    
    # Generate output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/analysis_{timestamp}.mp4"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Process video
    process_video(input_path, output_path, model_path, config_path, roi_coords)

if __name__ == "__main__":
    main()