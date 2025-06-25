import cv2
import numpy as np
import random
from pathlib import Path
import argparse
import torch
import json
import albumentations as A
from refine_net import EnhancedRefineNet, QUALITY_READABILITY

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
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove module prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
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

def create_slideshow(input_folder: str, output_path: str, model_path: str, config_path: str, 
                    num_frames: int = 30, frame_duration: int = 2, transition_duration: float = 0.5):
    """
    Create a slideshow video from randomly selected frames with model inference visualization.
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return
    
    # Initialize model
    print("Initializing model...")
    model = InferenceModel(model_path, config_path)
    
    # Get list of all frame files
    print(f"Scanning directory: {input_folder}")
    frame_files = list(input_path.glob("*.png"))
    if not frame_files:
        print(f"Error: No PNG files found in {input_folder}")
        return
        
    print(f"Found {len(frame_files)} frames")
    
    if len(frame_files) < num_frames:
        print(f"Warning: Only {len(frame_files)} frames available, using all frames")
        num_frames = len(frame_files)
    
    # Randomly select frames
    print(f"Selecting {num_frames} random frames...")
    selected_frames = random.sample(frame_files, num_frames)
    
    # ROI coordinates from original app
    roi_coords = (400, 50, 1200, 950)
    
    # Read first frame to get dimensions
    print("Reading first frame to get dimensions...")
    first_frame = cv2.imread(str(selected_frames[0]))
    if first_frame is None:
        print(f"Error: Could not read frame {selected_frames[0]}")
        return
        
    # Extract ROI from first frame to get output dimensions
    roi = extract_roi(first_frame, roi_coords)
    height, width = roi.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Calculate video parameters
    fps = 30
    frame_duration_frames = int(frame_duration * fps)
    transition_duration_frames = int(transition_duration * fps)
    
    # Initialize video writer
    print(f"Initializing video writer: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not initialize video writer!")
        return
    
    # Process each frame
    total_frames = len(selected_frames)
    print("\nCreating slideshow...")
    
    for i, frame_path in enumerate(selected_frames):
        print(f"Processing frame {i+1}/{total_frames}", end='\r')
        
        # Read current frame
        current_frame = cv2.imread(str(frame_path))
        if current_frame is None:
            print(f"\nError: Could not read frame {frame_path}")
            continue
        
        # Extract ROI and convert to RGB
        current_roi = extract_roi(current_frame, roi_coords)
        current_roi_rgb = cv2.cvtColor(current_roi, cv2.COLOR_BGR2RGB)
        
        # Get predictions and create visualization
        predictions = model.predict(current_roi_rgb)
        current_vis = overlay_masks(current_roi_rgb, predictions)
        
        # Convert back to BGR for video writing
        current_vis = cv2.cvtColor(current_vis, cv2.COLOR_RGB2BGR)
        
        # Write static frame
        for _ in range(frame_duration_frames):
            out.write(current_vis)
        
        # Create transition to next frame if not last frame
        if i < len(selected_frames) - 1:
            # Read and process next frame
            next_frame = cv2.imread(str(selected_frames[i + 1]))
            if next_frame is None:
                print(f"\nError: Could not read next frame {selected_frames[i + 1]}")
                continue
            
            next_roi = extract_roi(next_frame, roi_coords)
            next_roi_rgb = cv2.cvtColor(next_roi, cv2.COLOR_BGR2RGB)
            predictions = model.predict(next_roi_rgb)
            next_vis = overlay_masks(next_roi_rgb, predictions)
            next_vis = cv2.cvtColor(next_vis, cv2.COLOR_RGB2BGR)
            
            # Create fade transition
            for j in range(transition_duration_frames):
                alpha = j / transition_duration_frames
                blended = cv2.addWeighted(current_vis, 1 - alpha, next_vis, alpha, 0)
                out.write(blended)
    
    # Release video writer
    out.release()
    
    # Verify the video was created
    output_path = Path(output_path)
    if not output_path.exists():
        print("\nError: Video file was not created!")
        return
    
    video_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
    total_duration = (frame_duration + transition_duration) * num_frames
    
    print(f"\nSlideshow created successfully!")
    print(f"Output: {output_path}")
    print(f"Size: {video_size:.1f} MB")
    print(f"Duration: {total_duration:.1f} seconds")
    print(f"Frames used: {num_frames}")

def main():
    parser = argparse.ArgumentParser(description="Create slideshow from frames with model inference")
    parser.add_argument("--input", required=True, help="Input folder containing frames")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to include")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration to show each frame (seconds)")
    parser.add_argument("--transition", type=float, default=0.5, help="Transition duration (seconds)")
    
    args = parser.parse_args()
    
    create_slideshow(
        args.input,
        args.output,
        args.model,
        args.config,
        args.frames,
        args.duration,
        args.transition
    )

if __name__ == "__main__":
    main()
