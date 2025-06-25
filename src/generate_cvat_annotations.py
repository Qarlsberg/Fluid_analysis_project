import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from inference_app import InferenceModel
from sklearn.cluster import KMeans
from collections import defaultdict
import logging
import pycocotools.mask as mask_util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVATAnnotationGenerator:
    def __init__(self, model_path: str, config_path: str, output_dir: str):
        """Initialize the annotation generator"""
        self.model = InferenceModel(model_path, config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create directories for frames and annotations
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Define ROI coordinates (matching inference_app.py)
        self.roi = {
            'x': 400,  # roi_x from inference_app
            'y': 50,   # roi_y from inference_app
            'w': 1200, # roi_w from inference_app
            'h': 950   # roi_h from inference_app
        }
    def mask_to_rle(self, mask):
        """Convert binary mask to RLE format"""
        rle = mask_util.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle

    def preprocess_frame(self, frame):
        """Extract ROI and resize to model input size"""
        # Extract ROI
        roi_frame = frame[
            self.roi['y']:self.roi['y'] + self.roi['h'],
            self.roi['x']:self.roi['x'] + self.roi['w']
        ]
        
        # Convert to uint8 if needed
        if roi_frame.dtype != np.uint8:
            roi_frame = (roi_frame * 255).astype(np.uint8)
            
        # Resize to 512x512 (model input size)
        resized_frame = cv2.resize(roi_frame, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Ensure output is uint8
        if resized_frame.dtype != np.uint8:
            resized_frame = resized_frame.astype(np.uint8)
            
        return resized_frame
    
    def compute_frame_histogram(self, frame):
        """Compute color and intensity histograms for a frame"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Compute intensity histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        i_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        i_hist = cv2.normalize(i_hist, i_hist).flatten()
        
        # Combine histograms with different weights
        combined_hist = np.concatenate([
            h_hist * 0.3,  # Color information
            s_hist * 0.2,  # Saturation
            v_hist * 0.2,  # Value/brightness
            i_hist * 0.3   # Intensity
        ])
        
        return combined_hist
    
    def analyze_video_histograms(self, video_path: str, n_segments: int) -> tuple:
        """Analyze entire video and compute histograms"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error opening video file: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise Exception(f"Could not get frame count for video: {video_path}")
            
        # Calculate frame step to get desired number of segments
        frame_step = max(1, total_frames // n_segments)
        
        frame_histograms = []
        frame_indices = []
        frames = []
        
        with tqdm(total=n_segments, desc="Analyzing video") as pbar:
            frame_idx = 0
            while len(frames) < n_segments and frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame (extract ROI and resize)
                processed_frame = self.preprocess_frame(frame)
                
                # Compute histogram on processed frame
                hist = self.compute_frame_histogram(processed_frame)
                
                frame_histograms.append(hist)
                frame_indices.append(frame_idx)
                frames.append(processed_frame)
                
                frame_idx += frame_step
                pbar.update(1)
        
        cap.release()
        return frames, frame_histograms, frame_indices
    
    def select_diverse_frames(self, frames: list, histograms: list, indices: list, 
                            n_frames: int = 8) -> tuple:
        """Select diverse frames using K-means clustering on histograms"""
        logger.info(f"Selecting {n_frames} diverse frames from {len(frames)} total frames")
        
        # Convert to numpy array for clustering
        histograms_array = np.array(histograms)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=min(n_frames, len(frames)), n_init=10)
        clusters = kmeans.fit_predict(histograms_array)
        
        # Select frames closest to cluster centers
        selected_indices = []
        selected_frames = []
        
        for i in range(min(n_frames, len(frames))):
            cluster_frames = np.where(clusters == i)[0]
            if len(cluster_frames) > 0:
                # Select frame closest to cluster center
                center_distances = np.linalg.norm(
                    histograms_array[cluster_frames] - kmeans.cluster_centers_[i], axis=1
                )
                selected_idx = cluster_frames[center_distances.argmin()]
                selected_indices.append(indices[selected_idx])
                selected_frames.append(frames[selected_idx])
        
        # Sort by frame index to maintain temporal order
        sorted_pairs = sorted(zip(selected_indices, selected_frames))
        selected_indices, selected_frames = zip(*sorted_pairs)
        
        return list(selected_frames), list(selected_indices)
    
    def predict_frames(self, frames: list) -> list:
        """Run model predictions on frames"""
        predictions = []
        difficulties = []
        
        with tqdm(total=len(frames), desc="Running predictions") as pbar:
            for frame in frames:
                # Frame is already preprocessed (ROI extracted and resized to 512x512)
                pred = self.model.predict(frame)
                predictions.append(pred)
                
                # Calculate difficulty score based on multiple factors
                quality_probs = pred['quality']['probabilities']
                max_prob = max(quality_probs.values())
                readability = pred['quality']['readability']
                
                # Calculate mask prediction confidence
                fluid_conf = np.mean(pred['fluid_mask'])
                slag_conf = np.mean(pred['slag_mask'])
                
                # Calculate mask edge complexity
                fluid_edges = cv2.Canny((pred['fluid_mask'] * 255).astype(np.uint8), 100, 200)
                slag_edges = cv2.Canny((pred['slag_mask'] * 255).astype(np.uint8), 100, 200)
                edge_complexity = (np.sum(fluid_edges) + np.sum(slag_edges)) / (fluid_edges.size * 255)
                
                # Combine factors into difficulty score
                difficulty = (1 - max_prob) * 0.3 + \
                           (1 - readability) * 0.2 + \
                           (1 - (fluid_conf + slag_conf) / 2) * 0.2 + \
                           edge_complexity * 0.3
                
                difficulties.append(difficulty)
                pbar.update(1)
        
        return predictions, difficulties
    
    def select_final_frames(self, frames: list, predictions: list, difficulties: list, 
                          frame_indices: list, n_frames: int = 8) -> tuple:
        """Select final frames based on quality distribution and difficulty"""
        # Group frames by quality
        quality_groups = defaultdict(list)
        for i, pred in enumerate(predictions):
            quality = pred['quality']['label']
            quality_groups[quality].append(i)
        
        selected_indices = []
        
        # Calculate target frames per quality group
        total_frames = sum(len(indices) for indices in quality_groups.values())
        frames_per_quality = {
            quality: max(int(len(indices) / total_frames * n_frames), 1)
            for quality, indices in quality_groups.items()
        }
        
        # Adjust to meet target total
        total_allocated = sum(frames_per_quality.values())
        if total_allocated < n_frames:
            # Distribute remaining frames to largest groups
            remaining = n_frames - total_allocated
            sorted_qualities = sorted(
                quality_groups.keys(),
                key=lambda q: len(quality_groups[q]),
                reverse=True
            )
            for i in range(remaining):
                frames_per_quality[sorted_qualities[i % len(sorted_qualities)]] += 1
        
        # Select frames from each quality group
        for quality, target_frames in frames_per_quality.items():
            indices = quality_groups[quality]
            if indices:
                # Sort by difficulty
                group_difficulties = [difficulties[i] for i in indices]
                sorted_indices = [x for _, x in sorted(
                    zip(group_difficulties, indices),
                    reverse=True
                )]
                
                # Select top N most difficult frames
                selected_indices.extend(sorted_indices[:target_frames])
        
        # Get selected frames and their metadata
        selected_frames = [frames[i] for i in selected_indices]
        selected_predictions = [predictions[i] for i in selected_indices]
        selected_frame_indices = [frame_indices[i] for i in selected_indices]
        
        # Sort by frame index to maintain temporal order
        sorted_tuples = sorted(zip(selected_frame_indices, selected_frames, selected_predictions))
        selected_frame_indices, selected_frames, selected_predictions = zip(*sorted_tuples)
        
        return list(selected_frames), list(selected_predictions), list(selected_frame_indices)
    
    def generate_cvat_xml(self, video_name: str, frame_indices: list, 
                         predictions: list, output_path: str):
        """Generate CVAT-compatible XML annotations with RLE masks"""
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        
        # Create meta section
        meta = ET.SubElement(root, "meta")
        job = ET.SubElement(meta, "job")
        
        # Add job details
        ET.SubElement(job, "id").text = str(hash(video_name) % 10000000)
        ET.SubElement(job, "size").text = str(len(frame_indices))
        ET.SubElement(job, "mode").text = "annotation"
        ET.SubElement(job, "overlap").text = "0"
        ET.SubElement(job, "bugtracker").text = ""
        ET.SubElement(job, "created").text = datetime.now().isoformat()
        ET.SubElement(job, "updated").text = datetime.now().isoformat()
        ET.SubElement(job, "subset").text = "Train"
        ET.SubElement(job, "start_frame").text = "0"
        ET.SubElement(job, "stop_frame").text = str(len(frame_indices) - 1)
        ET.SubElement(job, "frame_filter").text = ""
        
        # Add segments
        segments = ET.SubElement(job, "segments")
        segment = ET.SubElement(segments, "segment")
        ET.SubElement(segment, "id").text = str(hash(video_name + "_segment") % 10000000)
        ET.SubElement(segment, "start").text = "0"
        ET.SubElement(segment, "stop").text = str(len(frame_indices) - 1)
        ET.SubElement(segment, "url").text = f"https://app.cvat.ai/api/jobs/{hash(video_name) % 10000000}"
        
        # Add labels
        labels = ET.SubElement(job, "labels")
        
        # Slag obstruction label
        slag_label = ET.SubElement(labels, "label")
        ET.SubElement(slag_label, "name").text = "Slag_obstruction"
        ET.SubElement(slag_label, "color").text = "#018ce8"
        ET.SubElement(slag_label, "type").text = "mask"
        ET.SubElement(slag_label, "attributes")
        
        # Fluid label
        fluid_label = ET.SubElement(labels, "label")
        ET.SubElement(fluid_label, "name").text = "Fluid"
        ET.SubElement(fluid_label, "color").text = "#fa3253"
        ET.SubElement(fluid_label, "type").text = "mask"
        
        # Add fluid attributes
        fluid_attrs = ET.SubElement(fluid_label, "attributes")
        fluid_attr = ET.SubElement(fluid_attrs, "attribute")
        ET.SubElement(fluid_attr, "name").text = "Type"
        ET.SubElement(fluid_attr, "mutable").text = "False"
        ET.SubElement(fluid_attr, "input_type").text = "radio"
        ET.SubElement(fluid_attr, "default_value").text = "Start_stable"
        ET.SubElement(fluid_attr, "values").text = "\n".join([
            "Start_stable", "Start_slow", "Fermenting", "Overfermenting",
            "Boiling_low_viscosity", "Boiling_high_viscosity"
        ])
        
        # Image quality label
        quality_label = ET.SubElement(labels, "label")
        ET.SubElement(quality_label, "name").text = "Image quality"
        ET.SubElement(quality_label, "color").text = "#66ff66"
        ET.SubElement(quality_label, "type").text = "tag"
        
        # Add quality attributes
        quality_attrs = ET.SubElement(quality_label, "attributes")
        quality_attr = ET.SubElement(quality_attrs, "attribute")
        ET.SubElement(quality_attr, "name").text = "Type"
        ET.SubElement(quality_attr, "mutable").text = "False"
        ET.SubElement(quality_attr, "input_type").text = "select"
        ET.SubElement(quality_attr, "default_value").text = "Clear"
        ET.SubElement(quality_attr, "values").text = "Clear\nFuzzy\nSteam"
        
        # Add dumped timestamp
        ET.SubElement(meta, "dumped").text = datetime.now().isoformat()
        
        # Add images and their annotations
        for idx, (frame_idx, pred) in enumerate(zip(frame_indices, predictions)):
            frame_name = f'frame_{frame_idx:04d}.png'
            image = ET.SubElement(root, "image")
            image.set("id", str(idx))
            image.set("name", frame_name)
            image.set("width", "512")
            image.set("height", "512")
            
            # Add fluid mask if present
            if 'fluid_mask' in pred and np.any(pred['fluid_mask']):
                mask = ET.SubElement(image, "mask")
                mask.set("label", "Fluid")
                mask.set("source", "semi-auto")
                mask.set("occluded", "0")
                mask.set("z_order", "0")
                
                # Convert mask to RLE
                rle = self.mask_to_rle(pred['fluid_mask'])
                mask.set("rle", json.dumps(rle))
                
                # Add fluid type attribute
                attr = ET.SubElement(mask, "attribute")
                attr.set("name", "Type")
                attr.text = pred.get('fluid_type', 'Start_stable')
            
            # Add slag mask if present
            if 'slag_mask' in pred and np.any(pred['slag_mask']):
                mask = ET.SubElement(image, "mask")
                mask.set("label", "Slag_obstruction")
                mask.set("source", "semi-auto")
                mask.set("occluded", "0")
                mask.set("z_order", "0")
                
                # Convert mask to RLE
                rle = self.mask_to_rle(pred['slag_mask'])
                mask.set("rle", json.dumps(rle))
            
            # Add image quality tag
            if 'quality' in pred:
                tag = ET.SubElement(image, "tag")
                tag.set("label", "Image quality")
                tag.set("source", "manual")
                
                quality_attr = ET.SubElement(tag, "attribute")
                quality_attr.set("name", "Type")
                quality_attr.text = pred['quality']['label']
        
        # Write the XML file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def process_video(self, video_path: str, initial_frames: int = 8, 
                     final_frames: int = 8):
        """Process a video and generate CVAT annotations"""
        video_path = Path(video_path)
        video_name = video_path.stem
        logger.info(f"Processing video: {video_name}")
        
        # Create output directories for this video
        video_output_dir = self.frames_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        # Step 1: Analyze video frames throughout its duration
        frames, histograms, frame_indices = self.analyze_video_histograms(
            str(video_path), initial_frames
        )
        logger.info(f"Analyzed {len(frames)} frames across entire video")
        
        # Step 2: Select diverse frames based on histograms
        diverse_frames, diverse_indices = self.select_diverse_frames(
            frames, histograms, frame_indices, initial_frames
        )
        logger.info(f"Selected {len(diverse_frames)} diverse frames")
        
        # Step 3: Get predictions for diverse frames
        predictions, difficulties = self.predict_frames(diverse_frames)
        logger.info("Generated predictions for diverse frames")
        
        # Step 4: Select final frames based on predictions
        selected_frames, selected_predictions, selected_indices = self.select_final_frames(
            diverse_frames, predictions, difficulties, diverse_indices, final_frames
        )
        logger.info(f"Selected {len(selected_frames)} final frames")
        
        # Save selected frames (512x512 preprocessed frames)
        for idx, (frame, frame_idx) in enumerate(zip(selected_frames, selected_indices)):
            frame_path = video_output_dir / f'frame_{frame_idx:04d}.png'
            cv2.imwrite(str(frame_path), frame)
        
        # Generate CVAT annotations
        xml_path = self.output_dir / f'{video_name}_annotations.xml'
        self.generate_cvat_xml(video_name, selected_indices, selected_predictions, str(xml_path))
        
        # Save frame selection metadata
        metadata = {
            'video_name': video_name,
            'total_frames_analyzed': len(frames),
            'diverse_frames_selected': len(diverse_frames),
            'final_frames_selected': len(selected_frames),
            'frame_indices': selected_indices,
            'quality_distribution': defaultdict(int),
            'average_difficulty': np.mean(difficulties),
            'roi_settings': self.roi,  # Include ROI settings in metadata
            'frame_size': {
                'width': 512,
                'height': 512
            }
        }
        
        # Count quality distribution
        for pred in selected_predictions:
            metadata['quality_distribution'][pred['quality']['label']] += 1
        
        metadata_path = self.output_dir / f'{video_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'n_frames': len(selected_frames),
            'frame_indices': selected_indices,
            'output_dir': str(video_output_dir),
            'xml_path': str(xml_path),
            'metadata_path': str(metadata_path)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate CVAT annotations from videos')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--initial_frames', type=int, default=8, 
                       help='Number of diverse frames to select initially')
    parser.add_argument('--final_frames', type=int, default=8, 
                       help='Number of final frames to select for annotation')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CVATAnnotationGenerator(args.model_path, args.config_path, args.output_dir)
    
    # Process all videos in directory
    video_dir = Path(args.video_dir)
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
    
    results = []
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            result = generator.process_video(
                video_file, 
                initial_frames=args.initial_frames,
                final_frames=args.final_frames
            )
            results.append({
                'video': str(video_file),
                'status': 'success',
                **result
            })
        except Exception as e:
            logger.error(f"Error processing {video_file}: {str(e)}")
            results.append({
                'video': str(video_file),
                'status': 'error',
                'error': str(e)
            })
    
    # Save processing results
    results_path = Path(args.output_dir) / 'processing_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
