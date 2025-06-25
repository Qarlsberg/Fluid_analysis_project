import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from typing import Tuple, Dict, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quality label mapping
QUALITY_LABELS = {
    'Clear': 0,
    'Steam': 1,
    'Fuzzy': 2
}

def collate_fn(batch):
    """Custom collate function to handle None values"""
    images = torch.stack([item[0] for item in batch])
    masks = {
        k: torch.stack([item[1][k] for item in batch])
        for k in batch[0][1].keys()
    }
    quality_labels = torch.tensor([item[2] for item in batch])
    return images, masks, quality_labels

class CustomMotionBlur(A.ImageOnlyTransform):
    """Custom motion blur to simulate subtle camera movement"""
    def __init__(self, kernel_size=(3,5), angle_range=(-5,5), p=0.5):
        super().__init__(always_apply=False, p=p)
        self.kernel_size = kernel_size
        self.angle_range = angle_range

    def apply(self, img, **params):
        ksize = np.random.randint(self.kernel_size[0], self.kernel_size[1])
        if ksize % 2 == 0:
            ksize += 1  # Ensure odd kernel size
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        
        # Create motion blur kernel
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize-1)/2), :] = np.ones(ksize)
        kernel = cv2.warpAffine(kernel, 
                              cv2.getRotationMatrix2D((ksize/2, ksize/2), angle, 1.0), 
                              (ksize, ksize))
        kernel = kernel / np.sum(kernel)
        
        # Apply motion blur
        return cv2.filter2D(img, -1, kernel)

    def get_transform_init_args_names(self):
        return ("kernel_size", "angle_range")

class CVATDatasetLoader:
    """Efficient CVAT dataset loader with streaming capabilities"""
    def __init__(self, annotation_path: str, image_dir: str):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.annotations = {}
        self.image_info = {}
        self._initialize_cvat()

    def _initialize_cvat(self):
        """Initialize CVAT annotations from XML"""
        try:
            tree = ET.parse(self.annotation_path)
            root = tree.getroot()
            
            # Track quality distribution
            quality_counts = {label: 0 for label in QUALITY_LABELS.keys()}
            
            # Parse image information and annotations
            for image in root.findall('.//image'):
                image_name = image.get('name')
                image_width = int(image.get('width'))
                image_height = int(image.get('height'))
                
                # Get image quality from tag attribute
                quality = 'Clear'  # Default quality
                quality_tag = image.find(".//tag[@label='Image quality']")
                if quality_tag is not None:
                    quality_type = quality_tag.find(".//attribute[@name='Type']")
                    if quality_type is not None:
                        quality = quality_type.text
                
                quality_counts[quality] += 1
                
                # Verify image exists
                img_path = os.path.join(self.image_dir, image_name)
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found: {img_path}")
                    continue
                
                # Store image info
                self.image_info[image_name] = {
                    'file_name': image_name,
                    'width': image_width,
                    'height': image_height,
                    'quality': QUALITY_LABELS[quality]
                }
                
                # Parse polygon annotations
                fluid_polygons = []
                slag_polygons = []
                
                for polygon in image.findall('.//polygon'):
                    label = polygon.get('label')
                    points_str = polygon.get('points')
                    points = [float(coord) for point in points_str.split(';') 
                             for coord in point.split(',')]
                    
                    if label == 'Fluid':
                        fluid_polygons.append({
                            'segmentation': points,
                            'category_id': 1
                        })
                    elif label == 'Slag_obstruction':
                        slag_polygons.append({
                            'segmentation': points,
                            'category_id': 2
                        })
                
                self.annotations[image_name] = {
                    'fluid': fluid_polygons,
                    'slag': slag_polygons
                }
            
            logger.info(f"Successfully loaded CVAT annotations from {self.annotation_path}")
            logger.info(f"Found {len(self.image_info)} valid images with annotations")
            logger.info("Quality type distribution:")
            for quality, count in quality_counts.items():
                logger.info(f"  {quality}: {count} images")
            
        except Exception as e:
            logger.error(f"Error loading CVAT annotations: {str(e)}")
            raise

    def get_image_ids(self) -> List[str]:
        return list(self.image_info.keys())

    def get_image_info(self, img_id: str) -> Dict:
        return self.image_info[img_id]

    def get_segmentation_annotations(self, img_id: str) -> Dict:
        return self.annotations.get(img_id, {
            'fluid': [],
            'slag': []
        })

    def points_to_mask(self, points: List[float], height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) >= 6:
            points_array = np.array(points).reshape(-1, 2)
            points_array = points_array.astype(np.int32)
            cv2.fillPoly(mask, [points_array], 1)
        return mask

class SegmentationDataset(Dataset):
    """Dataset for segmentation with quality labels"""
    def __init__(
        self,
        cvat_loader: CVATDatasetLoader,
        image_size: Tuple[int, int],
        transform: A.Compose = None,
        is_train: bool = True
    ):
        self.cvat_loader = cvat_loader
        self.image_size = image_size
        self.transform = transform
        self.is_train = is_train
        self.image_ids = self.cvat_loader.get_image_ids()
        
        logger.info(f"Created dataset with {len(self.image_ids)} images")

    def _create_masks(self, annotations: Dict, height: int, width: int) -> Dict:
        fluid_mask = np.zeros((height, width), dtype=np.uint8)
        slag_mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in annotations['fluid']:
            polygon_mask = self.cvat_loader.points_to_mask(
                ann['segmentation'], height, width
            )
            fluid_mask = np.maximum(fluid_mask, polygon_mask)
        
        for ann in annotations['slag']:
            polygon_mask = self.cvat_loader.points_to_mask(
                ann['segmentation'], height, width
            )
            slag_mask = np.maximum(slag_mask, polygon_mask)
        
        return {
            'fluid': fluid_mask,
            'slag': slag_mask
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        img_id = self.image_ids[idx]
        img_info = self.cvat_loader.get_image_info(img_id)
        
        # Load image
        img_path = os.path.join(self.cvat_loader.image_dir, img_info['file_name'])
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
        
        # Create masks from annotations
        annotations = self.cvat_loader.get_segmentation_annotations(img_id)
        masks = self._create_masks(annotations, img_info['height'], img_info['width'])
        
        # Get quality label
        quality_label = img_info['quality']
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=image,
                masks=[masks['fluid'], masks['slag']]
            )
            image = transformed['image']
            masks['fluid'] = transformed['masks'][0]
            masks['slag'] = transformed['masks'][1]
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        masks = {
            'fluid': torch.from_numpy(masks['fluid']).float(),
            'slag': torch.from_numpy(masks['slag']).float()
        }
        
        return image, masks, quality_label

def create_data_pipeline(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders with quality labels"""
    data_config = config['data_pipeline']
    aug_config = config['augmentation']
    
    # Initialize CVAT loader
    cvat_loader = CVATDatasetLoader(
        data_config['cvat_annotations_path'],
        data_config['images_dir']
    )
    
    # Create augmentation pipeline
    train_transform = A.Compose([
        A.Resize(data_config['image_size'][0], data_config['image_size'][1]),
        A.Rotate(limit=aug_config['rotation_range'], p=0.7),
        A.RandomBrightnessContrast(
            brightness_limit=aug_config['brightness_range'],
            contrast_limit=aug_config['contrast_range'],
            p=0.5
        ),
        A.GaussNoise(var_limit=(5.0, 30.0), p=aug_config['gaussian_noise']),
        A.GaussianBlur(blur_limit=(3, 7), p=aug_config['gaussian_blur']),
        CustomMotionBlur(
            kernel_size=tuple(aug_config['motion_blur']['kernel_size']),
            angle_range=(-aug_config['rotation_range'], aug_config['rotation_range']),
            p=aug_config['motion_blur']['probability']
        ),
        A.Perspective(
            scale=aug_config['perspective']['scale'],
            p=aug_config['perspective']['probability']
        )
    ])
    
    val_transform = A.Compose([
        A.Resize(data_config['image_size'][0], data_config['image_size'][1]),
    ])
    
    # Create datasets
    full_dataset = SegmentationDataset(
        cvat_loader,
        data_config['image_size'],
        transform=train_transform,
        is_train=True
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * data_config['validation_split'])
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        prefetch_factor=data_config['prefetch_factor'],
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        prefetch_factor=data_config['prefetch_factor'],
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collate_fn
    )
    
    logger.info(f"Created data pipeline with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    return train_loader, val_loader
