import os
import zipfile
from pathlib import Path
import shutil
import json
import argparse

def create_project_zips(validation_only=False, custom_batch_size=None, custom_image_size=None):
    """Create separate zip files for code and dataset with flexible configuration"""
    # Files to include in code zip
    code_files = [
        'src/refine_net.py',
        'src/data_pipeline.py',
        'src/train_model.py',
        'config/config.json',
        'requirements/requirements.txt'
    ]
    
    # Create temp directories
    temp_code_dir = Path('temp_project')
    temp_code_dir.mkdir(exist_ok=True)
    
    temp_data_dir = Path('temp_dataset')
    temp_data_dir.mkdir(exist_ok=True)
    dataset_dir = temp_data_dir / 'Cvat_dataset'
    dataset_dir.mkdir(exist_ok=True)
    
    # Create src directory in temp
    (temp_code_dir / 'src').mkdir(exist_ok=True)
    
    # Copy code files
    print("Copying code files...")
    for file_path in code_files:
        source_path = Path(file_path)
        dest_path = temp_code_dir / source_path.name
        if source_path.exists():
            if source_path.parent.name == 'src':
                dest_path = temp_code_dir / 'src' / source_path.name
            shutil.copy2(source_path, dest_path)
            print(f"Copied {file_path}")
        else:
            print(f"Warning: {file_path} not found")
    
    # Update config
    config_path = temp_code_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update paths for Colab
        config['data_pipeline']['cvat_annotations_path'] = 'Cvat_dataset/annotations.xml'
        config['data_pipeline']['images_dir'] = 'Cvat_dataset/images/Train'
        
        # Update batch size and image size if provided
        if custom_batch_size:
            config['data_pipeline']['batch_size'] = custom_batch_size
        if custom_image_size:
            config['data_pipeline']['image_size'] = custom_image_size
            
        # Add validation mode flag
        config['validation_only'] = validation_only
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print("Updated configuration")
    
    # Copy dataset files
    print("\nCopying dataset files...")
    
    # Copy annotations
    annotations_path = 'data/Cvat_dataset/annotations/annotations.xml'
    if os.path.exists(annotations_path):
        shutil.copy2(annotations_path, dataset_dir / 'annotations.xml')
        print("Copied annotations.xml")
    else:
        print("Warning: annotations.xml not found")
    
    # Copy images
    images_dir = Path('data/Cvat_dataset/images/Train')
    if images_dir.exists():
        dest_images_dir = dataset_dir / 'images' / 'Train'
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all images
        image_count = 0
        for img in images_dir.glob('*.png'):
            shutil.copy2(img, dest_images_dir)
            image_count += 1
        print(f"Copied {image_count} training images")
    else:
        print("Warning: images directory not found")
    
    # Create Colab notebook
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# RefineNet Training with Quality Classification\n", 
                          "This notebook trains a RefineNet model with quality classification and TensorBoard monitoring."]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Disable TensorFlow warnings and initialization\n",
                          "import os\n",
                          "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TF logging\n",
                          "os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Use GPU device ordering\n",
                          "os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Install required packages\n",
                          "!pip install -q torch torchvision tensorboard"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Verify GPU\n",
                          "!nvidia-smi"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Mount Google Drive\n",
                          "from google.colab import drive\n",
                          "drive.mount('/content/drive')\n",
                          "\n",
                          "# Create project directory\n",
                          "!mkdir -p /content/drive/MyDrive/colab/refine_net\n",
                          "%cd /content/drive/MyDrive/colab/refine_net"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Install requirements\n",
                          "!pip install -r requirements.txt"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Extract project files\n",
                          "!unzip -o project_code.zip\n",
                          "!unzip -o project_dataset.zip"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Load TensorBoard extension\n",
                          "%load_ext tensorboard"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Clear any existing TensorBoard instances\n",
                          "!kill -9 $(ps -ef | grep tensorboard | grep -v grep | awk '{print $2}') 2>/dev/null || true\n",
                          "\n",
                          "# Start TensorBoard\n",
                          "%tensorboard --logdir logs"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Configure PyTorch\n",
                          "import torch\n",
                          "print(f'PyTorch version: {torch.__version__}')\n",
                          "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                          "if torch.cuda.is_available():\n",
                          "    print(f'CUDA device: {torch.cuda.get_device_name(0)}')\n",
                          "    # Enable TF32 for better performance on Ampere GPUs\n",
                          "    if torch.cuda.get_device_capability()[0] >= 8:\n",
                          "        torch.backends.cuda.matmul.allow_tf32 = True\n",
                          "        torch.backends.cudnn.allow_tf32 = True\n",
                          "        print('TF32 enabled for Ampere+ GPUs')"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Start training or validation\n",
                          "!python src/train_model.py"]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## TensorBoard Metrics\n",
                          "\n",
                          "The training progress can be monitored in real-time through TensorBoard above. The following metrics are tracked:\n",
                          "\n",
                          "- Loss (training and validation)\n",
                          "- Quality classification metrics (accuracy, precision, recall, F1)\n",
                          "- Segmentation IoU scores\n",
                          "- Learning rate\n",
                          "- GPU memory usage\n",
                          "- Training time statistics\n",
                          "\n",
                          "You can also visualize:\n",
                          "- Model architecture graph\n",
                          "- Input images\n",
                          "- Predicted vs ground truth masks"]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('train_refine_net.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    print("Created Colab notebook with TensorBoard support")
    
    # Create zip files
    print("\nCreating zip files...")
    
    # Create code zip
    code_zip_path = 'project_code.zip'
    shutil.make_archive('project_code', 'zip', temp_code_dir)
    code_size = os.path.getsize(code_zip_path) / (1024 * 1024)  # MB
    print(f"Created {code_zip_path} ({code_size:.1f} MB)")
    
    # Create dataset zip
    data_zip_path = 'project_dataset.zip'
    shutil.make_archive('project_dataset', 'zip', temp_data_dir)
    data_size = os.path.getsize(data_zip_path) / (1024 * 1024)  # MB
    print(f"Created {data_zip_path} ({data_size:.1f} MB)")
    
    # Cleanup
    shutil.rmtree(temp_code_dir)
    shutil.rmtree(temp_data_dir)
    
    print("\nProject files prepared for Colab:")
    print(f"1. {code_zip_path} - Code files ({code_size:.1f} MB)")
    print(f"2. {data_zip_path} - Dataset files ({data_size:.1f} MB)")
    print("3. train_refine_net.ipynb - Colab notebook")
    
    print("\nInstructions:")
    print("1. Open train_refine_net.ipynb in Google Colab")
    print("2. Upload both zip files to /content/drive/MyDrive/colab/refine_net/")
    print("3. Run all cells to start training")
    print("\nFeatures enabled:")
    print("- Automatic GPU optimization")
    print("- TF32 precision (when available)")
    print("- Multi-task learning")
    print("- Real-time TensorBoard monitoring")
    print("- Comprehensive metrics tracking")
    print("- Early stopping")
    if validation_only:
        print("- Validation-only mode")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare project for Colab')
    parser.add_argument('--validation-only', action='store_true', help='Run only validation')
    parser.add_argument('--batch-size', type=int, help='Custom batch size')
    parser.add_argument('--image-size', type=int, nargs=2, help='Custom image size (width height)')
    args = parser.parse_args()
    
    create_project_zips(
        validation_only=args.validation_only,
        custom_batch_size=args.batch_size,
        custom_image_size=args.image_size if args.image_size else None
    )
