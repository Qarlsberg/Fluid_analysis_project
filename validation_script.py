#!/usr/bin/env python3
"""
Comprehensive validation script for Fluid Analysis Application
Tests all components to ensure functionality after streamlining changes.
"""

import sys
import json
import warnings
from pathlib import Path
import traceback

# Add src to path for imports
sys.path.append('src')

def test_imports():
    """Test all required imports for both modules"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    # Test refine_net.py imports
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models import resnet34, ResNet34_Weights
        from typing import Dict
        print("✅ refine_net.py dependencies imported successfully")
    except Exception as e:
        print(f"❌ refine_net.py import error: {e}")
        return False
    
    # Test inference_app.py imports
    try:
        import streamlit as st
        import cv2
        import numpy as np
        import pandas as pd
        from pathlib import Path
        import time
        import albumentations as A
        import plotly.graph_objects as go
        print("✅ inference_app.py dependencies imported successfully")
    except Exception as e:
        print(f"❌ inference_app.py import error: {e}")
        return False
    
    # Test custom module imports
    try:
        from refine_net import EnhancedRefineNet, QUALITY_READABILITY
        print("✅ Custom modules imported successfully")
    except Exception as e:
        print(f"❌ Custom module import error: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration file loading and validation"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    config_path = Path("config/config.json")
    
    # Check if config file exists
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False, None
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False, None
    
    # Validate configuration structure
    required_sections = ['data_pipeline', 'model']
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required configuration section: {section}")
            return False, None
    
    # Validate model configuration
    model_config = config['model']
    required_model_keys = ['backbone', 'pretrained', 'dropout', 'aux_loss', 'memory_efficient', 'channels']
    for key in required_model_keys:
        if key not in model_config:
            print(f"❌ Missing required model config key: {key}")
            return False, None
    
    # Validate data pipeline configuration
    data_config = config['data_pipeline']
    required_data_keys = ['image_size', 'batch_size']
    for key in required_data_keys:
        if key not in data_config:
            print(f"❌ Missing required data pipeline config key: {key}")
            return False, None
    
    print(f"✅ Configuration validation passed")
    print(f"   - Backbone: {model_config['backbone']}")
    print(f"   - Image size: {data_config['image_size']}")
    print(f"   - Batch size: {data_config['batch_size']}")
    print(f"   - Dropout: {model_config['dropout']}")
    
    return True, config

def test_model_architecture(config):
    """Test model instantiation with current config"""
    print("\n" + "=" * 60)
    print("TESTING MODEL ARCHITECTURE")
    print("=" * 60)
    
    if config is None:
        print("❌ Cannot test model - no valid configuration")
        return False
    
    try:
        from refine_net import EnhancedRefineNet
        
        # Suppress warnings during model creation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EnhancedRefineNet(config)
        
        print("✅ Model instantiated successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Test model components
        print("\n   Testing model components:")
        
        # Check encoder layers
        if hasattr(model, 'layer1') and hasattr(model, 'layer2'):
            print("   ✅ Encoder layers present")
        else:
            print("   ❌ Encoder layers missing")
            return False
        
        # Check decoder/refinement blocks
        if hasattr(model, 'refine1') and hasattr(model, 'refine2'):
            print("   ✅ Refinement blocks present")
        else:
            print("   ❌ Refinement blocks missing")
            return False
        
        # Check output heads
        if hasattr(model, 'fluid_head') and hasattr(model, 'slag_head'):
            print("   ✅ Segmentation heads present")
        else:
            print("   ❌ Segmentation heads missing")
            return False
        
        # Check quality classifier
        if hasattr(model, 'quality_classifier'):
            print("   ✅ Quality classifier present")
        else:
            print("   ❌ Quality classifier missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        return False

def test_file_paths():
    """Test that all required files and paths exist"""
    print("\n" + "=" * 60)
    print("TESTING FILE PATHS")
    print("=" * 60)
    
    required_files = [
        "src/refine_net.py",
        "src/inference_app.py",
        "config/config.json",
        "models/model002.pt",
        "requirements/requirements.txt"
    ]
    
    all_files_exist = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            if file_path.endswith('.pt'):
                size_mb = size / 1024 / 1024
                print(f"✅ {file_path} exists ({size_mb:.1f} MB)")
            else:
                print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_files_exist = False
    
    # Check for example image
    example_img = Path("src/Example.jpg")
    if example_img.exists():
        print(f"✅ {example_img} exists (for testing)")
    else:
        print(f"⚠️  {example_img} not found (optional test image)")
    
    return all_files_exist

def test_model_loading():
    """Test actual model loading with weights"""
    print("\n" + "=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    model_path = Path("models/model002.pt")
    config_path = Path("config/config.json")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        import torch
        from refine_net import EnhancedRefineNet
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        model = EnhancedRefineNet(config)
        
        # Load weights
        device = torch.device('cpu')  # Use CPU for compatibility
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Clean up weight names
        cleaned_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith('module.'):
                name = name[7:]
            cleaned_state_dict[name] = param
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(
            cleaned_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"⚠️  Missing keys in model: {len(missing_keys)} keys")
            if len(missing_keys) <= 5:  # Show first few
                for key in missing_keys[:5]:
                    print(f"     - {key}")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 5:  # Show first few
                for key in unexpected_keys[:5]:
                    print(f"     - {key}")
        
        print("✅ Model weights loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        return False

def test_basic_inference():
    """Test basic inference functionality"""
    print("\n" + "=" * 60)
    print("TESTING BASIC INFERENCE")
    print("=" * 60)
    
    try:
        import torch
        import numpy as np
        from refine_net import EnhancedRefineNet
        
        # Load config
        config_path = Path("config/config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EnhancedRefineNet(config)
            model.eval()
        
        # Create dummy input
        image_size = config['data_pipeline']['image_size']
        dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
        
        print(f"   Input shape: {dummy_input.shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(dummy_input)
        
        # Check outputs
        expected_keys = ['fluid_mask', 'slag_mask', 'quality']
        for key in expected_keys:
            if key in outputs:
                print(f"   ✅ {key}: {outputs[key].shape}")
            else:
                print(f"   ❌ Missing output: {key}")
                return False
        
        # Check output shapes
        h, w = image_size
        if outputs['fluid_mask'].shape == (1, 1, h, w):
            print("   ✅ Fluid mask shape correct")
        else:
            print(f"   ❌ Fluid mask shape incorrect: {outputs['fluid_mask'].shape}")
            return False
        
        if outputs['slag_mask'].shape == (1, 1, h, w):
            print("   ✅ Slag mask shape correct")
        else:
            print(f"   ❌ Slag mask shape incorrect: {outputs['slag_mask'].shape}")
            return False
        
        if outputs['quality'].shape == (1, 3):
            print("   ✅ Quality output shape correct")
        else:
            print(f"   ❌ Quality output shape incorrect: {outputs['quality'].shape}")
            return False
        
        print("✅ Basic inference test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic inference failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        return False

def main():
    """Run all validation tests"""
    print("FLUID ANALYSIS APPLICATION - VALIDATION REPORT")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    print()
    
    # Track test results
    test_results = {}
    
    # Run all tests
    test_results['imports'] = test_imports()
    test_results['config_valid'], config = test_configuration()
    test_results['file_paths'] = test_file_paths()
    test_results['model_architecture'] = test_model_architecture(config)
    test_results['model_loading'] = test_model_loading()
    test_results['basic_inference'] = test_basic_inference()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper().replace('_', ' '):<20}: {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Application is ready for deployment!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Issues need to be addressed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)