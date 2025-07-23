#!/usr/bin/env python3
"""
Test inference functionality without Streamlit UI
"""

import sys
import json
import numpy as np
from pathlib import Path
import cv2

# Add src to path
sys.path.append('src')

def test_inference_model():
    """Test the InferenceModel class directly"""
    print("Testing InferenceModel class...")
    
    try:
        from inference_app import InferenceModel
        
        # Initialize model
        model_path = "models/model002.pt"
        config_path = "config/config.json"
        
        print("Initializing inference model...")
        inference_model = InferenceModel(model_path, config_path)
        print("✅ InferenceModel initialized successfully")
        
        # Test with example image if available
        example_img_path = Path("src/Example.jpg")
        if example_img_path.exists():
            print("\nTesting with example image...")
            
            # Load image
            image = cv2.imread(str(example_img_path))
            if image is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                print(f"Image shape: {image_rgb.shape}")
                
                # Run prediction
                print("Running prediction...")
                predictions = inference_model.predict(image_rgb)
                
                print("✅ Prediction completed successfully")
                print("Prediction results:")
                print(f"  - Fluid mask shape: {predictions['fluid_mask'].shape}")
                print(f"  - Slag mask shape: {predictions['slag_mask'].shape}")
                print(f"  - Quality label: {predictions['quality']['label']}")
                print(f"  - Quality readability: {predictions['quality']['readability']:.3f}")
                print(f"  - Quality probabilities: {predictions['quality']['probabilities']}")
                
                # Check mask values
                fluid_count = np.sum(predictions['fluid_mask'] > 0.5)
                slag_count = np.sum(predictions['slag_mask'] > 0.5)
                print(f"  - Fluid pixels detected: {fluid_count}")
                print(f"  - Slag pixels detected: {slag_count}")
                
                return True
            else:
                print("❌ Could not load example image")
                return False
        else:
            print("⚠️  Example image not found, testing with synthetic image...")
            
            # Create synthetic test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            print(f"Synthetic image shape: {test_image.shape}")
            
            # Run prediction
            predictions = inference_model.predict(test_image)
            print("✅ Prediction on synthetic image completed")
            print("Prediction results:")
            print(f"  - Quality label: {predictions['quality']['label']}")
            print(f"  - Quality readability: {predictions['quality']['readability']:.3f}")
            
            return True
            
    except Exception as e:
        print(f"❌ InferenceModel test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_helper_functions():
    """Test helper functions from inference_app"""
    print("\nTesting helper functions...")
    
    try:
        from inference_app import (
            extract_roi_from_frame, 
            calculate_fluid_level_metrics,
            get_available_models
        )
        
        # Test get_available_models
        available_models = get_available_models()
        print(f"✅ Available models: {available_models}")
        
        # Test ROI extraction
        test_frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        roi_coords = (100, 50, 400, 300)  # x, y, w, h
        roi_frame = extract_roi_from_frame(test_frame, roi_coords)
        expected_shape = (300, 400, 3)
        if roi_frame.shape == expected_shape:
            print(f"✅ ROI extraction works correctly: {roi_frame.shape}")
        else:
            print(f"❌ ROI extraction shape mismatch: got {roi_frame.shape}, expected {expected_shape}")
            return False
        
        # Test fluid level calculation
        test_mask = np.zeros((200, 300))
        test_mask[150:180, 50:250] = 1.0  # Add some "fluid" pixels
        metrics = calculate_fluid_level_metrics(test_mask)
        print(f"✅ Fluid level metrics: {metrics}")
        
        if 'height' in metrics and 'mass' in metrics:
            print("✅ Fluid level calculation works correctly")
        else:
            print("❌ Fluid level calculation missing keys")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Helper functions test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run inference tests"""
    print("INFERENCE FUNCTIONALITY TEST")
    print("=" * 50)
    
    results = {}
    
    # Test core inference functionality
    results['inference_model'] = test_inference_model()
    results['helper_functions'] = test_helper_functions()
    
    # Summary
    print("\n" + "=" * 50)
    print("INFERENCE TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper().replace('_', ' '):<20}: {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL INFERENCE TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)