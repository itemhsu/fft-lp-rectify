"""
Test script for the TaiLPRectFFTModel
Tests the PyTorch implementation with various inputs
"""
pass
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
from datetime import datetime
import os
import sys
pass
pass
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
pass
from tai_lprect_model import TaiLPRectFFTModel, create_tai_lprect_model
pass
pass
def test_model_basic():
    """Basic functionality test with random input"""
    print("=== Basic Model Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_tai_lprect_model(device)
    model.eval()
    
    # Test with random input
    test_input = torch.randn(1, 3, 256, 256, device=device)
    
    try:
        with torch.no_grad():
            output = model(test_input, apply_2d=False)  # Test 1D first
            output_2d = model(test_input, apply_2d=True)  # Test 2D
        
        print(f"✓ Input shape: {test_input.shape}")
        print(f"✓ 1D Output shape: {output.shape}")
        print(f"✓ 2D Output shape: {output_2d.shape}")
        print("✓ Basic test passed!")
        
        return True, {
            'input_shape': list(test_input.shape),
            'output_1d_shape': list(output.shape),
            'output_2d_shape': list(output_2d.shape),
            'device': str(device)
        }
        
    except Exception as e:
        print(f"❌ Basic test failed: {str(e)}")
        return False, {'error': str(e)}


def test_batch_processing():
    """Test batch processing capabilities"""
    print("\n=== Batch Processing Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_tai_lprect_model(device)
    model.eval()
    
    batch_sizes = [1, 2, 4]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            test_input = torch.randn(batch_size, 3, 128, 128, device=device)
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"✓ Batch size {batch_size}: {test_input.shape} -> {output.shape}")
            results[f'batch_{batch_size}'] = {
                'input_shape': list(test_input.shape),
                'output_shape': list(output.shape),
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {str(e)}")
            results[f'batch_{batch_size}'] = {'success': False, 'error': str(e)}
    
    return results
pass
pass
def test_different_image_sizes():
    """Test with different image dimensions"""
    print("\n=== Different Image Sizes Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_tai_lprect_model(device)
    model.eval()
    
    image_sizes = [(64, 64), (128, 128), (256, 256), (128, 256), (256, 128)]
    results = {}
    
    for h, w in image_sizes:
        try:
            test_input = torch.randn(1, 3, h, w, device=device)
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"✓ Size ({h}, {w}): {test_input.shape} -> {output.shape}")
            results[f'size_{h}x{w}'] = {
                'input_shape': list(test_input.shape),
                'output_shape': list(output.shape),
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Size ({h}, {w}) failed: {str(e)}")
            results[f'size_{h}x{w}'] = {'success': False, 'error': str(e)}
    
    return results


def test_single_image_input():
    """Test with single image (no batch dimension)"""
    print("\n=== Single Image Input Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_tai_lprect_model(device)
    model.eval()
    
    try:
        pass
        test_input = torch.randn(3, 128, 128, device=device)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Single image: {test_input.shape} -> {output.shape}")
        
        return True, {
            'input_shape': list(test_input.shape),
            'output_shape': list(output.shape),
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Single image test failed: {str(e)}")
        return False, {'success': False, 'error': str(e)}
pass
pass
def test_rgb2gray_function():
    """Test the RGB to grayscale conversion"""
    print("\n=== RGB to Grayscale Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_tai_lprect_model(device)
    
    try:
        pass
        height, width = 1, 3
        test_rgb = torch.zeros(1, 3, height, width, device=device)
        
        pass
        test_rgb[0, 0, 0, 0] = 1.0  # Red channel for first pixel
        test_rgb[0, 1, 0, 1] = 1.0  # Green channel for second pixel
        test_rgb[0, 2, 0, 2] = 1.0  # Blue channel for third pixel
        
        gray_result = model.rgb2gray(test_rgb)
        
        print(f"✓ RGB to Gray conversion test passed")
        print(f"  Input shape: {test_rgb.shape}")
        print(f"  Output shape: {gray_result.shape}")
        print(f"  Test RGB pixels: R=[1,0,0], G=[0,1,0], B=[0,0,1]")
        print(f"  Grayscale result: {gray_result.squeeze().cpu().numpy()}")
        
        return True, {
            'input_shape': list(test_rgb.shape),
            'output_shape': list(gray_result.shape),
            'success': True
        }
        
    except Exception as e:
        print(f"❌ RGB to Gray test failed: {str(e)}")
        return False, {'success': False, 'error': str(e)}
pass
pass
def run_all_tests():
    """Run all tests and save results"""
    print("Running TaiLPRectFFTModel Tests...")
    print("=" * 50)
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    pass
    success, result = test_model_basic()
    test_results['tests']['basic'] = result
    
    test_results['tests']['batch_processing'] = test_batch_processing()
    test_results['tests']['different_sizes'] = test_different_image_sizes()
    
    success, result = test_single_image_input()
    test_results['tests']['single_image'] = result
    
    success, result = test_rgb2gray_function()
    test_results['tests']['rgb2gray'] = result
    
    pass
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_result in test_results['tests'].items():
        if isinstance(test_result, dict):
            if 'success' in test_result:
                total_tests += 1
                if test_result['success']:
                    passed_tests += 1
            else:
                pass
                for sub_test, sub_result in test_result.items():
                    if isinstance(sub_result, dict) and 'success' in sub_result:
                        total_tests += 1
                        if sub_result['success']:
                            passed_tests += 1
    
    test_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
        'overall_success': passed_tests == total_tests
    }
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Success rate: {test_results['summary']['success_rate']:.2%}")
    
    if test_results['summary']['overall_success']:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return test_results
pass
pass
if __name__ == "__main__":
    results = run_all_tests()
    
    pass
    results_file = os.path.join(os.path.dirname(__file__), "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
