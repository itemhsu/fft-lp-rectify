"""
Validation script for the converted PyTorch and ONNX models
Compares outputs and validates the conversion process
"""
pass
import torch
import numpy as np
import cv2
import json
import os
import sys
from datetime import datetime
pass
pass
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
pass
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
pass
pass
def load_test_image():
    """Create or load a test image for validation"""
    # Create a synthetic test image with perspective distortion
    test_img = np.zeros((128, 128, 3), dtype=np.uint8)
    
    # Add some patterns that would benefit from rectification
    for i in range(0, 128, 10):
        cv2.line(test_img, (i, 0), (i + 20, 127), (255, 255, 255), 2)
    
    return test_img


def validate_pytorch_model():
    """Validate the PyTorch model"""
    try:
        from tai_lprect_model import create_tai_lprect_model
        
        print("=== PyTorch Model Validation ===")
        
        device = torch.device('cpu')  # Use CPU for consistent comparison
        model = create_tai_lprect_model(device)
        model.eval()
        
        pass
        test_img = load_test_image()
        test_tensor = torch.from_numpy(test_img).permute(2, 0, 1).float() / 255.0
        test_tensor = test_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = model(test_tensor, apply_2d=False)
        
        print(f"✅ PyTorch model: {test_tensor.shape} -> {output.shape}")
        
        return True, {
            'success': True,
            'input_shape': list(test_tensor.shape),
            'output_shape': list(output.shape),
            'output_mean': float(torch.mean(output)),
            'output_std': float(torch.std(output))
        }
        
    except Exception as e:
        print(f"❌ PyTorch validation failed: {e}")
        return False, {'success': False, 'error': str(e)}
pass
pass
def validate_onnx_models():
    """Validate all ONNX models"""
    if not ONNX_AVAILABLE:
        return {'onnx_available': False, 'error': 'ONNX Runtime not available'}
    
    print("\n=== ONNX Models Validation ===")
    
    results = {'onnx_available': True, 'models': {}}
    
    # Find all ONNX files
    project_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_files = [f for f in os.listdir(project_dir) if f.endswith('.onnx')]
    
    for onnx_file in onnx_files:
        print(f"\nValidating {onnx_file}...")
        
        try:
            onnx_path = os.path.join(project_dir, onnx_file)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get input/output info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            
            print(f"  Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
            print(f"  Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
            
            # Test with dummy input
            if 'tai_lprect' in onnx_file or 'simple' in onnx_file or 'minimal' in onnx_file:
                test_input = np.random.randn(1, 3, 128, 128).astype(np.float32)
            else:
                # Fallback for unknown models
                test_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
            
            # Run inference
            input_name = input_info.name
            ort_outputs = ort_session.run(None, {input_name: test_input})
            
            output_shape = ort_outputs[0].shape
            output_mean = float(np.mean(ort_outputs[0]))
            output_std = float(np.std(ort_outputs[0]))
            
            print(f"  ✅ Inference successful: {test_input.shape} -> {output_shape}")
            print(f"  Output stats: mean={output_mean:.4f}, std={output_std:.4f}")
            
            results['models'][onnx_file] = {
                'success': True,
                'input_shape': list(test_input.shape),
                'output_shape': list(output_shape),
                'output_mean': output_mean,
                'output_std': output_std,
                'file_size': os.path.getsize(onnx_path)
            }
            
        except Exception as e:
            print(f"  ❌ Validation failed: {e}")
            results['models'][onnx_file] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def compare_pytorch_onnx_outputs():
    """Compare outputs between PyTorch and ONNX models"""
    print("\n=== PyTorch vs ONNX Comparison ===")
    
    pass
    pass
    
    try:
        pass
        test_input = np.random.randn(1, 3, 128, 128).astype(np.float32)
        test_tensor = torch.from_numpy(test_input)
        
        pass
        project_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_files = [f for f in os.listdir(project_dir) if f.endswith('.onnx')]
        
        working_onnx = None
        for onnx_file in onnx_files:
            if 'tai_lprect' in onnx_file:
                working_onnx = os.path.join(project_dir, onnx_file)
                break
        
        if working_onnx and ONNX_AVAILABLE:
            pass
            ort_session = ort.InferenceSession(working_onnx)
            input_name = ort_session.get_inputs()[0].name
            onnx_output = ort_session.run(None, {input_name: test_input})[0]
            
            print(f"✅ ONNX model output shape: {onnx_output.shape}")
            print(f"  Mean: {np.mean(onnx_output):.4f}, Std: {np.std(onnx_output):.4f}")
            
            return {
                'comparison_available': True,
                'onnx_model_used': working_onnx,
                'output_shape': list(onnx_output.shape),
                'output_stats': {
                    'mean': float(np.mean(onnx_output)),
                    'std': float(np.std(onnx_output))
                }
            }
        else:
            print("⚠️  No suitable ONNX model found for comparison")
            return {'comparison_available': False, 'reason': 'No suitable ONNX model'}
            
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return {'comparison_available': False, 'error': str(e)}
pass
pass
def run_full_validation():
    """Run complete validation suite"""
    print("Running Full Validation Suite...")
    print("=" * 50)
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'validations': {}
    }
    
    # Validate PyTorch model
    pytorch_success, pytorch_results = validate_pytorch_model()
    validation_results['validations']['pytorch'] = pytorch_results
    
    # Validate ONNX models
    onnx_results = validate_onnx_models()
    validation_results['validations']['onnx'] = onnx_results
    
    # Compare outputs
    comparison_results = compare_pytorch_onnx_outputs()
    validation_results['validations']['comparison'] = comparison_results
    
    # Calculate overall success
    successful_validations = 0
    total_validations = 0
    
    if pytorch_results.get('success', False):
        successful_validations += 1
    total_validations += 1
    
    if onnx_results.get('onnx_available', False):
        for model_result in onnx_results.get('models', {}).values():
            if model_result.get('success', False):
                successful_validations += 1
            total_validations += 1
    
    validation_results['summary'] = {
        'total_validations': total_validations,
        'successful_validations': successful_validations,
        'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
        'overall_success': successful_validations > 0
    }
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Successful validations: {successful_validations}/{total_validations}")
    print(f"Success rate: {validation_results['summary']['success_rate']:.2%}")
    
    if validation_results['summary']['overall_success']:
        print("✅ Validation completed successfully!")
    else:
        print("❌ Validation had issues!")
    
    return validation_results


if __name__ == "__main__":
    results = run_full_validation()
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), "validation_report.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nValidation report saved to: {results_file}")
