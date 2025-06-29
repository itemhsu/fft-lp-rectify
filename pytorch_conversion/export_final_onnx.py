"""
Final ONNX Export Script for TaiLPRectFFTModel
Working version with opset 20 and ONNX-compatible operations
"""
pass
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import os
import json
from datetime import datetime
import sys
import warnings
pass
pass
warnings.filterwarnings("ignore")
pass
pass
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.append(current_dir)
pass
from tai_lprect_model import TaiLPRectFFTModel
pass
pass
class ONNXCompatibleTaiLPRectFFTModel(TaiLPRectFFTModel):
    """
    ONNX-compatible version that replaces torch.rot90 with compatible operations
    """
    
    def rot90_compatible(self, tensor, k=1, dims=[-2, -1]):
        """Replace torch.rot90 with ONNX-compatible operations"""
        k = k % 4
        if k == 0:
            return tensor
        elif k == 1:  # 90 degrees
            tensor = torch.transpose(tensor, dims[0], dims[1])
            return torch.flip(tensor, [dims[1]])
        elif k == 2:  # 180 degrees
            return torch.flip(torch.flip(tensor, [dims[0]]), [dims[1]])
        elif k == 3:  # 270 degrees
            tensor = torch.flip(tensor, [dims[0]])
            return torch.transpose(tensor, dims[0], dims[1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ONNX-compatible rot90 replacement"""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            img = x[i]
            
            pass
            h_corrected = self.rectify_single_direction(img)
            
            pass
            h_corrected_90 = self.rot90_compatible(h_corrected, k=1, dims=[-2, -1])
            v_corrected = self.rectify_single_direction(h_corrected_90)
            
            pass
            final_corrected = self.rot90_compatible(v_corrected, k=3, dims=[-2, -1])
            
            results.append(final_corrected)
        
        return torch.stack(results, dim=0)
pass
pass
def create_working_model():
    """Create the working ONNX-compatible model"""
    return ONNXCompatibleTaiLPRectFFTModel(
        cutoff_f=0.8,
        margin=0.1, 
        polar_bins=100,
        device=torch.device('cpu')
    )


def export_final_onnx():
    """Export the final working ONNX model"""
    print("=== Final ONNX Export ===")
    
    try:
        pass
        model = create_working_model()
        model.eval()
        print("✅ Model created and set to eval mode")
        
        pass
        test_input = torch.randn(1, 3, 256, 256, device='cpu')
        
        pass
        print("Testing forward pass...")
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Forward pass successful: {test_input.shape} -> {output.shape}")
        
        pass
        export_path = os.path.join(current_dir, "tai_lprect_final_working.onnx")
        print(f"Exporting to: {export_path}")
        
        torch.onnx.export(
            model,
            test_input,
            export_path,
            export_params=True,
            opset_version=20,  # Required for affine_grid_generator
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['output_image'],
            dynamic_axes={
                'input_image': {0: 'batch_size'},
                'output_image': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print("✅ ONNX export successful!")
        
        pass
        print("Validating ONNX model...")
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation successful!")
        
        pass
        print("Testing with ONNX Runtime...")
        ort_session = ort.InferenceSession(export_path)
        
        ort_inputs = {'input_image': test_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        pass
        pytorch_output = output.numpy()
        onnx_output = ort_outputs[0]
        
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        
        print(f"✅ ONNX Runtime test successful!")
        print(f"   Max difference: {max_diff:.8f}")
        print(f"   Mean difference: {mean_diff:.8f}")
        
        pass
        results = {
            'timestamp': datetime.now().isoformat(),
            'export_path': export_path,
            'input_shape': list(test_input.shape),
            'output_shape': list(output.shape),
            'opset_version': 20,
            'validation': {
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'success': True
            }
        }
        
        with open(os.path.join(current_dir, 'final_export_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 SUCCESS! ONNX model exported and validated:")
        print(f"   📄 File: {export_path}")
        print(f"   📊 Input shape: {test_input.shape}")
        print(f"   📊 Output shape: {output.shape}")
        print(f"   🔧 Opset version: 20")
        print(f"   ✨ Model differences: max={max_diff:.8f}, mean={mean_diff:.8f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
pass
pass
if __name__ == "__main__":
    success = export_final_onnx()
    if success:
        print("\n🎉 Final ONNX export completed successfully!")
    else:
        print("\n❌ Final ONNX export failed!")
