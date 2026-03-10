import argparse
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_dynamic_onnx(input_onnx_path: str, output_onnx_path: str):
    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        per_channel=False,          # Key change: avoids ConvInteger issues
        weight_type=QuantType.QInt8,
        # Optional: try QuantType.QUInt8 if you see asymmetry issues (common for ReLU-heavy models)
        # activation_type=QuantType.QUInt8,  # Default is already QUInt8
    )
    original_size = os.path.getsize(input_onnx_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Quantization complete (per-tensor INT8)!")
    print(f"Original FP32 size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Quantized model saved to: {output_onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize an ONNX model to INT8 (dynamic, per-tensor).")
    parser.add_argument("input_onnx", type=str, help="Path to the input FP32 ONNX model")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: input + '_quant_int8.onnx')")
    args = parser.parse_args()

    input_path = args.input_onnx
    output_path = args.output or input_path.replace(".onnx", "_quant_int8.onnx")

    quantize_dynamic_onnx(input_path, output_path)