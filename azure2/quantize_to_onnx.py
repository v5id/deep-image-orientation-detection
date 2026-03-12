import argparse
import os
import random

import numpy as np
from PIL import Image, ImageOps


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _letterbox(image, size, fill=(255, 255, 255)):
    size = int(size)
    w, h = image.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), fill)

    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = image.resize((nw, nh), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), fill)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _preprocess_image(image_path, img_size, rotation_mode, rng):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    if rotation_mode == "all":
        # handled outside: caller repeats 4 rotations
        pass
    elif rotation_mode == "random":
        angle = rng.choice([0, 90, 180, 270])
        if angle != 0:
            image = image.rotate(angle, expand=True)
    elif rotation_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")

    image = _letterbox(image, img_size)

    arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC, 0..1
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, axis=0)  # NCHW
    return arr.astype(np.float32)


def _list_images(calib_dir):
    files = []
    for f in os.listdir(calib_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(calib_dir, f))
    files.sort()
    return files


def _get_model_input_name(model_path):
    import onnx

    model = onnx.load(model_path)
    initializer_names = {i.name for i in model.graph.initializer}
    for i in model.graph.input:
        if i.name not in initializer_names:
            return i.name
    # fallback: first declared input
    return model.graph.input[0].name


def _print_size_stats(input_onnx_path, output_onnx_path):
    original_size = os.path.getsize(input_onnx_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Original size: {original_size:.2f} MB")
    print(f"Output size:   {quantized_size:.2f} MB")
    print(f"Saved to: {output_onnx_path}")


def quantize_dynamic_onnx(input_onnx_path: str, output_onnx_path: str):
    """
    Dynamic quantization (no calibration set).

    Note: For CNNs (MobileNet, etc.) dynamic quantization often gives limited gains and/or accuracy drops.
    Prefer quantize_static_onnx with a representative calibration set.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        weight_type=QuantType.QInt8,
    )

    print("Dynamic INT8 quantization complete.")
    _print_size_stats(input_onnx_path, output_onnx_path)


def quantize_static_onnx(
    input_onnx_path: str,
    output_onnx_path: str,
    *,
    calib_dir: str,
    img_size: int,
    calib_count: int,
    seed: int,
    rotation_mode: str,
    per_channel: bool,
    quant_format: str,
    calibrate_method: str,
    op_types: list[str],
    nodes_to_exclude: list[str] | None,
    weight_symmetric: bool,
    activation_symmetric: bool,
):
    """
    Static (calibrated) INT8 quantization.

    This is the recommended path for MobileNet-like CNNs when you care about accuracy.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    # CalibrationMethod is optional across ORT versions; import defensively.
    try:
        from onnxruntime.quantization import CalibrationMethod  # type: ignore
    except Exception:  # pragma: no cover
        CalibrationMethod = None

    input_name = _get_model_input_name(input_onnx_path)

    image_paths = _list_images(calib_dir)
    if not image_paths:
        raise RuntimeError(f"No calibration images found in: {calib_dir}")

    rng = random.Random(seed)
    rng.shuffle(image_paths)

    if calib_count > 0:
        image_paths = image_paths[: min(calib_count, len(image_paths))]

    class ImageFolderCalibrationDataReader(CalibrationDataReader):

        def __init__(self):
            self._iter = None
            self._data = []

            if rotation_mode == "all":
                for p in image_paths:
                    for angle in [0, 90, 180, 270]:
                        img = Image.open(p)
                        img = ImageOps.exif_transpose(img).convert("RGB")
                        if angle != 0:
                            img = img.rotate(angle, expand=True)
                        img = _letterbox(img, img_size)
                        arr = np.asarray(img, dtype=np.float32) / 255.0
                        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                        arr = np.transpose(arr, (2, 0, 1))
                        arr = np.expand_dims(arr, axis=0).astype(np.float32)
                        self._data.append({input_name: arr})
            else:
                for p in image_paths:
                    arr = _preprocess_image(p, img_size, rotation_mode, rng)
                    self._data.append({input_name: arr})

        def get_next(self):
            if self._iter is None:
                self._iter = iter(self._data)
            return next(self._iter, None)

    format_map = {
        "qdq": QuantFormat.QDQ,
        "qoperator": QuantFormat.QOperator,
    }
    if quant_format not in format_map:
        raise ValueError("quant_format must be 'qdq' or 'qoperator'")

    method_map = None
    if CalibrationMethod is not None:
        method_map = {"minmax": CalibrationMethod.MinMax}
        if hasattr(CalibrationMethod, "Entropy"):
            method_map["entropy"] = CalibrationMethod.Entropy

    if method_map is None:
        if calibrate_method != "minmax":
            raise ValueError(
                "This onnxruntime version does not support entropy calibration. Use --calibrate_method minmax."
            )
    else:
        if calibrate_method not in method_map:
            print(
                f"Warning: calibrate_method='{calibrate_method}' not supported by this onnxruntime. Falling back to 'minmax'."
            )
            calibrate_method = "minmax"

    data_reader = ImageFolderCalibrationDataReader()

    quantize_kwargs = {
        "model_input": input_onnx_path,
        "model_output": output_onnx_path,
        "calibration_data_reader": data_reader,
        "quant_format": format_map[quant_format],
        "per_channel": per_channel,
        "activation_type": QuantType.QUInt8,
        "weight_type": QuantType.QInt8,
        "op_types_to_quantize": op_types,
        "extra_options": {
            # These defaults are generally good for ReLU-heavy CNNs.
            "WeightSymmetric": bool(weight_symmetric),
            "ActivationSymmetric": bool(activation_symmetric),
        },
    }

    if method_map is not None:
        quantize_kwargs["calibrate_method"] = method_map[calibrate_method]

    if nodes_to_exclude:
        quantize_kwargs["nodes_to_exclude"] = nodes_to_exclude

    quantize_static(**quantize_kwargs)

    print("Static INT8 quantization complete.")
    _print_size_stats(input_onnx_path, output_onnx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize an ONNX model to INT8. Use static quantization with calibration for best accuracy."
    )
    parser.add_argument("input_onnx", type=str, help="Path to the input FP32 ONNX model")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: input + '_quant_int8.onnx')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["static", "dynamic"],
        default="static",
        help="Quantization mode. 'static' is recommended for CNNs.",
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        default=None,
        help="Directory of representative images used for calibration (recommended for --mode static).",
    )
    parser.add_argument(
        "--calib_count",
        type=int,
        default=0,
        help="Max number of calibration images to use (0 = all).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Model input size (must match export/training size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used for randomized calibration rotations.",
    )
    parser.add_argument(
        "--calib_rotations",
        type=str,
        choices=["random", "all", "none"],
        default="all",
        help="Rotation augmentation for calibration images. Use 'random' or 'all' if inference sees all orientations.",
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Enable per-channel weight quantization.",
    )
    parser.add_argument(
        "--per_tensor",
        action="store_true",
        help="Force per-tensor weights (may reduce accuracy but can improve compatibility).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["qdq", "qoperator"],
        default="qdq",
        help="Quant format. 'qdq' is typically most compatible.",
    )
    parser.add_argument(
        "--calibrate_method",
        type=str,
        choices=["minmax", "entropy"],
        default="entropy",
        help="Calibration method.",
    )
    parser.add_argument(
        "--op_types",
        type=str,
        default="Conv,MatMul,Gemm",
        help="Comma-separated ONNX op types to quantize (e.g. 'Conv,MatMul,Gemm').",
    )
    parser.add_argument(
        "--nodes_to_exclude",
        type=str,
        default="",
        help="Comma-separated node names to exclude from quantization.",
    )
    parser.add_argument(
        "--no_weight_symmetric",
        action="store_true",
        help="Disable symmetric weight quantization.",
    )
    parser.add_argument(
        "--activation_symmetric",
        action="store_true",
        help="Use symmetric activation quantization.",
    )
    args = parser.parse_args()

    input_path = args.input_onnx
    output_path = args.output or input_path.replace(".onnx", "_quant_int8.onnx")

    if args.mode == "dynamic":
        quantize_dynamic_onnx(input_path, output_path)
        raise SystemExit(0)

    # static mode
    calib_dir = args.calib_dir
    if not calib_dir:
        for candidate in [
            os.path.join(".", "data", "upright_images"),
            os.path.join(".", "data"),
        ]:
            if os.path.isdir(candidate):
                calib_dir = candidate
                print(f"Using inferred calibration directory: {calib_dir}")
                break

    if not calib_dir:
        print(
            "Warning: --calib_dir not provided and no default calibration directory found. "
            "Falling back to dynamic quantization (accuracy may drop)."
        )
        quantize_dynamic_onnx(input_path, output_path)
        raise SystemExit(0)

    if args.per_channel and args.per_tensor:
        raise SystemExit("Error: choose only one of --per_channel or --per_tensor")

    quantize_static_onnx(
        input_path,
        output_path,
        calib_dir=calib_dir,
        img_size=args.img_size,
        calib_count=args.calib_count,
        seed=args.seed,
        rotation_mode=args.calib_rotations,
        per_channel=(not args.per_tensor),
        quant_format=args.format,
        calibrate_method=args.calibrate_method,
        op_types=[t.strip() for t in args.op_types.split(",") if t.strip()],
        nodes_to_exclude=[t.strip() for t in args.nodes_to_exclude.split(",") if t.strip()] or None,
        weight_symmetric=(not bool(args.no_weight_symmetric)),
        activation_symmetric=bool(args.activation_symmetric),
    )
