import argparse
import logging
import os
import time

import numpy as np
import onnxruntime
from PIL import Image, ImageOps


CLASS_MAP = {
    0: "0° (correctly oriented)",
    1: "90° Clockwise",
    2: "180°",
    3: "90° Counter-Clockwise",
}

ROTATE_TO_UPRIGHT = {
    0: 0,
    1: 90,   # rotate CCW
    2: 180,
    3: -90,  # rotate CW
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_image_safely(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def letterbox(image, size, fill=(255, 255, 255)):
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


def preprocess(image, img_size):
    image = letterbox(image, img_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC, 0..1
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, axis=0)  # NCHW
    return arr


def predict_single_image_onnx(ort_session, image_path, img_size):
    start_time = time.time()

    try:
        image = load_image_safely(image_path)
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    input_tensor = preprocess(image, img_size=img_size)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)

    logits = ort_outs[0]
    predicted_class = int(np.argmax(logits, axis=1)[0])

    duration = time.time() - start_time
    result = CLASS_MAP.get(predicted_class, f"Unknown class {predicted_class}")
    rotate_to_upright = ROTATE_TO_UPRIGHT.get(predicted_class, 0)

    print(
        f"-> Image: '{os.path.basename(image_path)}' | Prediction: Code={predicted_class}. {result} | RotateToUpright={rotate_to_upright} (Took {duration:.4f} seconds)"
    )


def _create_session(model_path):
    preferred_providers = [
        "CUDAExecutionProvider",
        "MpsExecutionProvider",
        "ROCmExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    available_providers = onnxruntime.get_available_providers()
    logging.info(f"Available ONNX Runtime providers: {available_providers}")

    chosen_provider = next(
        (p for p in preferred_providers if p in available_providers),
        "CPUExecutionProvider",
    )
    logging.info(f"Loading ONNX model with provider: {chosen_provider}")

    ort_session = onnxruntime.InferenceSession(model_path, providers=[chosen_provider])
    logging.info(
        f"Loaded ONNX model using provider: {ort_session.get_providers()[0]}"
    )
    return ort_session


def run_prediction_onnx(args):
    setup_logging()

    if not os.path.exists(args.model_path):
        logging.error(f"ONNX model file not found at {args.model_path}.")
        return

    try:
        ort_session = _create_session(args.model_path)
    except Exception as e:
        logging.error(f"Error loading ONNX model {args.model_path}: {e}")
        return

    input_path = args.input_path
    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        predict_single_image_onnx(ort_session, input_path, args.img_size)
        return

    if os.path.isdir(input_path):
        print(f"Processing all images in directory: {input_path}")
        total_dir_start_time = time.time()

        image_files = [
            f
            for f in os.listdir(input_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            print(f"No image files found in directory: {input_path}")
            return

        for image_file in image_files:
            full_path = os.path.join(input_path, image_file)
            predict_single_image_onnx(ort_session, full_path, args.img_size)

        total_duration = time.time() - total_dir_start_time
        print(
            f"Finished processing directory '{input_path}'. Total time: {total_duration:.4f} seconds for {len(image_files)} images."
        )
        return

    print(f"Input path is not a valid file or directory: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict image orientation using an ONNX model."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(".", "models", "orientation_model.onnx"),
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Model input size (must match the export/training size).",
    )

    args = parser.parse_args()
    run_prediction_onnx(args)
