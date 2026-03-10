import os
import argparse
import logging
import time
import numpy as np
import onnxruntime as ort
import torchvision.transforms as T

import config
from src.utils import setup_logging, load_image_safely


# ------------------------------------------------------------
# Single image prediction
# ------------------------------------------------------------

def predict_single_image_onnx(ort_session, image_path, image_transforms):
    start_time = time.time()

    try:
        image = load_image_safely(image_path)
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        return

    # IMPORTANT:
    # INT8 ONNX models STILL expect FP32 input tensors
    input_tensor = (
        image_transforms(image)
        .unsqueeze(0)
        .numpy()
        .astype(np.float32)
    )

    ort_inputs = {
        ort_session.get_inputs()[0].name: input_tensor
    }

    ort_outputs = ort_session.run(None, ort_inputs)

    output = ort_outputs[0]  # NumPy array
    predicted_class = int(np.argmax(output, axis=1)[0])
    result = config.CLASS_MAP[predicted_class]

    duration = time.time() - start_time

    print(
        f"-> Image: '{os.path.basename(image_path)}' | "
        f"Prediction: Code={predicted_class}. {result} "
        f"(Took {duration:.4f} seconds)"
    )


# ------------------------------------------------------------
# Main ONNX inference routine
# ------------------------------------------------------------

def run_prediction_onnx(args):
    setup_logging()

    if not os.path.exists(args.model_path):
        logging.error(f"ONNX model not found: {args.model_path}")
        return

    # Same transforms used during training / validation
    image_transforms = T.Compose(
        [
            T.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
            T.CenterCrop(config.IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # --------------------------------------------------------
    # CRITICAL:
    # INT8 ConvInteger ops ONLY work on CPUExecutionProvider
    # --------------------------------------------------------
    providers = ["CPUExecutionProvider"]

    try:
        available_providers = ort.get_available_providers()
        logging.info(f"Available ONNX Runtime providers: {available_providers}")
        logging.info("INT8 model detected — forcing CPUExecutionProvider")

        ort_session = ort.InferenceSession(
            args.model_path,
            providers=providers,
        )

        logging.info(
            f"Model loaded successfully using providers: {ort_session.get_providers()}"
        )

    except Exception as e:
        logging.error(f"Failed to load ONNX model: {e}")
        logging.error(
            "INT8 models with ConvInteger require CPUExecutionProvider "
            "(or TensorRT / OpenVINO)."
        )
        return

    # --------------------------------------------------------
    # Input handling
    # --------------------------------------------------------
    input_path = args.input_path

    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        predict_single_image_onnx(
            ort_session, input_path, image_transforms
        )

    elif os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")

        image_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            logging.warning(f"No images found in directory: {input_path}")
            return

        dir_start = time.time()

        for image_file in image_files:
            full_path = os.path.join(input_path, image_file)
            predict_single_image_onnx(
                ort_session, full_path, image_transforms
            )

        total_time = time.time() - dir_start
        print(
            f"Finished {len(image_files)} images in "
            f"{total_time:.4f} seconds"
        )

    else:
        logging.error(f"Invalid input path: {input_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":

    print(f"ONNX Runtime version: {ort.__version__}")
    
    parser = argparse.ArgumentParser(
        description="Predict image orientation using an INT8 ONNX model."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to an image file or directory of images",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(
            config.MODEL_SAVE_DIR,
            f"{config.MODEL_NAME}_quant_int8.onnx",
        ),
        help="Path to INT8 ONNX model",
    )

    args = parser.parse_args()
    run_prediction_onnx(args)