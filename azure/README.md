Use:
```
python train.py --data_dir ${{inputs.orientation_detection_data}} --model_dir ${{inputs.orientation_detection_model}}
```

Use docker file:
```
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# Install build tools needed for torch.compile
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install azureml-mlflow onnx onnxruntime pillow "numpy<2" faiss-cpu azure-ai-ml azure-identity
RUN pip install torchvision==0.18.0 albumentations tqdm tensorboard onnxscript onnxsim
```


```
python export_onnx.py --model_path ./models/orientation_model_best.pth --out_path ./models/orientation_model_best.onnx
```

## Quantization

Slim model first:
```
python -m onnxsim ./models/orientation_model_best.onnx ./models/best_model_slim.onnx
```

To descrease model size and be used on the edge devices use
```
python .\quantize_to_onnx.py .\models\best_model_slim.onnx --calib_dir .\data\upright_images  --op_types MatMul,Gemm
```  

## Quantized Image Prediction
To predict the orientation of an image or a directory of images, there's a `predict.py` script.

- **Predict a single image:**

  ```bash
  python predict_onnx_int8.py --input_path /path/to/image.jpg --model_path ./models/best_model_slim_quant_int8.onnx
  ```