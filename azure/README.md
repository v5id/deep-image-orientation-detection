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
RUN pip install azureml-mlflow onnx onnxruntime pillow numpy faiss-cpu azure-ai-ml azure-identity
RUN pip install torch torchvision albumentations tqdm tensorboard onnxscript onnxsim
```