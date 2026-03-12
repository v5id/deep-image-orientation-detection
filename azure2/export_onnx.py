import torch
from model import OrientationNet


model = OrientationNet()

model.load_state_dict(torch.load("orientation_model.pth"))

model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "orientation_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)

print("ONNX model exported")