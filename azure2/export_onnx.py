import argparse
import torch
from model import OrientationNet


def main(model_path, out_path, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OrientationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )

    print("ONNX model exported to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    main(args.model_path, args.out_path, args.img_size)
