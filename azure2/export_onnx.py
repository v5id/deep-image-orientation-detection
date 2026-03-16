import argparse
import os
from pathlib import Path

import torch
from model import OrientationNet


def _make_single_file_onnx(out_path: str) -> None:
    """
    If the exported ONNX uses external data (e.g. writes an extra *.data file),
    re-save it as a single ONNX file with embedded weights.
    """
    try:
        import onnx
        from onnx.external_data_helper import convert_model_from_external_data, uses_external_data
    except Exception:
        # ONNX is an explicit dependency in requirements.txt, but keep export usable if it's missing.
        return

    out_path = str(out_path)
    base_dir = os.path.dirname(os.path.abspath(out_path))

    try:
        model = onnx.load_model(out_path, load_external_data=True)
    except Exception:
        return

    external_locations: set[str] = set()
    for initializer in list(model.graph.initializer):
        if uses_external_data(initializer):
            for kv in initializer.external_data:
                if kv.key == "location" and kv.value:
                    external_locations.add(kv.value)

    if not external_locations and not os.path.exists(out_path + ".data"):
        return

    # Convert to embedded tensor data and write atomically.
    tmp_path = out_path + ".tmp.onnx"
    try:
        convert_model_from_external_data(model)
        onnx.save_model(model, tmp_path)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return

    # Best-effort cleanup of external data files referenced by the model.
    for location in external_locations:
        try:
            resolved = os.path.abspath(os.path.join(base_dir, location))
            if os.path.commonpath([base_dir, resolved]) != base_dir:
                continue
            if os.path.exists(resolved):
                os.remove(resolved)
        except Exception:
            pass

    # Some exporters also create a conventional sibling "<model>.onnx.data".
    try:
        data_path = out_path + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
    except Exception:
        pass


def main(model_path, out_path, img_size, opset_version):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Avoid downloading ImageNet weights during export; the checkpoint provides all weights.
    model = OrientationNet(weights=None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    # Torch 2.6+ defaults to the new (dynamo) exporter; force legacy mode for more
    # stable behavior across environments.
    export_kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset_version,
    )
    try:
        try:
            torch.onnx.export(model, dummy, out_path, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, out_path, **export_kwargs)
    except Exception as e:
        # Fallback to the dynamo exporter if legacy export fails.
        print(f"Legacy ONNX export failed ({type(e).__name__}: {e}); retrying with dynamo exporter...")
        try:
            torch.onnx.export(model, dummy, out_path, dynamo=True, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, out_path, **export_kwargs)

    _make_single_file_onnx(out_path)

    print("ONNX model exported to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--opset_version", type=int, default=12)
    args = parser.parse_args()
    out_path = str(Path(args.out_path))
    main(args.model_path, out_path, args.img_size, args.opset_version)
