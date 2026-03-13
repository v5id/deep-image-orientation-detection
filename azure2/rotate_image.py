import argparse
import os
from pathlib import Path

from PIL import Image, ImageOps


def _rotation_degrees(direction: str, degrees: int | None) -> int:
    if degrees is not None:
        return int(degrees)

    direction = direction.lower().strip()
    if direction in ("none", "0", "0deg", "0°"):
        return 0
    if direction in ("cw", "right", "clockwise", "90", "90deg", "90°"):
        return -90  # PIL positive is CCW
    if direction in ("ccw", "left", "counterclockwise", "-90", "-90deg", "-90°"):
        return 90
    if direction in ("180", "180deg", "180°", "flip", "upside-down", "upsidedown"):
        return 180

    raise ValueError(
        "Unknown direction. Use one of: none, cw, ccw, 180 (or pass --degrees)."
    )


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def rotate_one(input_path: Path, output_path: Path, rotation_deg: int) -> None:
    image = Image.open(input_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    if rotation_deg % 360 != 0:
        image = image.rotate(rotation_deg, expand=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate images to create test cases (pixel-rotation; EXIF is normalized first)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (when --input is a file) or output directory (when --input is a directory). "
        "Default: alongside input with a suffix.",
    )
    parser.add_argument(
        "--direction",
        default="cw",
        help="Rotation direction preset: none, cw, ccw, 180. (Ignored if --degrees is set.)",
    )
    parser.add_argument(
        "--degrees",
        type=int,
        default=None,
        help="Rotate by explicit degrees. Positive is counter-clockwise (PIL convention).",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Filename suffix when generating output paths (default: '_rot{deg}').",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    rotation_deg = _rotation_degrees(args.direction, args.degrees)
    suffix = args.suffix or f"_rot{rotation_deg}"

    if input_path.is_file():
        if not _is_image_file(input_path):
            raise SystemExit(f"Not an image file: {input_path}")

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")

        rotate_one(input_path, output_path, rotation_deg)
        print(f"Saved: {output_path}")
        return

    # directory
    out_dir = Path(args.output) if args.output else input_path / "rotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_path.iterdir() if p.is_file() and _is_image_file(p)]
    if not files:
        raise SystemExit(f"No images found in: {input_path}")

    for p in files:
        out_path = out_dir / f"{p.stem}{suffix}{p.suffix}"
        rotate_one(p, out_path, rotation_deg)

    print(f"Saved {len(files)} images to: {out_dir}")


if __name__ == "__main__":
    main()

