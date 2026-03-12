import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from model import OrientationNet


class Letterbox:

    def __init__(self, size, fill=(255, 255, 255)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image):
        w, h = image.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (self.size, self.size), self.fill)

        scale = min(self.size / w, self.size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = image.resize((nw, nh), resample=Image.BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - nw) // 2
        top = (self.size - nh) // 2
        canvas.paste(resized, (left, top))
        return canvas


def load_image(image_path):

    transform = transforms.Compose([
        Letterbox(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    return image


def main(input_path, model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OrientationNet().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    image = load_image(input_path).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()

    orientation_labels = {
        0: "0° (correctly oriented)",
        1: "90° Clockwise",
        2: "180°",
        3: "90° Counter-Clockwise"
    }

    # how to rotate the input to make it upright
    rotate_to_upright = {
        0: 0,
        1: 90,   # rotate CCW
        2: 180,
        3: -90,  # rotate CW
    }

    print("\nPrediction:", prediction)
    print("Orientation:", orientation_labels[prediction])
    print("Confidence:", probabilities[0][prediction].item())
    print("Rotate to upright (degrees):", rotate_to_upright[prediction])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    main(args.input_path, args.model_path)
