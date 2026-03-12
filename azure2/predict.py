import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import OrientationNet


def load_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")

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

    print("\nPrediction:", prediction)
    print("Orientation:", orientation_labels[prediction])
    print("Confidence:", probabilities[0][prediction].item())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    main(args.input_path, args.model_path)