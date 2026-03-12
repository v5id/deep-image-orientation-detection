from PIL import Image
import torch
from model import OrientationNet
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

model = OrientationNet().to(device)
model.load_state_dict(torch.load("./models/orientation_model.pth", map_location=device))
model.eval()

# test all rotations of a training image
for angle in [0, 90, 180, 270]:
    img = Image.open("./data/upright_images/revert.png").convert("RGB")
    img = img.rotate(angle, expand=True)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()
    print(f"Rotated {angle}° -> Prediction: {pred}")