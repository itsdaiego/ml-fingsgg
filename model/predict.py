"""Run inference on a single image."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from network import HotdogCNN


CHECKPOINT = Path(__file__).parent / "checkpoints" / "best.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict(image_path: str) -> tuple[str, float]:
    """Return (label, confidence) for a single image."""
    model = HotdogCNN()
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        # model returns a raw logit (see network.py), so sigmoid is needed here
        # to turn it into an actual [0, 1] probability
        prob = torch.sigmoid(model(tensor)).item()

    label = "hotdog" if prob >= 0.5 else "not hotdog"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    args = parser.parse_args()

    label, confidence = predict(args.image)
    print(f"{label} ({confidence:.1%})")


if __name__ == "__main__":
    main()
