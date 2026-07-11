"""PyTorch dataset for hotdog binary classification."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    # not really needed since we are not using a predtrained model, but it's a good practice to normalize images when training a CNN
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class HotdogDataset(Dataset):
    """Binary image dataset: hotdog (1) vs not_hotdog (0)."""

    def __init__(self, transform: transforms.Compose | None = None) -> None:
        self.transform = transform or VAL_TRANSFORMS
        self.samples: list[tuple[Path, float]] = []

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in (DATA_DIR / "hotdog").glob(ext):
                self.samples.append((path, 1.0))
            for path in (DATA_DIR / "not_hotdog").glob(ext):
                self.samples.append((path, 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
