"""PyTorch dataset for hotdog binary classification."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class HotdogDataset(Dataset):
    """Binary image dataset: hotdog (1) vs not_hotdog (0)."""

    def __init__(self, transform: transforms.Compose | None = None) -> None:
        self.transform = transform or VAL_TRANSFORMS
        self.samples: list[tuple[Path, float]] = []

        for path in (DATA_DIR / "hotdog").glob("*.jpg"):
            self.samples.append((path, 1.0))
        for path in (DATA_DIR / "not_hotdog").glob("*.jpg"):
            self.samples.append((path, 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
