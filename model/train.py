"""Train the HotdogCNN from scratch."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import HotdogDataset, TRAIN_TRANSFORMS, VAL_TRANSFORMS
from network import HotdogCNN


CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary classification accuracy."""
    preds = (outputs >= 0.5).float()
    return (preds == labels).float().mean().item()


def train_epoch(
    model: HotdogCNN,
    loader: DataLoader,
    criterion: nn.BCELoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch, return (avg_loss, accuracy)."""
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


def val_epoch(
    model: HotdogCNN,
    loader: DataLoader,
    criterion: nn.BCELoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run one validation epoch, return (avg_loss, accuracy)."""
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            total_acc += accuracy(outputs, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = HotdogDataset()
    print(f"Total samples: {len(full_dataset)}")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_ds.dataset.transform = TRAIN_TRANSFORMS
    val_ds.dataset.transform = VAL_TRANSFORMS

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = HotdogCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
            print(f"  → saved checkpoint (val acc {val_acc:.3f})")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to {CHECKPOINT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
