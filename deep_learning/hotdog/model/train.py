"""Train the HotdogCNN from scratch."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import HotdogDataset, TRAIN_TRANSFORMS, VAL_TRANSFORMS
from network import HotdogCNN


CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
TEST_SPLIT = 0.15


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary classification accuracy."""
    # outputs are raw logits (BCEWithLogitsLoss expects that), so the decision
    # boundary is 0.0, not 0.5 — sigmoid(0) == 0.5 is where a probability would cross 50%
    preds = (outputs >= 0.0).float()
    return (preds == labels).float().mean().item()


def train_epoch(
    model: HotdogCNN,
    loader: DataLoader,
    loss_fn: nn.Module,
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
        loss = loss_fn(outputs, labels)

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
    loss_fn: nn.Module,
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
            total_loss += loss_fn(outputs, labels).item()
            total_acc += accuracy(outputs, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    # this is important since we need to make sure that the same images are in the test split every time we run the script, otherwise we won't be able to compare results across runs
    parser.add_argument("--seed", type=int, default=42, help="Controls which images land in the test split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_samples = len(HotdogDataset())
    print(f"Total samples: {num_samples}")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()

    test_size = int(num_samples * TEST_SPLIT)
    val_size = int(num_samples * args.val_split)

    train_indices = indices[: num_samples - val_size - test_size]
    val_indices = indices[num_samples - val_size - test_size : num_samples - test_size]
    test_indices = indices[num_samples - test_size :]

    train_ds = Subset(HotdogDataset(transform=TRAIN_TRANSFORMS), train_indices)
    val_ds = Subset(HotdogDataset(transform=VAL_TRANSFORMS), val_indices)
    test_ds = Subset(HotdogDataset(transform=VAL_TRANSFORMS), test_indices)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test (held out): {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = HotdogCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, loss_fn, device)

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

    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best.pt", map_location=device))
    test_loss, test_acc = val_epoch(model, test_loader, loss_fn, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
