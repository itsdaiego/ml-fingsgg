"""Evaluate the trained model on the held-out test split."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import HotdogDataset, VAL_TRANSFORMS
from network import HotdogCNN

CHECKPOINT = Path(__file__).parent / "checkpoints" / "best.pt"
TEST_SPLIT = 0.15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HotdogDataset(transform=VAL_TRANSFORMS)
    num_samples = len(dataset)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()

    test_size = int(num_samples * TEST_SPLIT)
    test_indices = indices[num_samples - test_size:]
    test_ds = Subset(dataset, test_indices)

    print(f"Device: {device}")
    print(f"Test samples: {len(test_ds)}")

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = HotdogCNN()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            preds = (outputs >= 0.0).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAccuracy:  {acc:.3f} ({correct}/{total})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"\nConfusion matrix:")
    print(f"  TP={int(tp)} FP={int(fp)}")
    print(f"  FN={int(fn)} TN={int(tn)}")


if __name__ == "__main__":
    main()
