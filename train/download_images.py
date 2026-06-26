"""Download hotdog or not-hotdog images from multiple sources."""

import argparse
from pathlib import Path

from datasets import load_dataset
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

NOT_HOTDOG_FOOD101_CLASSES = [
    "pizza", "steak", "sushi", "hamburger", "french_fries",
    "ice_cream", "tacos", "ramen", "fried_rice", "grilled_salmon",
    "chicken_wings", "caesar_salad", "waffles", "pancakes", "donuts",
]

HOTDOG_SEARCH_QUERIES = [
    "hotdog food",
    "hot dog sausage bun",
    "hotdog sandwich",
    "chicago hotdog",
    "hotdog mustard ketchup",
    "corn dog",
    "hotdog street food",
    "hotdog grilled",
    "ballpark hotdog",
    "hotdog with toppings",
]

NOT_HOTDOG_SEARCH_QUERIES = [
    "pizza slice",
    "hamburger burger food",
    "steak dinner plate",
    "sushi rolls",
    "french fries",
    "ice cream bowl",
    "tacos mexican food",
    "ramen noodles bowl",
    "fried chicken",
    "salad bowl food",
]


def from_search_engines(n: int, queries: list[str], output_dir: Path) -> None:
    """Download images via Bing and Google image search."""
    per_engine = n // 2
    per_query = max(1, per_engine // len(queries))

    for engine_cls, name in [(BingImageCrawler, "bing"), (GoogleImageCrawler, "google")]:
        for query in queries:
            crawler = engine_cls(storage={"root_dir": str(output_dir)})
            print(f"[{name}] '{query}' — {per_query} images")
            crawler.crawl(keyword=query, max_num=per_query, min_size=(100, 100))


def from_food101_hotdog(n: int, output_dir: Path) -> None:
    """Download hot_dog images from HuggingFace food-101 dataset."""
    print("Loading food-101 dataset (hotdog)...")
    ds = load_dataset("ethz/food101", split="train+validation")

    label_idx = ds.features["label"].str2int("hot_dog")
    filtered = ds.filter(lambda x: x["label"] == label_idx)
    filtered = filtered.select(range(min(n, len(filtered))))

    print(f"Saving {len(filtered)} hotdog images from food-101...")
    for i, sample in enumerate(tqdm(filtered)):
        dst = output_dir / f"food101_hotdog_{i:04d}.jpg"
        if not dst.exists():
            sample["image"].save(dst, "JPEG")


def from_food101_not_hotdog(n: int, output_dir: Path) -> None:
    """Download non-hotdog food images from HuggingFace food-101 dataset."""
    print("Loading food-101 dataset (not hotdog)...")
    ds = load_dataset("ethz/food101", split="train+validation")

    per_class = max(1, n // len(NOT_HOTDOG_FOOD101_CLASSES))
    saved = 0

    for cls_name in NOT_HOTDOG_FOOD101_CLASSES:
        if saved >= n:
            break
        label_idx = ds.features["label"].str2int(cls_name)
        filtered = ds.filter(lambda x, idx=label_idx: x["label"] == idx)
        filtered = filtered.select(range(min(per_class, len(filtered))))

        print(f"Saving {len(filtered)} '{cls_name}' images...")
        for i, sample in enumerate(tqdm(filtered, leave=False)):
            dst = output_dir / f"food101_{cls_name}_{i:04d}.jpg"
            if not dst.exists():
                sample["image"].save(dst, "JPEG")
                saved += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        choices=["hotdog", "not_hotdog"],
        default="hotdog",
    )
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["food101", "search"],
        choices=["food101", "search"],
    )
    args = parser.parse_args()

    output_dir = DATA_DIR / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    per_source = args.total // len(args.sources)

    if args.category == "hotdog":
        if "food101" in args.sources:
            from_food101_hotdog(per_source, output_dir)
        if "search" in args.sources:
            from_search_engines(per_source, HOTDOG_SEARCH_QUERIES, output_dir)
    else:
        if "food101" in args.sources:
            from_food101_not_hotdog(per_source, output_dir)
        if "search" in args.sources:
            from_search_engines(per_source, NOT_HOTDOG_SEARCH_QUERIES, output_dir)

    total = len(list(output_dir.glob("*.jpg")))
    print(f"\nTotal images in {output_dir}: {total}")


if __name__ == "__main__":
    main()
