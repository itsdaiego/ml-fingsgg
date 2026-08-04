"""Download hotdog images from multiple sources."""

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from tqdm import tqdm


OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "hotdog"


def from_search_engines(n: int) -> None:
    """Download hotdog images via Bing and Google image search."""
    per_engine = n // 2
    queries = ["hotdog food", "hot dog sausage bun", "hotdog sandwich"]

    for engine_cls, name in [(BingImageCrawler, "bing"), (GoogleImageCrawler, "google")]:
        per_query = per_engine // len(queries)
        for query in queries:
            crawler = engine_cls(storage={"root_dir": str(OUTPUT_DIR)})
            print(f"[{name}] '{query}' — {per_query} images")
            crawler.crawl(keyword=query, max_num=per_query, min_size=(100, 100))


def from_food101(n: int) -> None:
    """Download hotdog images from HuggingFace food-101 dataset."""
    print("Loading food-101 dataset...")
    ds = load_dataset("ethz/food101", split="train+validation")

    label_idx = ds.features["label"].str2int("hot_dog")
    hotdog_ds = ds.filter(lambda x: x["label"] == label_idx)
    hotdog_ds = hotdog_ds.select(range(min(n, len(hotdog_ds))))

    print(f"Saving {len(hotdog_ds)} food-101 hotdog images...")
    for i, sample in enumerate(tqdm(hotdog_ds)):
        img = sample["image"]
        dst = OUTPUT_DIR / f"food101_{i:04d}.jpg"
        if not dst.exists():
            img.save(dst, "JPEG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=600, help="Target total images")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["food101", "search"],
        choices=["food101", "search"],
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_source = args.total // len(args.sources)

    if "food101" in args.sources:
        from_food101(per_source)

    if "search" in args.sources:
        from_search_engines(per_source)

    total = len(list(OUTPUT_DIR.glob("*.jpg")))
    print(f"\nTotal images in {OUTPUT_DIR}: {total}")


if __name__ == "__main__":
    main()
