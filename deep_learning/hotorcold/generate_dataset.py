"""Generate a reproducible starter dataset for temperature regression.

The generated temperatures are synthetic. Replace them with measured historical
temperatures before using a model for real-world forecasts.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


SAMPLES_PER_CITY = 1_000
SEED = 42
START = date(2016, 1, 1)
END = date(2025, 12, 31)
OUTPUT_PATH = Path(__file__).parent / "data" / "temperature_samples.csv"

# Approximate climate parameters used only to make a learnable demo dataset.
CITIES = (
    ("Sao Paulo", "Brazil", 19.6, 4.0),
    ("Rio de Janeiro", "Brazil", 23.5, 5.5),
    ("New York", "United States", 12.5, 13.0),
    ("London", "United Kingdom", 11.5, 8.0),
    ("Tokyo", "Japan", 16.0, 12.0),
    ("Cairo", "Egypt", 22.5, 11.0),
    ("Sydney", "Australia", 18.0, 7.0),
    ("Cape Town", "South Africa", 17.5, 6.0),
    ("Mumbai", "India", 27.5, 3.5),
    ("Reykjavik", "Iceland", 5.0, 10.0),
)


def temperature_c(
    city_index: int,
    sample_date: date,
    average_c: float,
    seasonal_range_c: float,
    rng: random.Random,
) -> float:
    """Return synthetic daily temperature with seasonal and random variation."""
    day_angle = 2 * math.pi * (sample_date.timetuple().tm_yday - 173) / 365.25
    if city_index in (0, 1, 6, 7):  # Southern Hemisphere seasons are reversed.
        day_angle += math.pi
    return round(
        average_c
        + seasonal_range_c * math.cos(day_angle)
        + rng.gauss(0, 2.0),
        1,
    )


def main() -> None:
    rng = random.Random(SEED)
    total_days = (END - START).days + 1
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("city", "country", "date", "temperature_c"),
        )
        writer.writeheader()
        for city_index, (city, country, average_c, seasonal_range_c) in enumerate(CITIES):
            for _ in range(SAMPLES_PER_CITY):
                sample_date = START + timedelta(days=rng.randrange(total_days))
                writer.writerow(
                    {
                        "city": city,
                        "country": country,
                        "date": sample_date.isoformat(),
                        "temperature_c": temperature_c(
                            city_index,
                            sample_date,
                            average_c,
                            seasonal_range_c,
                            rng,
                        ),
                    }
                )

    print(f"Wrote {len(CITIES) * SAMPLES_PER_CITY} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
