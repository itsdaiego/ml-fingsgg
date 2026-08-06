from pathlib import Path

import numpy as np
import pandas as pd
import torch


def encode_dates(dates: pd.Series) -> pd.DataFrame:
    "Encode each date position in relation to the annual values"
    timestamps = pd.to_datetime(dates)

    day_of_year = timestamps.dt.dayofyear - 1

    return pd.DataFrame(
        {
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        },
        index=dates.index,
    )


def encode_cities(cities: pd.Series) -> pd.DataFrame:
    "Encode cities such that each represent an unique combination of zeroes and a single 1 at a given index"
    city_names = sorted(cities.unique())
    city_to_index = {city: index for index, city in enumerate(city_names)}

    city_ids = cities.map(city_to_index).to_numpy(dtype=int)
    encoded = np.zeros((len(cities), len(city_names)), dtype=np.float32)

    encoded[np.arange(len(cities)), city_ids] = 1

    return pd.DataFrame(encoded, columns=city_names, index=cities.index)


def prepare_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    city_features = encode_cities(df["city"])
    date_features = encode_dates(df["date"])

    features = pd.concat([city_features, date_features], axis="columns")
    x = torch.tensor(features.to_numpy(dtype=np.float32))
    y = torch.tensor(df["temperature_c"].to_numpy(dtype=np.float32)).unsqueeze(1)

    return (x, y)


if __name__ == "__main__":
    df = pd.read_csv("./data/temperature_samples.csv")
    x, y = prepare_tensors(df)
    print(f"X shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"First X row: {x[0]}")
    print(f"First y value: {y[0]}")
