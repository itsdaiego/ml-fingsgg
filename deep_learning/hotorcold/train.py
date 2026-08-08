from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss


DATA_PATH = Path(__file__).parent / "data" / "temperature_samples.csv"
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "temperature_model.pt"
TEST_YEAR = 2025
EPOCH_ROUNDS = 1_000


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


def encode_cities(cities: pd.Series, city_names: list[str]) -> pd.DataFrame:
    "Encode cities such that each represent an unique combination of zeroes and a single 1 at a given index"
    city_to_index = {city: index for index, city in enumerate(city_names)}
    city_ids = cities.map(city_to_index)

    encoded = np.zeros((len(cities), len(city_names)), dtype=np.float32)
    encoded[np.arange(len(cities)), city_ids.to_numpy(dtype=int)] = 1

    return pd.DataFrame(encoded, columns=city_names, index=cities.index)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    years = pd.to_datetime(df["date"]).dt.year
    train_df = df.loc[years < TEST_YEAR].copy()
    test_df = df.loc[years >= TEST_YEAR].copy()
    city_names = sorted(train_df["city"].unique())
    city_features = encode_cities(train_df["city"], city_names)
    date_features = encode_dates(train_df["date"])
    features = pd.concat([city_features, date_features], axis="columns")
    x_train = torch.tensor(features.to_numpy(dtype=np.float32))
    y_train = torch.tensor(train_df["temperature_c"].to_numpy(dtype=np.float32)).unsqueeze(1)

    torch.manual_seed(42)
    model = torch.nn.Linear(in_features=x_train.shape[1], out_features=1)
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    total_loss = 0.0

    for epoch in range(EPOCH_ROUNDS):
        predictions = model(x_train)
        loss = loss_fn(predictions, y_train)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "city_names": city_names,
        },
        CHECKPOINT_PATH,
    )
    print(f"Mean training loss: {total_loss / EPOCH_ROUNDS:.4f}")
