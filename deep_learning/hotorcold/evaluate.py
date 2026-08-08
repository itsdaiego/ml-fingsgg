import numpy as np
import pandas as pd
import torch

from train import CHECKPOINT_PATH, DATA_PATH, TEST_YEAR, create_model, encode_cities, encode_dates


CORRECT_TOLERANCE_C = 2.0

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)

df = pd.read_csv(DATA_PATH)

years = pd.to_datetime(df["date"]).dt.year
test_df = df.loc[years >= TEST_YEAR].copy()

city_names = checkpoint["city_names"]
city_features = encode_cities(test_df["city"], city_names)

date_features = encode_dates(test_df["date"])
features = pd.concat([city_features, date_features], axis="columns")

x_test = torch.tensor(features.to_numpy(dtype=np.float32))
y_test = torch.tensor(test_df["temperature_c"].to_numpy(dtype=np.float32)).unsqueeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(x_test.shape[1]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    predictions = model(x_test.to(device))
    errors = predictions - y_test.to(device)
    mse = torch.mean(errors.square()).item()
    mae = torch.mean(errors.abs()).item()
    correct = errors.abs() <= CORRECT_TOLERANCE_C

for city, prediction, actual, error, is_correct in zip(
    test_df["city"],
    predictions.cpu().squeeze(1),
    y_test.squeeze(1),
    errors.cpu().squeeze(1),
    correct.cpu().squeeze(1),
):
    status = "Correct" if is_correct.item() else "Incorrect"
    print(f"{city}: predicted {prediction.item():.1f} C | actual {actual.item():.1f} C | error {error.item():+.1f} C | {status}")

print(f"Result MSE: {mse:.4f}")
print(f"Result MAE: {mae:.2f} C")
print(f"Within {CORRECT_TOLERANCE_C:.0f} C: {correct.float().mean().item():.1%}")
