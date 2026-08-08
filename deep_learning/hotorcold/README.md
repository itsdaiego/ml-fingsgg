# City Temperature Predictor

Predict daily `temperature_c` from city and date.

Data: 10,000 synthetic rows, 10 cities, 1,000 samples each, dates from 2016 through 2025. Labels use city-specific seasonal curves plus Gaussian noise (`NOISE_STD_C=0.25`).

Features: 10 city one-hot values, day-of-year sine, day-of-year cosine.

Model: `12 -> Linear(16) -> ReLU -> Linear(1)`.

## Run

```bash
uv sync
uv run python generate_dataset.py
uv run python train.py
uv run python evaluate.py
```

## Results

```text
MSE: 0.0658
MAE: 0.21C
Within 0.5C: 94.8%
```
