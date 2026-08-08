# Hotdog Classifier

<img width="846" height="730" alt="image" src="https://github.com/user-attachments/assets/e3dea0eb-bc1a-4cee-85a8-b92c7b06dfee" />

Binary CNN: `hotdog=1`, `not_hotdog=0`.

Model: `3x224x224 -> Conv(32,64,128,256) -> Linear(1)`.

## Run

```bash
cd deep_learning/hotdog/model
uv sync
uv run python train.py --epochs 20 --batch-size 32 --lr 0.001
```

```bash
cd ..
./run.sh path/to/image.jpg
```
