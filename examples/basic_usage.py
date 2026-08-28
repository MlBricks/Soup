import torch
from mlbricks import soup

model = soup(
    dim=512,
    width=1116,
    depth=2,
    mixer="esa",
    ffn="saffn",
    backend="auto",
    precision="fp16",
)

x = torch.randn(2, 128, 512)
y = model(x)
print(y.shape)
