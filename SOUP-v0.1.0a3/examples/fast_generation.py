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

# After moving to the final inference device:
model.eval().prepare_generation()
