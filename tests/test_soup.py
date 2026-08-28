import inspect

import pytest
import torch
import torch.nn as nn


def test_public_callable_api():
    from mlbricks import soup
    from mlbricks.soup import SOUP

    assert callable(soup)
    sig = inspect.signature(soup)
    assert list(sig.parameters) == [
        "dim", "width", "depth", "mixer", "ffn",
        "mixer_config", "ffn_config", "backend", "precision",
    ]
    model = soup(
        dim=16,
        width=24,
        depth=1,
        mixer="esa",
        ffn="saffn",
        mixer_config={"head": 4},
        backend="pytorch",
        precision="fp32",
    )
    assert isinstance(model, SOUP)


def test_memory_and_fusion_are_fixed_not_api_options():
    from mlbricks import soup
    sig = inspect.signature(soup)
    assert "memory" not in sig.parameters
    assert "fusion" not in sig.parameters
    model = soup(
        dim=16, width=24, mixer="esa", mixer_config={"head": 4},
        backend="pytorch", precision="fp32",
    )
    assert model.observer.mem_dim == 128
    assert model.fusion.in_proj.out_features == 768


def test_default_saffn_forward_and_width_bridges():
    from mlbricks import soup

    model = soup(
        dim=16,
        width=[24, 12],
        depth=2,
        mixer="esa",
        ffn="saffn",
        mixer_config={"head": 4},
        backend="pytorch",
        precision="fp32",
    )
    x = torch.randn(2, 5, 16)
    y = model(x)
    assert y.shape == x.shape
    assert isinstance(model.state_bridges[0], nn.Linear)
    assert model.observer.state_write.in_features == 12


def test_mixed_esa_bolt_and_bolt_uses_own_latent_default():
    from mlbricks import soup

    model = soup(
        dim=16,
        width=24,
        depth=2,
        mixer=["esa", "bolt"],
        ffn="saffn",
        mixer_config=[{"head": 4}, {"num_heads": 4}],
        backend="pytorch",
        precision="fp32",
    )
    assert model.layers[0].mixer_name == "esa"
    assert model.layers[1].mixer_name == "bolt"
    assert model.layers[1].mixer.latent_dim == 32
    x = torch.randn(2, 4, 16)
    assert model(x).shape == x.shape


def test_bolt_explicit_latent_override():
    from mlbricks import soup

    model = soup(
        dim=16,
        width=24,
        mixer="bolt",
        ffn="saffn",
        mixer_config={"num_heads": 4, "latent_dim": 8},
        backend="pytorch",
        precision="fp32",
    )
    assert model.layers[0].mixer.latent_dim == 8


def test_plain_ffn_is_allowed_and_keeps_shape():
    from mlbricks import soup

    model = soup(
        dim=16,
        width=24,
        mixer="esa",
        ffn="ffn",
        mixer_config={"head": 4},
        ffn_config={"hidden": 32, "activation": "gelu"},
        backend="pytorch",
        precision="fp32",
    )
    x = torch.randn(2, 3, 16)
    assert model(x).shape == x.shape


class CustomMixer(nn.Module):
    def forward(self, x):
        return 0.5 * x


class CustomStateAwareFFN(nn.Module):
    def __init__(self, dim=16, state_dim=24):
        super().__init__()
        self.out = nn.Linear(state_dim, dim, bias=False)

    def forward(self, x, current_context, previous_context, state):
        del x, previous_context
        next_state = state + current_context.mean(-1, keepdim=True)
        return self.out(next_state), next_state


def test_custom_components():
    from mlbricks import soup

    model = soup(
        dim=16,
        width=24,
        mixer=CustomMixer(),
        ffn=lambda dim=16, state_dim=24: CustomStateAwareFFN(dim, state_dim),
        backend="pytorch",
        precision="fp32",
    )
    x = torch.randn(2, 3, 16)
    assert model(x).shape == x.shape


def test_observer_is_causal_for_first_token():
    from mlbricks.soup.core import ObserverStateMemory

    observer = ObserverStateMemory(16, 24)
    h = torch.randn(2, 4, 16)
    state = torch.randn(2, 4, 24)
    memory, relevance = observer(h, state)
    assert memory.shape == h.shape
    assert relevance.shape == (2, 4, 1)
    # Current-token write is subtracted, so token zero has no preceding memory.
    assert torch.allclose(memory[:, 0], torch.zeros_like(memory[:, 0]), atol=1e-6)


def test_fusion_gates_sum_to_one():
    from mlbricks.soup.core import SOUPFusion

    fusion = SOUPFusion(16, 24)
    h = torch.randn(2, 4, 16)
    state = torch.randn(2, 4, 24)
    memory = torch.randn(2, 4, 16)
    y, gates = fusion(h, state, memory)
    assert y.shape == h.shape
    assert gates.shape == (2, 4, 3)
    assert torch.allclose(gates.sum(-1), torch.ones_like(gates[..., 0]), atol=1e-5)


def test_backend_propagation_and_config():
    from mlbricks import soup

    model = soup(
        dim=16,
        width=24,
        depth=2,
        mixer=["esa", "bolt"],
        ffn="saffn",
        mixer_config=[{"head": 4}, {"head": 4}],
        backend="pytorch",
        precision="fp32",
    )
    assert model.set_backend("auto") is model
    assert model.backend == "auto"
    assert model.layers[0].mixer.backend == "auto"
    assert model.layers[1].mixer.backend == "auto"
    cfg = model.to_config()
    assert cfg["mixer"] == ["esa", "bolt"]
    assert cfg["ffn"] == ["saffn", "saffn"]
    assert model.parameter_count > 0


def test_per_layer_length_validation():
    from mlbricks import soup
    from mlbricks.soup.core import ConfigurationError

    with pytest.raises(ConfigurationError):
        soup(dim=16, depth=3, mixer=["esa", "bolt"])
