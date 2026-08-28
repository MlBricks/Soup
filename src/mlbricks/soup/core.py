from __future__ import annotations

import copy
import inspect
import math
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

_BACKENDS = {"auto", "native", "pytorch"}
_OBSERVER_DIM = 128
_FUSION_HIDDEN = 768
_DEPTH_DIM = 64


class ConfigurationError(ValueError):
    """Raised when a SOUP architecture configuration is inconsistent."""


def _normalize_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in _BACKENDS:
        raise ConfigurationError(
            f"backend must be one of {sorted(_BACKENDS)}, got {value!r}"
        )
    return backend


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _normalize_width(width: int | Sequence[int], depth: int) -> tuple[int, ...]:
    if isinstance(width, int):
        if width <= 0:
            raise ConfigurationError("width must be > 0")
        return (int(width),) * depth
    values = tuple(int(v) for v in width)
    if len(values) != depth:
        raise ConfigurationError(
            f"width list must have one value per depth: depth={depth}, got {len(values)}"
        )
    if any(v <= 0 for v in values):
        raise ConfigurationError("all widths must be > 0")
    return values


def _normalize_components(value: Any, depth: int, *, name: str) -> tuple[Any, ...]:
    if _is_sequence(value):
        values = tuple(value)
        if len(values) != depth:
            raise ConfigurationError(
                f"{name} list must have one value per depth: depth={depth}, got {len(values)}"
            )
        return values
    return (value,) * depth


def _normalize_configs(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any] | None] | None,
    depth: int,
    *,
    name: str,
) -> tuple[dict[str, Any], ...]:
    if value is None:
        return tuple({} for _ in range(depth))
    if isinstance(value, Mapping):
        return tuple(copy.deepcopy(dict(value)) for _ in range(depth))
    values = tuple(value)
    if len(values) != depth:
        raise ConfigurationError(
            f"{name} list must have one value per depth: depth={depth}, got {len(values)}"
        )
    out: list[dict[str, Any]] = []
    for item in values:
        if item is None:
            out.append({})
        elif isinstance(item, Mapping):
            out.append(copy.deepcopy(dict(item)))
        else:
            raise ConfigurationError(f"each {name} entry must be a mapping or None")
    return tuple(out)


def _filter_kwargs(factory: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _custom_component(spec: Any, kwargs: dict[str, Any], *, kind: str) -> nn.Module:
    if isinstance(spec, nn.Module):
        if kwargs:
            raise ConfigurationError(
                f"{kind}_config cannot be used with a pre-built nn.Module instance"
            )
        try:
            return copy.deepcopy(spec)
        except Exception:
            return spec
    if callable(spec):
        try:
            module = spec(**_filter_kwargs(spec, kwargs))
        except TypeError as exc:
            raise ConfigurationError(f"could not construct custom {kind}: {exc}") from exc
        if not isinstance(module, nn.Module):
            raise ConfigurationError(f"custom {kind} factory must return torch.nn.Module")
        return module
    raise ConfigurationError(
        f"{kind} must be a supported name, nn.Module, or module factory; got {type(spec).__name__}"
    )


class RMSNorm(nn.Module):
    """RMSNorm used by the validated notebook SOUP block."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.float()
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
        return (z * self.weight.float()).to(x.dtype)


class _SOUPStateAwareFFN(nn.Module):
    """Exact state-aware FFN equations used by the supplied SOUP notebook."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        *,
        depth_dim: int = _DEPTH_DIM,
        layer_index: int,
        total_layers: int,
    ):
        super().__init__()
        self.dim = int(dim)
        self.state_dim = int(state_dim)

        s = self.state_dim
        self.x_proj = nn.Linear(dim, 3 * s, bias=True)
        self.context_candidate = nn.Linear(dim, s, bias=False)
        self.context_write = nn.Linear(dim, s, bias=False)
        self.state_proj = nn.Linear(s, 2 * s, bias=False)
        # Notebook SOUP uses output_bias=False.
        self.output = nn.Linear(s, dim, bias=False)

        self.depth_embedding = nn.Parameter(torch.empty(depth_dim))
        self.depth_proj = nn.Linear(depth_dim, 3 * s, bias=False)

        depth = layer_index / max(total_layers - 1, 1)
        self.retain_logit = nn.Parameter(torch.full((s,), 1.4 - 0.5 * depth))
        self.read_logit = nn.Parameter(torch.full((s,), -0.2 + 0.4 * depth))
        self.candidate_transition_logit = nn.Parameter(torch.tensor(-2.0))
        self.write_transition_logit = nn.Parameter(torch.tensor(-2.0))
        self.retain_delta_scale = nn.Parameter(torch.full((s,), 0.10))
        self.read_delta_scale = nn.Parameter(torch.full((s,), 0.10))
        self.delta_magnitude_log_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.normal_(self.depth_embedding, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        current_context: torch.Tensor,
        previous_context: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.state_dim
        xc, xw, xv = self.x_proj(x).split(s, dim=-1)
        sc, sw = self.state_proj(state).split(s, dim=-1)
        dc, dw, dv = self.depth_proj(self.depth_embedding).split(s, dim=-1)

        delta = current_context - previous_context
        dm = torch.sqrt(
            delta.float().square().mean(-1, keepdim=True) + 1e-6
        ).to(current_context.dtype)
        scaled = torch.exp(self.delta_magnitude_log_scale) * dm

        candidate_context = current_context + torch.sigmoid(
            self.candidate_transition_logit
        ) * delta
        write_context = current_context + torch.sigmoid(
            self.write_transition_logit
        ) * delta

        candidate = torch.tanh(
            xc + self.context_candidate(candidate_context) + sc + dc
        )
        write = torch.sigmoid(
            xw + self.context_write(write_context) + sw + dw
        )
        retain = torch.sigmoid(
            self.retain_logit - scaled * self.retain_delta_scale
        )
        next_state = (1.0 - write) * (retain * state) + write * candidate

        value = F.silu(xv + dv)
        read = torch.sigmoid(self.read_logit + scaled * self.read_delta_scale)
        out = self.output(next_state * value * read)
        return out, next_state


class _PlainFFN(nn.Module):
    """Conventional FFN adapter for SOUP's replaceable FFN slot.

    A plain FFN has no recurrent-state equation, so it leaves the incoming SOUP
    state unchanged. This makes it composable with state-aware layers without
    inventing a new memory rule that is absent from the validated SOUP notebook.
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden: int | None = None,
        activation: str = "silu",
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = int(hidden or 4 * dim)
        if hidden <= 0:
            raise ConfigurationError("FFN hidden must be > 0")
        name = str(activation).strip().lower()
        activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "silu": F.silu,
            "gelu": F.gelu,
            "relu": F.relu,
            "mish": F.mish,
        }
        if name not in activations:
            raise ConfigurationError(
                f"plain ffn activation must be one of {sorted(activations)}, got {activation!r}"
            )
        self.activation_name = name
        self.activation = activations[name]
        self.in_proj = nn.Linear(dim, hidden, bias=bias)
        self.out_proj = nn.Linear(hidden, dim, bias=bias)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        current_context: torch.Tensor,
        previous_context: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del current_context, previous_context
        out = self.out_proj(self.dropout(self.activation(self.in_proj(x))))
        return out, state


class _CustomFFNAdapter(nn.Module):
    """Accept either SOUP-state-aware FFNs or conventional ``forward(x)`` FFNs."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._state_aware = self._detect_state_aware(module)

    @staticmethod
    def _detect_state_aware(module: nn.Module) -> bool:
        try:
            sig = inspect.signature(module.forward)
        except (TypeError, ValueError):
            return True
        params = list(sig.parameters.values())
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
            return True
        positional = [
            p for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) >= 4

    def forward(
        self,
        x: torch.Tensor,
        current_context: torch.Tensor,
        previous_context: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._state_aware:
            result = self.module(x, current_context, previous_context, state)
            if not (isinstance(result, tuple) and len(result) == 2):
                raise RuntimeError(
                    "state-aware custom FFN must return (output, next_state)"
                )
            return result
        return self.module(x), state


# torch.compiler.disable is used in the validated notebook. Keep the same
# execution boundary while remaining importable on older compatible torch.
try:
    _compiler_disable = torch.compiler.disable
except Exception:  # pragma: no cover
    def _compiler_disable(fn):
        return fn


class ObserverStateMemory(nn.Module):
    """Fixed SOUP observer memory from the validated notebook architecture."""

    def __init__(self, dim: int, state_dim: int):
        super().__init__()
        memory_dim = _OBSERVER_DIM
        self.mem_dim = memory_dim
        self.state_write = nn.Linear(state_dim, memory_dim, bias=False)
        # The bias is intentional and is part of the validated notebook model.
        self.write_gate = nn.Linear(state_dim, memory_dim, bias=True)
        self.query = nn.Linear(dim + state_dim, memory_dim, bias=False)
        self.key = nn.Linear(memory_dim, memory_dim, bias=False)
        self.value = nn.Linear(memory_dim, dim, bias=False)
        self.relevance_bias = nn.Parameter(torch.tensor(-0.5))

    @_compiler_disable
    def forward(
        self, h: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cand = torch.tanh(self.state_write(state))
        w = torch.sigmoid(self.write_gate(state))
        written = w * cand

        csum = torch.cumsum(written.float(), dim=1)
        ccount = torch.cumsum(w.float(), dim=1)

        # Read only preceding memory: current-token write is subtracted.
        past = ((csum - written.float()) / (ccount - w.float() + 1e-6)).to(h.dtype)

        q = F.normalize(self.query(torch.cat([h, state], dim=-1)).float(), dim=-1)
        k = F.normalize(self.key(past).float(), dim=-1)
        relevance = torch.sigmoid(
            (q * k).sum(-1, keepdim=True) / math.sqrt(self.mem_dim)
            + self.relevance_bias
        ).to(h.dtype)
        memory = relevance * self.value(past)
        return memory, relevance


class SOUPFusion(nn.Module):
    """Fixed three-way representation/state/memory fusion from the notebook."""

    def __init__(self, dim: int, state_dim: int):
        super().__init__()
        self.state_to_d = nn.Linear(state_dim, dim, bias=False)
        self.gate = nn.Linear(3 * dim, 3, bias=True)
        self.in_proj = nn.Linear(3 * dim, _FUSION_HIDDEN, bias=False)
        self.out_proj = nn.Linear(_FUSION_HIDDEN, dim, bias=False)

    def forward(
        self, h: torch.Tensor, state: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.state_to_d(state)
        cat = torch.cat([h, s, memory], dim=-1)
        gates = torch.softmax(self.gate(cat).float(), dim=-1).to(h.dtype)
        weighted = torch.cat(
            [
                gates[..., 0:1] * h,
                gates[..., 1:2] * s,
                gates[..., 2:3] * memory,
            ],
            dim=-1,
        )
        fused = self.out_proj(F.silu(self.in_proj(weighted)))
        return h + fused, gates


def _build_mixer(
    spec: Any,
    *,
    dim: int,
    config: dict[str, Any],
    backend: str,
    precision: str,
) -> tuple[nn.Module, str]:
    if not isinstance(spec, str):
        if isinstance(spec, nn.Module):
            module = _custom_component(spec, dict(config), kind="mixer")
        else:
            cfg = dict(config)
            cfg.setdefault("dim", dim)
            cfg.setdefault("d_model", dim)
            cfg.setdefault("backend", backend)
            cfg.setdefault("precision", precision)
            module = _custom_component(spec, cfg, kind="mixer")
        name = spec.__class__.__name__ if isinstance(spec, nn.Module) else getattr(spec, "__name__", "custom")
        return module, name

    name = spec.strip().lower()
    cfg = dict(config)

    if name == "esa":
        from ..esa import ESA

        reserved = {"embd", "device", "auto_move_input", "auto_compile"}
        conflict = reserved.intersection(cfg)
        if conflict:
            raise ConfigurationError(
                f"ESA mixer_config cannot override SOUP-managed fields: {sorted(conflict)}"
            )
        kwargs = {
            "embd": dim,
            # Notebook SOUP used 8 heads. Updated ESA supplies the remaining defaults.
            "head": int(cfg.pop("head", 8)),
            "backend": cfg.pop("backend", backend),
            "precision": cfg.pop("precision", precision),
            "device": None,
            "auto_move_input": False,
            "auto_compile": False,
            **cfg,
        }
        return ESA(**kwargs), "esa"

    if name == "bolt":
        from ..bolt import Bolt

        if "d_model" in cfg:
            raise ConfigurationError("BOLT mixer_config cannot override SOUP-managed d_model")
        if "head" in cfg and "num_heads" in cfg:
            raise ConfigurationError("BOLT mixer_config must use only one of head or num_heads")
        num_heads = int(cfg.pop("head", cfg.pop("num_heads", 8)))
        kwargs: dict[str, Any] = {
            "d_model": dim,
            "num_heads": num_heads,
            "backend": cfg.pop("backend", backend),
        }
        # Do NOT synthesize latent_dim: omitted means BOLT uses its own default (32).
        if "latent_dim" in cfg:
            kwargs["latent_dim"] = cfg.pop("latent_dim")
        kwargs.update(cfg)
        return Bolt(**kwargs), "bolt"

    raise ConfigurationError(
        f"unsupported mixer {spec!r}; use 'esa', 'bolt', an nn.Module, or a module factory"
    )


def _build_ffn(
    spec: Any,
    *,
    dim: int,
    state_dim: int,
    layer_index: int,
    total_layers: int,
    config: dict[str, Any],
) -> tuple[nn.Module, str]:
    if not isinstance(spec, str):
        if isinstance(spec, nn.Module):
            module = _custom_component(spec, dict(config), kind="ffn")
        else:
            cfg = dict(config)
            cfg.setdefault("dim", dim)
            cfg.setdefault("d_model", dim)
            cfg.setdefault("width", state_dim)
            cfg.setdefault("state_dim", state_dim)
            module = _custom_component(spec, cfg, kind="ffn")
        name = spec.__class__.__name__ if isinstance(spec, nn.Module) else getattr(spec, "__name__", "custom")
        return _CustomFFNAdapter(module), name

    name = spec.strip().lower()
    cfg = dict(config)
    if name == "saffn":
        depth_dim = int(cfg.pop("depth_dim", _DEPTH_DIM))
        if cfg:
            raise ConfigurationError(
                f"unsupported saffn ffn_config keys: {sorted(cfg)}"
            )
        return _SOUPStateAwareFFN(
            dim,
            state_dim,
            depth_dim=depth_dim,
            layer_index=layer_index,
            total_layers=total_layers,
        ), "saffn"

    if name == "ffn":
        return _PlainFFN(dim, **cfg), "ffn"

    raise ConfigurationError(
        f"unsupported ffn {spec!r}; use 'saffn', 'ffn', an nn.Module, or a module factory"
    )


class SOUPLayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        state_dim: int,
        layer_index: int,
        total_layers: int,
        mixer: Any,
        ffn: Any,
        mixer_config: dict[str, Any],
        ffn_config: dict[str, Any],
        backend: str,
        precision: str,
    ):
        super().__init__()
        self.dim = int(dim)
        self.state_dim = int(state_dim)
        self.norm = RMSNorm(dim)
        self.mixer, self.mixer_name = _build_mixer(
            mixer,
            dim=dim,
            config=mixer_config,
            backend=backend,
            precision=precision,
        )
        self.ffn, self.ffn_name = _build_ffn(
            ffn,
            dim=dim,
            state_dim=state_dim,
            layer_index=layer_index,
            total_layers=total_layers,
            config=ffn_config,
        )
        self.mix = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        previous_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.norm(x)
        current_context = self.mixer(z)
        if not isinstance(current_context, torch.Tensor) or current_context.shape != x.shape:
            raise RuntimeError(
                "SOUP mixer must return a tensor with the same [B,T,D] shape as its input"
            )
        ffn_out, next_state = self.ffn(z, current_context, previous_context, state)
        if not isinstance(ffn_out, torch.Tensor) or ffn_out.shape != x.shape:
            raise RuntimeError("SOUP FFN output must have the same [B,T,D] shape as x")
        expected_state = (*x.shape[:-1], self.state_dim)
        if not isinstance(next_state, torch.Tensor) or tuple(next_state.shape) != expected_state:
            raise RuntimeError(
                f"SOUP FFN next_state must have shape {expected_state}, got "
                f"{None if not isinstance(next_state, torch.Tensor) else tuple(next_state.shape)}"
            )
        x = x + torch.sigmoid(self.mix) * (current_context + ffn_out)
        return x, next_state, current_context


class SOUP(nn.Module):
    """Standalone adaptable SOUP architecture.

    The layer-local mixer and FFN are replaceable. Observer State Memory and
    SOUP Fusion are fixed defining blocks and are therefore not constructor
    choices.
    """

    def __init__(
        self,
        dim: int = 512,
        width: int | Sequence[int] = 1116,
        depth: int = 2,
        mixer: Any = "esa",
        ffn: Any = "saffn",
        mixer_config: Mapping[str, Any] | Sequence[Mapping[str, Any] | None] | None = None,
        ffn_config: Mapping[str, Any] | Sequence[Mapping[str, Any] | None] | None = None,
        backend: str = "auto",
        precision: str = "fp16",
    ):
        super().__init__()
        if dim <= 0:
            raise ConfigurationError("dim must be > 0")
        if depth <= 0:
            raise ConfigurationError("depth must be > 0")

        self.dim = int(dim)
        self.depth = int(depth)
        self.widths = _normalize_width(width, self.depth)
        self.backend = _normalize_backend(backend)
        self.precision = str(precision).strip().lower()

        mixer_specs = _normalize_components(mixer, self.depth, name="mixer")
        ffn_specs = _normalize_components(ffn, self.depth, name="ffn")
        mixer_configs = _normalize_configs(mixer_config, self.depth, name="mixer_config")
        ffn_configs = _normalize_configs(ffn_config, self.depth, name="ffn_config")

        self._mixer_specs = tuple(
            s.strip().lower() if isinstance(s, str) else getattr(s, "__name__", s.__class__.__name__)
            for s in mixer_specs
        )
        self._ffn_specs = tuple(
            s.strip().lower() if isinstance(s, str) else getattr(s, "__name__", s.__class__.__name__)
            for s in ffn_specs
        )
        self._mixer_configs = tuple(copy.deepcopy(c) for c in mixer_configs)
        self._ffn_configs = tuple(copy.deepcopy(c) for c in ffn_configs)

        self.layers = nn.ModuleList(
            [
                SOUPLayer(
                    dim=self.dim,
                    state_dim=self.widths[i],
                    layer_index=i,
                    total_layers=self.depth,
                    mixer=mixer_specs[i],
                    ffn=ffn_specs[i],
                    mixer_config=mixer_configs[i],
                    ffn_config=ffn_configs[i],
                    backend=self.backend,
                    precision=self.precision,
                )
                for i in range(self.depth)
            ]
        )

        bridges: list[nn.Module] = []
        for i in range(self.depth - 1):
            src, dst = self.widths[i], self.widths[i + 1]
            bridges.append(nn.Identity() if src == dst else nn.Linear(src, dst, bias=False))
        self.state_bridges = nn.ModuleList(bridges)

        final_width = self.widths[-1]
        # Fixed defining SOUP blocks from the notebook architecture.
        self.observer = ObserverStateMemory(self.dim, final_width)
        self.fusion = SOUPFusion(self.dim, final_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SOUP input must have shape [B,T,D], got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"SOUP expected dim={self.dim}, got {x.shape[-1]}")

        state = x.new_zeros(*x.shape[:-1], self.widths[0])
        previous_context = torch.zeros_like(x)

        for i, layer in enumerate(self.layers):
            x, state, previous_context = layer(x, state, previous_context)
            if i < len(self.state_bridges):
                state = self.state_bridges[i](state)

        memory, _ = self.observer(x, state)
        x, _ = self.fusion(x, state, memory)
        return x

    def set_backend(self, backend: str, *, recursive: bool = True):
        value = _normalize_backend(backend)
        self.backend = value
        for layer in self.layers:
            for module in (layer.mixer, layer.ffn):
                target = module.module if isinstance(module, _CustomFFNAdapter) else module
                setter = getattr(target, "set_backend", None)
                if callable(setter):
                    try:
                        setter(value, recursive=recursive)
                    except TypeError:
                        setter(value)
                elif hasattr(target, "backend"):
                    try:
                        target.backend = value
                    except Exception:
                        pass
        return self

    def resolved_backend(self) -> str:
        values: list[str] = []
        for layer in self.layers:
            for module in (layer.mixer, layer.ffn):
                target = module.module if isinstance(module, _CustomFFNAdapter) else module
                resolver = getattr(target, "resolved_backend", None)
                if callable(resolver):
                    try:
                        values.append(str(resolver()))
                    except Exception:
                        values.append("unavailable")
        if not values:
            return self.backend
        return values[0] if len(set(values)) == 1 else "mixed"

    def to_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "width": list(self.widths),
            "depth": self.depth,
            "mixer": list(self._mixer_specs),
            "ffn": list(self._ffn_specs),
            "mixer_config": copy.deepcopy(list(self._mixer_configs)),
            "ffn_config": copy.deepcopy(list(self._ffn_configs)),
            "backend": self.backend,
            "precision": self.precision,
        }

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, depth={self.depth}, widths={list(self.widths)}, "
            f"mixers={list(self._mixer_specs)}, ffns={list(self._ffn_specs)}"
        )
