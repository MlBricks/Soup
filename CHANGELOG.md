# Changelog

## 0.1.0a0

- Initial standalone SOUP library.
- Preserves the supplied notebook's SOUP State-Aware FFN, Observer State Memory and SOUP Fusion equations.
- Makes the per-layer mixer configurable (`esa`, `bolt`, or custom).
- Makes the per-layer FFN configurable (`saffn`, conventional `ffn`, or custom).
- Keeps Observer State Memory and SOUP Fusion fixed as defining SOUP blocks.
- Supports scalar or per-layer width/mixer/FFN/config broadcasting.
- Supports MLBricks `auto`, `native`, and `pytorch` backend policy.
- Leaves BOLT `latent_dim` unspecified unless the user explicitly provides it, preserving BOLT's own default.
