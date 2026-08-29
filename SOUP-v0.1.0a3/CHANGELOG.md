# Changelog

## 0.1.0a3

- Restored the v0.1.0a1 training fast path: tensor-contract checks stay in `validate()` rather than repeated `forward()`.
- Restored the equal-width training execution path that skips registered `Identity` state bridges entirely.
- Restored the construction-resolved bridged training path with no per-layer bridge-presence branch.
- Preserved construction-time FFN adapter/calling-convention resolution.

- Added validated packed recurrent generation fast path via `prepare_generation()`.
- Packed ESA qgv + SAFFN x projection.
- Cached static SAFFN generation terms.
- Removed zero first-layer state-projection work during one-token decode.
- Kept training/forward equations and trainable parameter/state-dict structure unchanged.
- Added recurrent `prefill()` / `decode_step()` and `lightning_prefill` / `lightning_step` aliases.
- Packed Observer state-write/write-gate and key/value generation projections.
