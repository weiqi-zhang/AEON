# Third-party code

Aeon's own code is released under the MIT License (`LICENSE`). The third-party code listed below keeps
its original license.

`aeon4d/_vendor/` contains the third-party code that Aeon inference executes. Each copy is kept close
to its original source; the changes are listed below so the copies can be compared with upstream.

| Directory | Origin | License |
|---|---|---|
| `_vendor/mapanything/` | MapAnything (Meta Platforms, https://github.com/facebookresearch/map-anything), with modifications: `models/mapanything_gs/`, `renders/`, `utils/geometry.py`, `utils/misc.py`, `acceleration/checkpoint.py` | Apache-2.0 (`LICENSE`, file headers) |
| `_vendor/mapanything/utils/easyvolcap_compat.py` | Helper functions from EasyVolcap (https://github.com/zju3dv/EasyVolcap), `utils/{chunk,base,math,quat,flow}_utils.py` | EasyVolcap license (see the upstream repository) |
| `_vendor/uniception/` | UniCeption (AirLab, https://github.com/castacks/UniCeption) | BSD-3-Clause (`LICENSE`) |
| `_vendor/dinov2/` | DINOv2 (Meta, https://github.com/facebookresearch/dinov2): `models/vision_transformer.py`, `layers/` | Apache-2.0 (`LICENSE`) |

## Modifications

- **All copies:** imports rewritten to `aeon4d._vendor.*`. The `if __name__ == "__main__"` self-tests are removed.
- **`mapanything/models/mapanything_gs/model.py`:**
  - Only the alternating-attention multi-view transformer is kept.
  - The high-level `infer()` API and an unused Gaussian decoder are removed.
  - Decode-time smoothing of the recovered intrinsics is configured through the model attributes `fix_decode_k_mode` and `fix_decode_k_sigma`.
  - A missing import of `multi_gather` (used by `magic_filter(recalc_magic_pattern=True)`) is added.
- **`mapanything/models/mapanything_gs/adaptors.py`:** a missing `import torch` is added.
- **`mapanything/utils/geometry.py` and `utils/misc.py`:** reduced to the functions used at inference and their dependencies. The function bodies are unchanged.
- **`mapanything/renders/`:**
  - The EasyVolcap helpers are imported from `utils/easyvolcap_compat.py`.
  - A commented-out debug block is removed.
- **`uniception/models/encoders/__init__.py`:** the encoder registry is reduced to `dinov2`, `dense_rep_encoder` and `global_rep_encoder`.
- **`uniception/models/encoders/dinov2.py`:**
  - The ViT is built from `_vendor/dinov2` with the same arguments as the `torch.hub` entry points.
  - No pretrained weights are downloaded; all encoder weights come from the Aeon checkpoint.
