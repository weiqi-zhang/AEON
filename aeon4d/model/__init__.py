"""Aeon stitched 4DGS decoder: Wan VAE latent -> stitching bridge -> MapAnythingGS -> 4D Gaussians."""

import copy
import logging
import os

from aeon4d._vendor.mapanything.models.mapanything_gs.model import MapAnythingGS
from aeon4d.model.conv_spec import parse_conv_spec
from aeon4d.model.mapanything_stitched import MapAnythingGSStitched
from aeon4d.model.stitch_vae_4d import StitchVAE4D

logger = logging.getLogger(__name__)

__all__ = ["StitchVAE4D", "build_model", "load_model", "set_decode_k_smoothing"]


def build_model(config: dict) -> StitchVAE4D:
    """Build the (randomly initialized) Aeon decoder described by a config (see configs/*.yaml)."""
    st = config["stitching"]
    base = MapAnythingGS(**copy.deepcopy(config["mapanything_gs"]))  # the constructor mutates its config

    conv_spec = parse_conv_spec(st["stitching_layer_config"])
    if st["bridge_type"] == "perceiver":
        # perceiver ignores the conv spec; channels come from bridge_config.out_dim
        stitch_channels = (st.get("bridge_config") or {}).get("out_dim", 1024)
    else:
        stitch_channels = conv_spec.out_channels

    dia = st.get("depth_image_adapter")
    feedforward = MapAnythingGSStitched(
        base,
        stitch_channels=stitch_channels,
        stitch_skip_blocks=st["stitch_skip_blocks"],
        depth_image_adapter_config=dia if dia and dia.get("enabled") else None,
    )
    return StitchVAE4D(
        feedforward,
        stitching_layer_config=conv_spec,
        vae_resolution=st["vae_resolution"],
        ff_resolution=st["ff_resolution"],
        bridge_type=st["bridge_type"],
        bridge_config=st.get("bridge_config"),
    )


def set_decode_k_smoothing(model: StitchVAE4D, mode: str = "gaussian", sigma: float = 5.0) -> None:
    """Smooth the per-frame intrinsics K (recovered from the predicted rays) along time before the
    Gaussians are placed; suppresses zoom "breathing". mode: none | gaussian | median | mean | first.
    The reference setup uses gaussian with sigma = 5 frames."""
    backbone = model.stitched_4d_model.model
    backbone.fix_decode_k_mode = mode
    backbone.fix_decode_k_sigma = float(sigma)


def load_model(config: dict, weights_path: str, device="cuda", decode_k_mode: str = "gaussian",
               decode_k_sigma: float = 5.0) -> StitchVAE4D:
    """Build the decoder, load released weights (safetensors, strict) and set the decode-time K smoothing."""
    from safetensors.torch import load_model as load_safetensors

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Aeon weights not found: {weights_path}")
    model = build_model(config)
    logger.info(f"Loading Aeon weights from {weights_path}")
    load_safetensors(model, weights_path, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    set_decode_k_smoothing(model, decode_k_mode, decode_k_sigma)
    return model
