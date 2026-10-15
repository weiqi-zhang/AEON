"""Model loading and the per-sample computation: Wan video latent -> Aeon decoder -> 4D Gaussians."""

import logging
import os

import torch
import torch.nn.functional as F
import yaml

from aeon4d.model import load_model
from aeon4d.rendering import build_dummy_views

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "aeon.yaml")
DEFAULT_WAN_MODELS = {
    "t2v": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "i2v": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> str:
    """Relative paths are looked up in the working directory first, then in the repository root."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(REPO_ROOT, path)


def load_wan_pipeline(model_id: str, pipeline_type: str, flow_shift: float, device, cpu_offload: bool = False):
    """Load a Wan2.1 Diffusers pipeline (T2V: WanPipeline, I2V: WanImageToVideoPipeline)."""
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    logger.info(f"Loading Wan pipeline from {model_id} (type={pipeline_type})...")
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        num_train_timesteps=1000,
        use_flow_sigmas=True,
        flow_shift=flow_shift,
    )
    # VAE must use float32 to avoid noise/mosaic artifacts (per official Wan examples)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    if pipeline_type == "t2v":
        from diffusers import WanPipeline as pipeline_cls
    elif pipeline_type == "i2v":
        from diffusers import WanImageToVideoPipeline as pipeline_cls
    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")
    pipe = pipeline_cls.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.transformer.eval()
    pipe.scheduler = scheduler
    if cpu_offload:
        # Components live on the CPU and move to the GPU only while in use (needed for the 14B I2V
        # model on GPUs with < ~90 GB free). VAE encode/decode trigger the offload hook.
        pipe.enable_model_cpu_offload(device=device)
        logger.info("Model CPU offload enabled for the Wan pipeline")
    else:
        pipe.to(device)
    return pipe


def load_aeon(config: dict, weights: str, device, decode_k_mode: str = "gaussian", decode_k_sigma: float = 5.0):
    """Load the Aeon stitched 4DGS decoder."""
    logger.info("Loading Aeon decoder...")
    return load_model(config, resolve_path(weights), device, decode_k_mode=decode_k_mode, decode_k_sigma=decode_k_sigma)


def denoise_video_latents(pipe, prompt: str, *, image=None, height: int, width: int, num_frames: int,
                          num_inference_steps: int, guidance_scale: float, negative_prompt: str):
    """Run Wan diffusion and return VAE-ready (denormalized) latents (1, 16, T_vae, H/8, W/8)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        call_kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=[negative_prompt],
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="latent",
        )
        if image is not None:
            # WanImageToVideoPipeline consumes `image` as the first-frame condition
            call_kwargs["image"] = image
        latents = pipe(**call_kwargs)["frames"]

        # Denormalize latents; the latent cache stores this VAE-ready tensor.
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
    return latents


def reconstruct_4dgs(model, vae, latents, *, vae_resolution: int, ff_resolution: int, device):
    """Decode the latent video and run the Aeon decoder.

    Returns (decoded_video (1, 3, T, H, W) in [-1, 1], views, res, gs_params).
    """
    # Decode to pixels: feed-forward image for the Gaussian heads' image adapters + visualization
    with torch.no_grad():
        decoded_video = vae.decode(latents.to(vae.dtype), return_dict=False)[0]

    num_frames = decoded_video.shape[2]
    with torch.no_grad():
        feedforward_image = F.interpolate(
            decoded_video.float(),
            size=(num_frames, ff_resolution, ff_resolution),
            mode="trilinear",
            align_corners=False,
        )
    views = build_dummy_views(feedforward_image, num_frames, device)

    # The dual bridge has a fixed output grid trained on square latents (vae_resolution / 8).
    # Non-square I2V latents (480P: 60x104) are bilinearly resized to that grid; the decoded
    # video above keeps Wan's native aspect ratio.
    stitch_target = vae_resolution // 8
    stitch_latent = latents
    if stitch_latent.shape[-1] != stitch_target or stitch_latent.shape[-2] != stitch_target:
        B, C, T_lat, H_lat, W_lat = stitch_latent.shape
        logger.info(
            f"Coercing latent spatial {H_lat}x{W_lat} -> {stitch_target}x{stitch_target} "
            f"for stitching (trained on vae_resolution={vae_resolution})."
        )
        _flat = stitch_latent.permute(0, 2, 1, 3, 4).reshape(B * T_lat, C, H_lat, W_lat)
        _resized = F.interpolate(_flat, size=(stitch_target, stitch_target), mode="bilinear", align_corners=False)
        stitch_latent = _resized.reshape(B, T_lat, C, stitch_target, stitch_target).permute(0, 2, 1, 3, 4).contiguous()

    logger.info("Running Aeon decoder...")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        res, gs_params = model.forward_with_latent(
            stitch_latent.float().to(device),
            views=views,
            num_frames=num_frames,
            feedforward_image=feedforward_image.to(device),
        )

    if "recovered_K" in gs_params:
        K = gs_params["recovered_K"][0]  # [T, 3, 3]
        fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
        logger.info(
            f"Recovered intrinsics: fx std/mean={(fx.std() / fx.mean()).item():.5f} "
            f"range=[{fx.min():.2f}, {fx.max():.2f}], fy std/mean={(fy.std() / fy.mean()).item():.5f}, "
            f"cx std={cx.std().item():.3f}, cy std={cy.std().item():.3f}, T={K.shape[0]}"
        )
    return decoded_video, views, res, gs_params


def zero_dxyzt_(gs_params: dict) -> None:
    """Remove the dxyzt residual from xyz / t in place, so Gaussians stay at the pure ray * depth
    positions (recommended for generated videos; disable with --keep_dxyzt)."""
    if "dxyzt" not in gs_params:
        return
    d = gs_params["dxyzt"]
    if "xyz" in gs_params:
        gs_params["xyz"] = gs_params["xyz"] - d[..., :3]
    if "t" in gs_params:
        gs_params["t"] = gs_params["t"] - d[..., 3:]
    gs_params["dxyzt"] = torch.zeros_like(d)
    logger.info(f"Zeroed dxyzt residual (|d|.mean={d.abs().mean().item():.4g})")
