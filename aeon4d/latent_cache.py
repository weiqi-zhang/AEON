"""Cache of Wan denoised video latents, so re-running a sample skips diffusion."""

import hashlib
import json
import logging
import os
import re
import torch


logger = logging.getLogger(__name__)


VIDEO_LATENT_CACHE_FORMAT = "aeon_wan_denoised_video_latent"


VIDEO_LATENT_CACHE_VERSION = 1


def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def video_latent_cache_path(sample_dir: str, args) -> str:
    cache_name = getattr(args, "video_latent_cache_name", "wan_latents.pt")
    cache_name = cache_name or "wan_latents.pt"
    if os.path.isabs(cache_name):
        return cache_name
    cache_root = getattr(args, "video_latent_cache_root", None)
    if cache_root:
        sample_name = os.path.basename(os.path.normpath(sample_dir))
        return os.path.join(
            os.path.abspath(os.path.expanduser(cache_root)),
            _video_latent_cache_key(args),
            sample_name,
            cache_name,
        )
    return os.path.join(sample_dir, cache_name)


def _cache_component(value, max_len: int = 80) -> str:
    text = str(value if value is not None else "none")
    text = text.replace(os.sep, "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text).strip("._")
    return (text or "none")[:max_len]


def _cache_float(value) -> str:
    return f"{float(value):g}"


def _video_latent_cache_key(args) -> str:
    signature = {
        "format": VIDEO_LATENT_CACHE_FORMAT,
        "version": VIDEO_LATENT_CACHE_VERSION,
        "pipeline_type": str(args.pipeline_type),
        "model_id": str(args.model_id),
        "transformer_lora_path": str(args.transformer_lora_path) if args.transformer_lora_path else None,
        "num_frames": int(args.num_frames),
        "num_inference_steps": int(args.num_inference_steps),
        "cfg_scale": float(args.cfg_scale),
        "flow_shift": float(args.flow_shift),
        "negative_prompt": str(args.negative_prompt),
        "seed": int(args.seed),
        "vae_height": int(args.vae_height),
        "vae_width": int(args.vae_width),
        "vae_resolution": int(args.vae_resolution),
    }
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    readable = "_".join(
        [
            _cache_component(args.pipeline_type, 16),
            f"f{int(args.num_frames)}",
            f"s{int(args.num_inference_steps)}",
            f"cfg{_cache_float(args.cfg_scale)}",
            f"shift{_cache_float(args.flow_shift)}",
            f"seed{int(args.seed)}",
        ]
    )
    return f"{readable}_{digest}"


def _video_latent_cache_metadata(args, prompt, img_path, global_idx, latents=None):
    meta = {
        "format": VIDEO_LATENT_CACHE_FORMAT,
        "version": VIDEO_LATENT_CACHE_VERSION,
        "prompt": str(prompt),
        "sample_index": int(global_idx),
        "pipeline_type": str(args.pipeline_type),
        "model_id": str(args.model_id),
        "transformer_lora_path": str(args.transformer_lora_path) if args.transformer_lora_path else None,
        "num_frames": int(args.num_frames),
        "num_inference_steps": int(args.num_inference_steps),
        "cfg_scale": float(args.cfg_scale),
        "flow_shift": float(args.flow_shift),
        "negative_prompt": str(args.negative_prompt),
        "seed": int(args.seed),
        "vae_height": int(args.vae_height),
        "vae_width": int(args.vae_width),
        "vae_resolution": int(args.vae_resolution),
        "input_image": os.path.abspath(img_path) if img_path else None,
        "latent_space": "wan_vae_denormalized",
        "latent_layout": "B,C,T,H,W",
    }
    if isinstance(latents, torch.Tensor):
        meta["latent_shape"] = [int(x) for x in latents.shape]
        meta["latent_dtype"] = str(latents.dtype)
    return meta


def _video_latent_cache_metadata_mismatches(cached_meta, expected_meta):
    if not isinstance(cached_meta, dict):
        return []
    keys = (
        "prompt",
        "sample_index",
        "pipeline_type",
        "model_id",
        "transformer_lora_path",
        "num_frames",
        "num_inference_steps",
        "cfg_scale",
        "flow_shift",
        "negative_prompt",
        "seed",
        "vae_height",
        "vae_width",
        "vae_resolution",
        "input_image",
    )
    mismatches = []
    for k in keys:
        if cached_meta.get(k) != expected_meta.get(k):
            mismatches.append(f"{k}: cached={cached_meta.get(k)!r}, current={expected_meta.get(k)!r}")
    return mismatches


def _warn_video_latent_cache_mismatch(cache_path, mismatches, strict=False):
    if mismatches:
        shown = "; ".join(mismatches[:5])
        if len(mismatches) > 5:
            shown += f"; ... (+{len(mismatches) - 5} more)"
        action = "refusing this cache" if strict else "loading it anyway because the cache file exists in this sample folder"
        logger.warning(
            f"Video latent cache metadata differs for {cache_path}; {action}. {shown}"
        )


def load_video_latent_cache(cache_path, args, prompt, img_path, global_idx, device):
    if not getattr(args, "cache_video_latent", True):
        return None
    if getattr(args, "force_video_latent_denoise", False):
        if os.path.isfile(cache_path):
            logger.info(f"Ignoring existing video latent cache due to --force_video_latent_denoise: {cache_path}")
        return None
    if not os.path.isfile(cache_path):
        return None

    try:
        payload = _torch_load_compat(cache_path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            latents = payload
            cached_meta = {}
        elif isinstance(payload, dict):
            latents = payload.get("latents", payload.get("latent"))
            cached_meta = payload.get("metadata", {})
        else:
            raise TypeError(f"unsupported cache payload type: {type(payload).__name__}")

        if not isinstance(latents, torch.Tensor):
            raise TypeError("cache payload does not contain a tensor under 'latents'")
        if latents.ndim != 5:
            raise ValueError(f"expected latent tensor with 5 dims (B,C,T,H,W), got shape={tuple(latents.shape)}")

        expected_meta = _video_latent_cache_metadata(args, prompt, img_path, global_idx)
        strict_meta = bool(
            getattr(args, "strict_video_latent_cache_metadata", False)
            or getattr(args, "video_latent_cache_root", None)
        )
        mismatches = _video_latent_cache_metadata_mismatches(cached_meta, expected_meta)
        _warn_video_latent_cache_mismatch(cache_path, mismatches, strict=strict_meta)
        if mismatches and strict_meta:
            return None

        latents = latents.contiguous().to(device, non_blocking=True)
        logger.info(
            f"Loaded cached Wan denoised latent -> {cache_path} "
            f"(shape={tuple(latents.shape)}, dtype={latents.dtype}); skipping diffusion denoising."
        )
        return latents
    except Exception as e:
        logger.warning(f"Could not load video latent cache {cache_path}: {e}; falling back to Wan denoising.")
        return None


def save_video_latent_cache(cache_path, args, prompt, img_path, global_idx, latents):
    if not getattr(args, "cache_video_latent", True):
        return
    if not isinstance(latents, torch.Tensor):
        logger.warning("Skipping video latent cache save: latents is not a tensor.")
        return

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        meta = _video_latent_cache_metadata(args, prompt, img_path, global_idx, latents=latents)
        payload = {
            "format": VIDEO_LATENT_CACHE_FORMAT,
            "version": VIDEO_LATENT_CACHE_VERSION,
            "metadata": meta,
            "latents": latents.detach().cpu(),
        }
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
        logger.info(f"Saved Wan denoised latent cache -> {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save video latent cache {cache_path}: {e}")

