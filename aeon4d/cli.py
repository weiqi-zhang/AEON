"""Aeon command line: text / image -> Wan2.1 video latent -> 4D Gaussian Splatting.

Examples:
    aeon-generate --prompt "A red fox running through snow" --spiral --save_4dgs
    aeon-generate --prompts prompts.txt --output_dir outputs/batch
    aeon-generate --pipeline i2v --pairs pairs.tsv --cpu_offload --spiral
"""

import argparse
import logging
import os

import torch

from aeon4d.export import (
    save_4dgs_checkpoint,
    save_cov_t_distribution,
    save_generation,
    save_spiral_render,
    save_static_3dgs_checkpoint,
)
from aeon4d.inputs import read_pairs_file, sample_dir_name
from aeon4d.latent_cache import load_video_latent_cache, save_video_latent_cache, video_latent_cache_path
from aeon4d.pipeline import (
    DEFAULT_CONFIG,
    DEFAULT_WAN_MODELS,
    denoise_video_latents,
    load_config,
    load_aeon,
    load_wan_pipeline,
    reconstruct_4dgs,
    resolve_path,
    zero_dxyzt_,
)
from aeon4d.rendering import build_moving_anchor_spiral_c2ws, render_4dgs

logger = logging.getLogger("aeon4d")

I2V_DEFAULT_HW = (480, 832)  # Wan2.1-I2V-14B-480P native resolution


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aeon-generate",
        description="Generate 4D Gaussian scenes from text (Wan2.1 T2V) or image + text (Wan2.1 I2V).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("input (T2V: --prompt / --prompts; I2V: --pairs, or --image with --prompt / --prompts)")
    g.add_argument("--prompt", type=str, default=None, help="Single text prompt.")
    g.add_argument("--prompts", "--input_texts_path", dest="input_texts_path", type=str, default=None,
                   help="Text file, one prompt per line.")
    g.add_argument("--image", "--input_image", dest="input_image", type=str, default=None,
                   help="I2V: first-frame image shared by all prompts.")
    g.add_argument("--pairs", "--input_pairs_path", dest="input_pairs_path", type=str, default=None,
                   help="I2V: per-sample '<image>\\t<prompt>' TSV (or JSONL with 'image'/'prompt').")

    g = p.add_argument_group("models")
    g.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Aeon model config (YAML).")
    g.add_argument("--weights", type=str, default=None, help="Aeon weights (.safetensors); default: from config.")
    g.add_argument("--pipeline", "--pipeline_type", dest="pipeline_type", choices=["t2v", "i2v"], default="t2v")
    g.add_argument("--wan_model", "--model_id", dest="model_id", type=str, default=None,
                   help=f"Wan2.1 Diffusers model (path or HF id). Default: {DEFAULT_WAN_MODELS}.")
    g.add_argument("--cpu_offload", action="store_true",
                   help="Diffusers model CPU offload for the Wan pipeline (recommended for the 14B I2V model).")
    g.add_argument("--device", type=str, default="cuda:0")

    g = p.add_argument_group("video generation")
    g.add_argument("--num_frames", type=int, default=81)
    g.add_argument("--steps", "--num_inference_steps", dest="num_inference_steps", type=int, default=50)
    g.add_argument("--cfg", "--cfg_scale", dest="cfg_scale", type=float, default=5.0)
    g.add_argument("--flow_shift", type=float, default=3.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--negative_prompt", type=str, default="Blurred, artifacts, low quality, distorted, deformed")
    g.add_argument("--height", "--vae_height", dest="vae_height", type=int, default=None,
                   help="Video height. Default: T2V = config vae_resolution, I2V = 480.")
    g.add_argument("--width", "--vae_width", dest="vae_width", type=int, default=None,
                   help="Video width. Default: T2V = config vae_resolution, I2V = 832.")
    g.add_argument("--no_latent_cache", dest="cache_video_latent", action="store_false",
                   help="Do not read/write <sample_dir>/wan_latents.pt (by default a cached latent skips diffusion).")
    g.add_argument("--force_denoise", dest="force_video_latent_denoise", action="store_true",
                   help="Ignore an existing cached latent, regenerate and overwrite it.")

    g = p.add_argument_group("4DGS decoding / rendering / export")
    g.add_argument("--output_dir", type=str, default="outputs")
    g.add_argument("--index_offset", type=int, default=0,
                   help="Offset for sample directory numbering (for sharding prompts over several runs).")
    g.add_argument("--keep_dxyzt", dest="zero_dxyzt", action="store_false",
                   help="Keep the predicted dxyzt position residual (default: zeroed, as in the reference setup).")
    g.add_argument("--decode_k_mode", choices=["none", "gaussian", "median", "mean", "first"], default="gaussian",
                   help="Decode time: smooth the per-frame intrinsics recovered from the predicted rays along time "
                        "before placing the Gaussians (suppresses zoom breathing).")
    g.add_argument("--decode_k_sigma", type=float, default=5.0, help="Temporal sigma (frames) for --decode_k_mode gaussian.")
    g.add_argument("--fix_k_mode", choices=["none", "median", "first", "mean"], default="none",
                   help="Render time: aggregate the per-frame intrinsics along time before rasterization.")
    g.add_argument("--bg_color", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("R", "G", "B"),
                   help="Render background in [0,1]; black matches training.")
    g.add_argument("--render_h", type=int, default=None, help="Render height (default: config ff_resolution).")
    g.add_argument("--render_w", type=int, default=None, help="Render width (default: config ff_resolution).")
    g.add_argument("--save_4dgs", action="store_true", help="Save gs_params.pt + scene_info.json under <sample>/4dgs.")
    g.add_argument("--save_3dgs", action="store_true",
                   help="Save a static (time-collapsed) Inria-format point_cloud.ply under <sample>/3dgs.")
    g.add_argument("--save_cov_t_plot", action="store_true", help="Save the temporal-scale (cov_t) histogram.")

    g = p.add_argument_group("spiral novel-view rendering (defaults = Aeon main experiment)")
    g.add_argument("--spiral", "--spiral_render", dest="spiral_render", action="store_true",
                   help="Also render the 4DGS along a spiral camera path while time advances (<sample>/spiral).")
    g.add_argument("--spiral_motion_mode", choices=["moving_look_forward", "moving_keep"], default="moving_look_forward",
                   help="Camera anchored to each generated frame; look_forward re-orients toward that frame's "
                        "forward focal point, keep keeps the frame's rotation.")
    g.add_argument("--spiral_n_rots", type=int, default=1)
    g.add_argument("--spiral_zrate", type=float, default=0.0)
    g.add_argument("--spiral_radii", type=float, nargs=3, default=[0.06, 0.06, 0.0], metavar=("RX", "RY", "RZ"))
    g.add_argument("--spiral_look_forward_focal", type=float, default=0.65)
    g.add_argument("--spiral_zoom", type=float, default=1.45, help="Focal zoom for spiral renders (>1 narrows the FOV).")

    g = p.add_argument_group("motion mask (optional extra render channel from the Gaussian velocity field)")
    g.add_argument("--motion_mask", action="store_true")
    g.add_argument("--no_motion_mask_ms3_gate", action="store_true")
    g.add_argument("--motion_mask_th", type=float, default=0.25)
    g.add_argument("--motion_mask_cov_t_min", type=float, default=None)
    g.add_argument("--motion_mask_cov_t_max", type=float, default=None)
    g.add_argument("--motion_mask_flow_residual_th", type=float, default=None)
    g.add_argument("--motion_mask_alpha_th", type=float, default=None)

    # Fixed values recorded in the latent-cache metadata.
    p.set_defaults(
        transformer_lora_path=None,
        video_latent_cache_name="wan_latents.pt",
        video_latent_cache_root=None,
        strict_video_latent_cache_metadata=False,
    )
    return p


def finalize_args(args, config):
    st = config["stitching"]
    args.vae_resolution = int(st["vae_resolution"])
    args.ff_resolution = int(st["ff_resolution"])
    args.model_id = args.model_id or DEFAULT_WAN_MODELS[args.pipeline_type]

    if (args.vae_height is None) != (args.vae_width is None):
        raise SystemExit("--height and --width must be given together.")
    if args.vae_height is None:
        args.vae_height, args.vae_width = (
            I2V_DEFAULT_HW if args.pipeline_type == "i2v" else (args.vae_resolution, args.vae_resolution)
        )
    for name in ("vae_height", "vae_width"):  # Wan requires H, W % 16 == 0
        value = getattr(args, name)
        if value % 16:
            logger.warning(f"{name}={value} is not divisible by 16 (Wan requirement); using {value // 16 * 16}.")
            setattr(args, name, value // 16 * 16)

    args.render_h_eff = args.render_h or args.ff_resolution
    args.render_w_eff = args.render_w or args.ff_resolution


def collect_samples(args):
    """Normalize all input modes to a list of (image_path_or_None, prompt)."""
    if args.input_pairs_path is not None:
        if args.pipeline_type != "i2v":
            raise SystemExit("--pairs requires --pipeline i2v")
        if args.prompt is not None or args.input_texts_path is not None or args.input_image is not None:
            raise SystemExit("--pairs is mutually exclusive with --prompt / --prompts / --image")
        return read_pairs_file(args.input_pairs_path)
    if args.pipeline_type == "i2v":
        if not args.input_image:
            raise SystemExit("--pipeline i2v needs --pairs, or --image together with --prompt / --prompts")
        if not os.path.isfile(args.input_image):
            raise SystemExit(f"--image not found: {args.input_image}")
    elif args.input_image:
        raise SystemExit("--image is only used with --pipeline i2v")
    if args.prompt is not None and args.input_texts_path is not None:
        raise SystemExit("--prompt and --prompts are mutually exclusive")
    if args.prompt is not None:
        return [(args.input_image, args.prompt)]
    if args.input_texts_path is not None:
        with open(args.input_texts_path) as f:
            return [(args.input_image, line.strip()) for line in f if line.strip()]
    raise SystemExit("No input given: use --prompt, --prompts or --pairs")


def render_kwargs(args):
    return dict(
        render_h=args.render_h_eff,
        render_w=args.render_w_eff,
        fix_k_mode=args.fix_k_mode,
        bg_color=tuple(args.bg_color),
        motion_mask=args.motion_mask,
        motion_mask_th=args.motion_mask_th,
        motion_mask_cov_t_min=args.motion_mask_cov_t_min,
        motion_mask_cov_t_max=args.motion_mask_cov_t_max,
        motion_mask_use_ms3=not args.no_motion_mask_ms3_gate,
        motion_mask_flow_residual_th=args.motion_mask_flow_residual_th,
        motion_mask_alpha_th=args.motion_mask_alpha_th,
    )


def spiral_cameras(args, res, device):
    return build_moving_anchor_spiral_c2ws(
        res,
        n_rots=args.spiral_n_rots,
        zrate=args.spiral_zrate,
        radii=args.spiral_radii,
        rotation_mode="look_forward" if args.spiral_motion_mode == "moving_look_forward" else "keep",
        look_forward_focal=args.spiral_look_forward_focal,
        device=device,
    )


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    finalize_args(args, config)
    samples = collect_samples(args)
    weights = resolve_path(args.weights or config["weights"])
    if not os.path.isfile(weights):  # fail before spending a minute on loading Wan
        raise SystemExit(f"Aeon weights not found: {weights}\n"
                         "Download aeon.safetensors and pass its path with --weights.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)  # Wan noise for sample k depends on this seed and on samples 0..k-1

    pipe = load_wan_pipeline(args.model_id, args.pipeline_type, args.flow_shift, device, cpu_offload=args.cpu_offload)
    model = load_aeon(config, weights, device,
                        decode_k_mode=args.decode_k_mode, decode_k_sigma=args.decode_k_sigma)

    images = {}  # path -> PIL image, I2V conditioning images
    for idx, (img_path, prompt) in enumerate(samples):
        global_idx = idx + args.index_offset
        logger.info(f"[{idx + 1}/{len(samples)}] {prompt[:80]}" + (f"  (image: {img_path})" if img_path else ""))
        sample_dir = os.path.join(args.output_dir, f"{global_idx:03d}_{sample_dir_name(prompt)}")
        cache_path = video_latent_cache_path(sample_dir, args)

        # 1) Wan diffusion -> video latent (or reuse the cached latent)
        latents = load_video_latent_cache(cache_path, args, prompt, img_path, global_idx, device)
        if latents is None:
            image = None
            if args.pipeline_type == "i2v":
                if img_path not in images:
                    from PIL import Image

                    images[img_path] = Image.open(img_path).convert("RGB")
                image = images[img_path]
            latents = denoise_video_latents(
                pipe, prompt, image=image, height=args.vae_height, width=args.vae_width,
                num_frames=args.num_frames, num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale, negative_prompt=args.negative_prompt,
            )
            save_video_latent_cache(cache_path, args, prompt, img_path, global_idx, latents)

        # 2) Decode + Aeon decoder -> 4D Gaussians
        decoded_video, views, res, gs_params = reconstruct_4dgs(
            model, pipe.vae, latents, vae_resolution=args.vae_resolution,
            ff_resolution=args.ff_resolution, device=device,
        )
        if args.zero_dxyzt:
            zero_dxyzt_(gs_params)

        # 3) Render the input views and save
        logger.info("Rendering 4D Gaussians...")
        render_out = render_4dgs(gs_params, res, device, **render_kwargs(args))
        save_generation(sample_dir, prompt, decoded_video.float(), res, gs_params, render_out, args.ff_resolution)
        if args.save_4dgs:
            save_4dgs_checkpoint(os.path.join(sample_dir, "4dgs"), gs_params, res, views, args.ff_resolution)
        if args.save_3dgs:
            save_static_3dgs_checkpoint(os.path.join(sample_dir, "3dgs"), gs_params, res, views, args.ff_resolution)
        if args.save_cov_t_plot:
            save_cov_t_distribution(gs_params, sample_dir, tag="4dgs")

        # 4) Optional spiral novel-view video (camera orbits while time advances)
        if args.spiral_render:
            logger.info(f"Rendering spiral novel views (mode={args.spiral_motion_mode})...")
            spiral_out = render_4dgs(
                gs_params, res, device, override_c2ws=spiral_cameras(args, res, device),
                focal_zoom=args.spiral_zoom, **render_kwargs(args),
            )
            save_spiral_render(os.path.join(sample_dir, "spiral"), spiral_out)

        if img_path is not None:
            with open(os.path.join(sample_dir, "input_image.txt"), "w") as f:
                f.write(str(img_path) + "\n")
        logger.info(f"[{idx + 1}/{len(samples)}] done: {sample_dir}")

    logger.info("All generations complete.")


if __name__ == "__main__":
    main()
