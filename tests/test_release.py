"""Release sanity tests (CPU only):  python -m pytest tests -q

End-to-end GPU equivalence against reference outputs is checked with scripts/compare_outputs.py.
"""

import os

import pytest

from aeon4d.cli import build_parser, collect_samples, finalize_args
from aeon4d.pipeline import DEFAULT_CONFIG, load_config, resolve_path

CONFIG = load_config(DEFAULT_CONFIG)
WEIGHTS = resolve_path(CONFIG["weights"])


def test_config():
    # cov_t_min must be 0.03 for this checkpoint (see the comment at the top of the config)
    assert CONFIG["mapanything_gs"]["pred_head_config"]["cov_t_min"] == 0.03
    assert CONFIG["stitching"]["bridge_type"] == "dual_conv3d_perceiver"
    assert CONFIG["stitching"]["stitch_skip_blocks"] == 4


def _args(argv):
    args = build_parser().parse_args(argv)
    finalize_args(args, CONFIG)
    return args


def test_cli_defaults_t2v():
    args = _args(["--prompt", "a cat"])
    assert (args.vae_height, args.vae_width) == (512, 512)
    assert args.model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert args.decode_k_mode == "gaussian" and args.fix_k_mode == "none" and args.zero_dxyzt
    assert collect_samples(args) == [(None, "a cat")]


def test_cli_defaults_i2v(tmp_path):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("img.png\ta dog\n")
    args = _args(["--pipeline", "i2v", "--pairs", str(pairs)])
    assert (args.vae_height, args.vae_width) == (480, 832)
    assert args.model_id == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    assert collect_samples(args) == [("img.png", "a dog")]


def test_cli_rejects_bad_input():
    with pytest.raises(SystemExit):
        collect_samples(_args(["--pipeline", "i2v", "--prompt", "x"]))  # I2V without an image
    with pytest.raises(SystemExit):
        collect_samples(_args([]))  # no input at all


@pytest.mark.skipif(not os.path.isfile(WEIGHTS), reason="release weights not downloaded")
def test_build_and_strict_load():
    from aeon4d.model import load_model

    model = load_model(CONFIG, WEIGHTS, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    assert abs(n_params - 811.8e6) < 1e6
    assert not model.training
    assert model.stitched_4d_model.model.fix_decode_k_mode == "gaussian"
    assert len(model.stitched_4d_model.model.encoder.model.blocks) == 20  # 24 DINOv2 blocks - 4 skipped
