"""Input collection: prompts files, I2V (image, prompt) pairs files, sample directory names."""



def read_pairs_file(path: str):
    """Read an (image, prompt) pairs file for per-prompt I2V conditioning.

    Format auto-detected by extension:
      - ``.jsonl`` / ``.json`` : one JSON object per line with keys ``image`` and ``prompt``
      - otherwise              : TSV — each non-empty line is ``<image_path>\\t<prompt>``

    Empty lines and lines starting with ``#`` are skipped. Returns a list of
    ``(image_path, prompt)`` tuples preserving file order.
    """
    import json
    samples = []
    is_jsonl = path.lower().endswith((".jsonl", ".json"))
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if is_jsonl:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(
                        f"[ERROR] pairs file {path}:{lineno}: invalid JSON — {e}"
                    ) from None
                if "image" not in obj or "prompt" not in obj:
                    raise SystemExit(
                        f"[ERROR] pairs file {path}:{lineno}: missing 'image' or 'prompt' key"
                    )
                samples.append((str(obj["image"]), str(obj["prompt"])))
            else:
                if "\t" not in line:
                    raise SystemExit(
                        f"[ERROR] pairs file {path}:{lineno}: TSV expects "
                        f"'<image_path>\\t<prompt>' but no tab found. "
                        f"(Use .jsonl extension for JSONL format.)"
                    )
                img, prompt = line.split("\t", 1)
                samples.append((img.strip(), prompt.strip()))
    if not samples:
        raise SystemExit(f"[ERROR] pairs file {path}: no valid samples found")
    return samples


def sample_dir_name(prompt: str) -> str:
    """Keep sample directory naming compatible with the historical behavior."""
    return str(prompt)[:80].replace("/", "").replace(" ", "_")

