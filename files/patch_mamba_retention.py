"""Backport vLLM #54713 at d3bdc5f7cc6011f0ad30b80b417aa6cdbb4217ab.

Apply both upstream commits to pristine pinned-image sources, never live files.
git apply rejects context drift; generated overlays are syntax checked.
"""
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
NAMES = ("single_type_kv_cache_manager", "kv_cache_coordinator")

def generate():
    patch = (HERE / "vllm-54713.patch").read_text()
    # Pinned 8e685d198 predates the cache_hit_alignment_tokens rename.
    # Keep its scheduler alignment; change only this hunk's context.
    anchor = "             alignment_tokens=self.cache_hit_alignment_tokens,"
    if patch.count(anchor) != 1:
        raise RuntimeError("Upstream patch context drift")
    patch = patch.replace(anchor, "             alignment_tokens=self.scheduler_block_size,")
    with tempfile.TemporaryDirectory(prefix="mamba-retention-") as temp:
        root = Path(temp)
        core = root / "vllm/v1/core"
        core.mkdir(parents=True)
        for name in NAMES:
            (core / f"{name}.py").write_bytes(
                (HERE / f"{name}_54713.orig").read_bytes())
        commits = re.split(r"(?m)(?=^From [0-9a-f]{40} Mon Sep)", patch)
        for commit in filter(str.strip, commits):
            subprocess.run(["git", "apply", "--include=vllm/v1/core/*"],
                           input=commit, text=True, cwd=root, check=True)
        for name in NAMES:
            source = (core / f"{name}.py").read_text()
            compile(source, name, "exec")
            (HERE / f"{name}_54713.py").write_text(source)
    print("Mamba retention #54713: both pinned commits applied")

if __name__ == "__main__":
    generate()
