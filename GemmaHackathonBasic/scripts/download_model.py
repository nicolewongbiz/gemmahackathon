"""Download the Gemma LiteRT-LM model into models/gemma.litertlm.

    python scripts/download_model.py                 # Gemma 4 E4B (default, fits 8 GB GPU)
    python scripts/download_model.py --repo litert-community/gemma-4-E2B-it-litert-lm   # smaller/faster
Set HF_TOKEN for faster, unthrottled downloads.
"""
import argparse, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=config.LITERT_HF_REPO)
a = ap.parse_args()
dest = config.ROOT / "models"
print(f"Downloading {a.repo} ... (a few GB, be patient)")
local = Path(snapshot_download(repo_id=a.repo, local_dir=dest / a.repo.split("/")[-1],
                               allow_patterns=["*.litertlm", "*.json", "*.md"]))
files = sorted(local.rglob("*.litertlm"), key=lambda p: p.stat().st_size)
if not files:
    sys.exit(f"No .litertlm file found in {local}. Check the repo page for the file name.")
print("Found:", *[f.name for f in files], sep="\n  ")
# Prefer a general (non web / non npu-specific) file if several exist.
pick = next((f for f in files if not any(k in f.name.lower() for k in ("web", "qualcomm", "npu", "mtk"))), files[-1])
shutil.copy2(pick, config.LITERT_MODEL)
print(f"\nModel ready: {config.LITERT_MODEL}  (from {pick.name})")
