"""Upload a whitelisted subset of this repo to a Hugging Face Space.
Usage: python scripts/deploy_space.py <user/space> [--dry-run]   (HF_TOKEN in the environment)"""
import os, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "frontend", "configs"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")

def build(dest: Path):
    for d in INCLUDE:
        shutil.copytree(ROOT / d, dest / d, ignore=IGNORE)
    shutil.copyfile(ROOT / "deploy" / "space_requirements.txt", dest / "requirements.txt")
    shutil.copyfile(ROOT / "deploy" / "packages.txt", dest / "packages.txt")
    shutil.copyfile(ROOT / "deploy" / "SPACE_README.md", dest / "README.md")

if __name__ == "__main__":
    repo_id, dry = sys.argv[1], "--dry-run" in sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); build(tmp)
        files = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file())
        print(len(files), "files:"); print("\n".join(files))
        if not dry:
            from huggingface_hub import HfApi
            HfApi().upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="space",
                                  token=os.environ["HF_TOKEN"], commit_message="Deploy from GitHub repo")
