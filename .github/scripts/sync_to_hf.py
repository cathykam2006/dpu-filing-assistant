"""
Copy this repo's Streamlit app to a Hugging Face Space (Docker) so the app has
a second home besides Streamlit Community Cloud.

Run by .github/workflows/sync-to-hf.yml. Needs the repo secret HF_TOKEN
(a Hugging Face "write" token). The Space is created on the first run.

Settings come from environment variables set in the workflow:
  HF_SPACE      Space name, e.g. "dpu-filing-assistant"
  HF_TITLE      title shown on the Space card
  HF_EMOJI      emoji shown on the Space card
  HF_SHORT      short description (max 60 characters)
  APP_FILE      Streamlit entry file, e.g. "app.py"
  HF_EXCLUDE    comma-separated folders/files not to copy (optional)
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
ALWAYS_EXCLUDE = {".git", ".github", "__pycache__", ".venv", ".pytest_cache"}

DOCKERFILE = """\
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

EXPOSE 7860
CMD ["streamlit", "run", "{app}", \\
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \\
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", \\
     "--browser.gatherUsageStats=false"]
"""


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN secret is not set yet - skipping the Hugging Face sync.")
        return 0
    space = os.environ["HF_SPACE"]
    app = os.environ.get("APP_FILE", "app.py")
    exclude = ALWAYS_EXCLUDE | {x.strip() for x in os.environ.get("HF_EXCLUDE", "").split(",")
                                if x.strip()}

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{space}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for item in ROOT.iterdir():
            if item.name in exclude:
                continue
            dest = out / item.name
            if item.is_dir():
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns(*ALWAYS_EXCLUDE))
            else:
                shutil.copy2(item, dest)

        for rel in exclude:                      # nested excludes such as eval/results
            target = out / rel
            if "/" in rel and target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()

        (out / "Dockerfile").write_text(DOCKERFILE.replace("{app}", app))
        readme = ROOT / "README.md"
        body = readme.read_text(encoding="utf-8") if readme.exists() else ""
        q = lambda v: json.dumps(v, ensure_ascii=False)    # safe YAML scalars
        front = (f"---\ntitle: {q(os.environ.get('HF_TITLE', space))}\n"
                 f"emoji: {os.environ.get('HF_EMOJI', '📊')}\n"
                 "colorFrom: blue\ncolorTo: gray\nsdk: docker\napp_port: 7860\n"
                 f"short_description: {q(os.environ.get('HF_SHORT', '')[:60])}\n"
                 "pinned: false\n---\n\n")
        (out / "README.md").write_text(front + body, encoding="utf-8")

        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(out),
                          delete_patterns="*",
                          commit_message=f"Sync from GitHub {os.environ.get('GITHUB_SHA', '')[:7]}")

    print(f"Synced to https://huggingface.co/spaces/{repo_id}")
    print(f"App link: https://{user.lower()}-{space.lower()}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
