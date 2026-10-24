"""Build the Stage 02 Kaggle input ZIP from a real Git commit and frozen inputs.

python -m pipeline.package_e5_kaggle
No network, checkpoint loading, inference or experiment execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/kaggle_inference")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "paac_vn_stage02_kaggle.zip"
    notebook = output / "PAAC_VN_Stage02_Kaggle.ipynb"
    guide = output / "HUONG_DAN.md"
    if any(p.exists() for p in (archive, notebook, guide)):
        raise FileExistsError("Use a new --output-dir; existing deliverables are preserved.")
    commit = git("rev-parse", "HEAD").decode().strip()
    # Refuse a package whose executable source differs from its recorded commit.
    tracked = ["pipeline/e5_stage02.py", "pipeline/e5_stage02_contract.json",
               "pipeline/embed.py", "pipeline/models.yaml", "pipeline/package_e5_kaggle.py",
               "kaggle/requirements.txt", "kaggle/PAAC_VN_Stage02_Kaggle.ipynb", "kaggle/README.md"]
    for name in tracked:
        if git("show", f"HEAD:{name}") != (ROOT / name).read_bytes():
            raise RuntimeError(f"Uncommitted or byte-different deployment file: {name}")
    contract = json.loads((ROOT / "pipeline/e5_stage02_contract.json").read_text(encoding="utf-8"))
    payload = []
    for expected in contract["datasets"].values():
        path = ROOT / expected["path"]
        if sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen dataset hash mismatch: {path}")
        payload.append(path)
    manifest = {"schema_version": 1, "commit": commit,
                "source_sha256": {name: sha256(ROOT / name) for name in tracked},
                "payload": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                             "sha256": sha256(p)} for p in payload],
                "full_run_authorized": False, "execution_scope": "INFERENCE_ONLY_LOCAL_VALIDATION"}
    with tempfile.TemporaryDirectory(dir=output) as temp:
        bundle = Path(temp) / "repo.bundle"
        git("bundle", "create", str(bundle), "HEAD")
        manifest["git_bundle_sha256"] = sha256(bundle)
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                             allowZip64=True) as zipped:
            zipped.write(bundle, "repo.bundle")
            zipped.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
            for path in payload:
                zipped.write(path, "payload/" + path.relative_to(ROOT).as_posix())
    shutil.copyfile(ROOT / "kaggle/PAAC_VN_Stage02_Kaggle.ipynb", notebook)
    shutil.copyfile(ROOT / "kaggle/README.md", guide)
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "archive_sha256": sha256(archive), "commit": commit,
                      "payload_files": len(payload), "notebook": str(notebook)}, indent=2))


if __name__ == "__main__":
    main()
