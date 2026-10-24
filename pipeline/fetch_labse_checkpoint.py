"""Fetch only the pinned LaBSE PyTorch assets using system curl on Windows.

Keeps resumable partials and verifies HF Git-blob/LFS identities before publishing.
No GPU execution; no changes to E5 or the existing model cache.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import model_info
from pipeline.labse_stage03 import ROOT, SMALL_ASSETS, WEIGHT_ASSETS, load_contract


def digests(path):
    sha256, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if not args.run_id or not all(c.isalnum() or c in "_-" for c in args.run_id):
        raise ValueError("Invalid run ID")
    contract = load_contract()
    root = ROOT / "results/labse_stage03"
    output = root / "checkpoint" / contract["revision"]
    receipt_path = root / args.run_id / "download_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=False)
    info = model_info(contract["model_id"], revision=contract["revision"], files_metadata=True)
    if info.sha != contract["revision"]:
        raise RuntimeError("REVISION_MISMATCH")
    siblings = {s.rfilename: s for s in info.siblings}
    receipt = {"model_id": contract["model_id"], "revision": info.sha, "files": [],
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        for name in SMALL_ASSETS + WEIGHT_ASSETS:
            expected = siblings[name]
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                partial = path.with_name(path.name + ".part")
                url = f"https://huggingface.co/{contract['model_id']}/resolve/{info.sha}/{name}"
                print("Downloading", name, "expected_bytes", expected.size, flush=True)
                subprocess.run(["curl.exe", "--http1.1", "-4", "--location", "--fail",
                                "--silent", "--show-error", "--connect-timeout", "20",
                                "--retry", "2", "--retry-delay", "2", "--continue-at", "-",
                                "--output", str(partial), url], check=True)
                checked = partial
            else:
                checked = path
            sha256, blob = digests(checked)
            if checked.stat().st_size != expected.size:
                raise RuntimeError(f"SIZE_MISMATCH:{name}")
            if expected.lfs:
                if sha256 != expected.lfs.sha256:
                    raise RuntimeError(f"LFS_SHA256_MISMATCH:{name}")
            elif blob != expected.blob_id:
                raise RuntimeError(f"GIT_BLOB_MISMATCH:{name}")
            if checked != path:
                checked.rename(path)
            receipt["files"].append({"path": name, "bytes": expected.size,
                                     "sha256": sha256, "hf_git_blob": expected.blob_id,
                                     "hf_lfs_sha256": expected.lfs.sha256 if expected.lfs else None})
            print("Verified", name, flush=True)
        receipt["status"] = "PINNED_ASSETS_VERIFIED"
    except Exception as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
    print("Checkpoint:", output)


if __name__ == "__main__":
    main()
