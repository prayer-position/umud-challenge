"""Build a self-contained Kaggle script kernel and its push request.

    python kaggle/build_kernel.py --owner <kaggle-username> --run full
    python kaggle/build_kernel.py --owner <kaggle-username> --run fasc_experiments \
        --slug umud-fasc-exp1 --kernel-source <kaggle-username>/umud-unet-train

Writes kaggle/build/umud_train.py (the kernel source, with the committed repo
embedded as a tarball; it runs kaggle/runs/<run>.sh) and
kaggle/build/push_request.json, a body for
POST https://www.kaggle.com/api/v1/kernels/push.
"""
import argparse
import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"
REPO_PATHS = ["src", "scripts", "configs", "kaggle/runs", "pyproject.toml", "sample_submission.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Kaggle username that owns the kernel.")
    parser.add_argument("--slug", default="umud-unet-train")
    parser.add_argument("--title", default="UMUD UNet Train")
    parser.add_argument("--machine-shape", default="NvidiaTeslaT4")
    parser.add_argument("--run", default="full", help="Script in kaggle/runs/ to execute (without .sh).")
    parser.add_argument("--run-args", nargs="*", default=[], help="Arguments passed to the run script.")
    parser.add_argument(
        "--kernel-source",
        action="append",
        default=[],
        help="owner/slug of a kernel whose outputs to mount under /kaggle/input (repeatable).",
    )
    args = parser.parse_args()

    if not (ROOT / "kaggle" / "runs" / f"{args.run}.sh").exists():
        parser.error(f"no such run script: kaggle/runs/{args.run}.sh")
    tgz = subprocess.run(
        ["git", "archive", "--format=tar.gz", "HEAD", *REPO_PATHS], cwd=ROOT, check=True, capture_output=True
    ).stdout
    source = (
        (ROOT / "kaggle" / "kernel_template.py")
        .read_text()
        .replace("__REPO_TGZ_B64__", base64.b64encode(tgz).decode())
        .replace("__RUN_SCRIPT__", f"kaggle/runs/{args.run}.sh")
        .replace("__EXTRA_ARGS__", repr(args.run_args))
    )

    out_dir = ROOT / "kaggle" / "build"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "umud_train.py").write_text(source)
    request = {
        "slug": f"{args.owner}/{args.slug}",
        "newTitle": args.title,
        "text": source,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "machineShape": args.machine_shape,
        "competitionDataSources": [COMPETITION],
        "datasetDataSources": [],
        "kernelDataSources": args.kernel_source,
        "modelDataSources": [],
        "categoryIds": [],
    }
    (out_dir / "push_request.json").write_text(json.dumps(request))
    print(f"Wrote {out_dir / 'umud_train.py'} ({len(source) // 1024} KiB) and push_request.json")


if __name__ == "__main__":
    main()
