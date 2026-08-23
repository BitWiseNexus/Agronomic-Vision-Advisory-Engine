r"""Fetch the PlantVillage dataset into data/raw/plantvillage/ (Phase 1).

    .\.venv\Scripts\python.exe scripts\download_plantvillage.py --source github

The images are ~1.6 GB and are not redistributable from here, so this script only drives
the upstream sources. All options are free.

  github  The canonical repository from the original paper:
          https://github.com/spMohanty/PlantVillage-Dataset
          Recommended — no account, no API token. Needs git 2.25+ for sparse-checkout.
  kaggle  A mirror. Requires a free Kaggle account and ~/.kaggle/kaggle.json.
          Dataset: abdallahalidev/plantvillage-dataset
  manual  Prints instructions and verifies whatever you extracted yourself.

Why the github route uses a sparse, blobless, shallow clone rather than `git clone`:
the repository carries three copies of every image — raw/color, raw/grayscale and
raw/segmented — so a plain clone pulls roughly 3x the data plus the full history. Fetching
only `raw/<variant>` at depth 1 with `--filter=blob:none` downloads just the images of the
variant asked for.

Use `color` (the default). `segmented` has the background already removed, which would
short-circuit the leaf-segmentation step our severity index depends on, and make the
measured severity look far better than it would on a real phone photo.

After downloading, the tree must look like an ImageFolder:

    data/raw/plantvillage/Tomato___Late_blight/*.JPG
    data/raw/plantvillage/Tomato___healthy/*.JPG

Note on the paper's own split: `data_distribution_for_SVM/{train,test}_mapping.txt` in that
repository defines the split used for the paper's SVM baseline. `ml/train.py` uses its own
seeded random split instead. Reach for the upstream mapping only if you specifically want
numbers comparable to the paper — and note it covers a subset of the 54,306 images, with a
test half larger than the train half, so it is not a drop-in 80/20.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "plantvillage"
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
GITHUB_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
EXPECTED_CLASSES = 38
EXPECTED_IMAGES = 54_306  # as stated by the upstream README

MANUAL_INSTRUCTIONS = f"""
Manual download
---------------
1. Open https://www.kaggle.com/datasets/{KAGGLE_DATASET}
2. Download the archive (free account required) and extract it.
3. Inside the extract, find the colour image folder — usually:
       plantvillage dataset/color/
   It contains 38 subfolders named like 'Tomato___Late_blight'.
4. Copy those 38 subfolders into:
       {TARGET}
5. Re-run this script with --verify to confirm the layout.
"""


def verify() -> bool:
    if not TARGET.exists():
        print(f"Not found: {TARGET}")
        return False

    class_dirs = sorted(path for path in TARGET.iterdir() if path.is_dir())
    if not class_dirs:
        print(f"{TARGET} exists but contains no class subfolders.")
        return False

    total = 0
    thin: list[str] = []
    for directory in class_dirs:
        count = sum(1 for _ in directory.glob("*.[jJpP]*[gG]"))
        total += count
        if count < 20:
            thin.append(f"{directory.name} ({count})")

    print(f"{len(class_dirs)} classes, {total:,} images in {TARGET}")
    if len(class_dirs) != EXPECTED_CLASSES:
        print(f"  note: expected {EXPECTED_CLASSES} classes for the full colour set, "
              f"found {len(class_dirs)}")
    if total and abs(total - EXPECTED_IMAGES) > 0.02 * EXPECTED_IMAGES:
        print(f"  note: upstream states {EXPECTED_IMAGES:,} images; this tree has {total:,}")
    if thin:
        print(f"  note: very small classes: {', '.join(thin[:8])}")

    # A label-order mismatch between training and serving makes every prediction
    # confidently wrong, so check the folder names against the taxonomy we ship.
    sys.path.insert(0, str(ROOT / "backend"))
    try:
        from app.vision.labels import PLANTVILLAGE_CLASSES  # noqa: PLC0415
    except Exception:
        return total > 0

    found = [directory.name for directory in class_dirs]
    unexpected = sorted(set(found) - set(PLANTVILLAGE_CLASSES))
    missing = sorted(set(PLANTVILLAGE_CLASSES) - set(found))
    if unexpected or missing:
        print("  warning: folder names differ from app/vision/labels.py")
        for name in unexpected[:5]:
            print(f"    not in labels.py : {name}")
        for name in missing[:5]:
            print(f"    expected, absent : {name}")
        print("    Training still works — the class list is read from the folders and saved")
        print("    into the checkpoint — but update labels.py so the untrained-head")
        print("    placeholder path reports sensible names.")
    else:
        print(f"  class names match app/vision/labels.py ({len(found)}/{EXPECTED_CLASSES})")
    return total > 0


def download_github(variant: str = "color", keep_staging: bool = False) -> None:
    """Shallow, blobless, sparse clone of just `raw/<variant>`."""
    if shutil.which("git") is None:
        print("git was not found on PATH.")
        sys.exit(1)

    staging = ROOT / "data" / "raw" / "_plantvillage_repo"
    subdir = f"raw/{variant}"

    if not (staging / ".git").exists():
        staging.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {GITHUB_REPO}\n  sparse: {subdir}  (~1.6 GB for color)")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             GITHUB_REPO, str(staging)],
            check=True,
        )
    else:
        print(f"Reusing the existing clone at {staging}")

    # sparse-checkout in cone mode: only this directory's blobs are fetched.
    subprocess.run(["git", "-C", str(staging), "sparse-checkout", "set", subdir], check=True)

    source = staging / "raw" / variant
    if not source.is_dir():
        print(f"Expected {source} after checkout, but it is not there.")
        print("Check that the variant name is one of: color, grayscale, segmented.")
        sys.exit(1)

    TARGET.mkdir(parents=True, exist_ok=True)
    moved = 0
    skipped = 0
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        destination = TARGET / class_dir.name
        if destination.exists():
            skipped += 1
            continue
        shutil.move(str(class_dir), str(destination))
        moved += 1

    print(f"Moved {moved} class folders into {TARGET}" + (f" ({skipped} already present)" if skipped else ""))

    if keep_staging:
        print(f"Staging clone kept at {staging}")
    else:
        # The images were moved out, so only git metadata remains.
        shutil.rmtree(staging, ignore_errors=True)
        print(f"Removed the staging clone ({staging.name})")


def download_kaggle() -> None:
    if shutil.which("kaggle") is None:
        print("The `kaggle` CLI is not installed in this venv. Install it with:")
        print("    .\\.venv\\Scripts\\python.exe -m pip install kaggle")
        print("Then place your API token at %USERPROFILE%\\.kaggle\\kaggle.json")
        sys.exit(1)

    staging = ROOT / "data" / "raw" / "_plantvillage_zip"
    staging.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} (~1.6 GB) into {staging} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(staging), "--unzip"],
        check=True,
    )

    # The archive nests the classes under a 'color' directory; find it wherever it landed.
    colour_dirs = [path for path in staging.rglob("color") if path.is_dir()]
    if not colour_dirs:
        print(f"Downloaded, but no 'color' directory was found under {staging}.")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    source = colour_dirs[0]
    TARGET.mkdir(parents=True, exist_ok=True)
    moved = 0
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        destination = TARGET / class_dir.name
        if destination.exists():
            continue
        shutil.move(str(class_dir), str(destination))
        moved += 1
    print(f"Moved {moved} class folders into {TARGET}")
    print(f"You can now delete the staging directory: {staging}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch PlantVillage into data/raw/plantvillage/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Recommended:  python scripts/download_plantvillage.py --source github",
    )
    parser.add_argument("--source", choices=["github", "kaggle", "manual"], default="github")
    parser.add_argument(
        "--variant",
        choices=["color", "grayscale", "segmented"],
        default="color",
        help="github only. Keep 'color': 'segmented' has the background already removed, "
        "which would bypass the leaf segmentation the severity index depends on.",
    )
    parser.add_argument(
        "--keep-staging", action="store_true", help="github only: keep the staging clone"
    )
    parser.add_argument("--verify", action="store_true", help="Only check the existing layout")
    args = parser.parse_args()

    if args.verify:
        sys.exit(0 if verify() else 1)

    if args.source == "github":
        download_github(args.variant, args.keep_staging)
    elif args.source == "kaggle":
        download_kaggle()
    else:
        print(MANUAL_INSTRUCTIONS)

    verify()


if __name__ == "__main__":
    main()
