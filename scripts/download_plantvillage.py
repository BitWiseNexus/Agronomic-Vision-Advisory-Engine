r"""Fetch the PlantVillage dataset into data/raw/plantvillage/ (Phase 1).

    .\.venv\Scripts\python.exe scripts\download_plantvillage.py --source kaggle

The dataset is ~1.6 GB and is not redistributable from here, so this script only drives
the official sources. Both options are free.

  kaggle  Requires a free Kaggle account and ~/.kaggle/kaggle.json API token.
          Dataset: abdallahalidev/plantvillage-dataset
  manual  Prints instructions and verifies whatever you extracted yourself.

After downloading, the tree must look like an ImageFolder:

    data/raw/plantvillage/Tomato___Late_blight/*.jpg
    data/raw/plantvillage/Tomato___healthy/*.jpg
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

    print(f"{len(class_dirs)} classes, {total} images in {TARGET}")
    if len(class_dirs) != 38:
        print(f"  note: expected 38 classes for the full PlantVillage colour set, found {len(class_dirs)}")
    if thin:
        print(f"  note: very small classes: {', '.join(thin[:8])}")
    return total > 0


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
    parser = argparse.ArgumentParser(description="Fetch PlantVillage")
    parser.add_argument("--source", choices=["kaggle", "manual"], default="manual")
    parser.add_argument("--verify", action="store_true", help="Only check the existing layout")
    args = parser.parse_args()

    if args.verify:
        sys.exit(0 if verify() else 1)

    if args.source == "kaggle":
        download_kaggle()
    else:
        print(MANUAL_INSTRUCTIONS)

    verify()


if __name__ == "__main__":
    main()
