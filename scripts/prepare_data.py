"""Extract the Kaggle ZIP into data/raw/ and verify structure. Idempotent.

Usage: python scripts/prepare_data.py --zip ~/Downloads/archive.zip
"""
import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.config import CLASS_NAMES, DATA_DIR


def count_images(directory):
    return len([f for f in os.listdir(directory)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def verify(data_dir):
    ok = True
    for split in ("Training", "Testing"):
        for cls in CLASS_NAMES:
            d = os.path.join(data_dir, split, cls)
            if not os.path.isdir(d):
                print(f"MISSING: {d}")
                ok = False
                continue
            print(f"{split}/{cls}: {count_images(d)} images")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to Kaggle archive.zip")
    args = ap.parse_args()

    if not os.path.isfile(args.zip):
        sys.exit(f"ERROR: ZIP not found: {args.zip}")

    os.makedirs(DATA_DIR, exist_ok=True)

    corrupted = 0
    with zipfile.ZipFile(args.zip) as z:
        names = z.namelist()
        if not any(("Training" in n.split("/") and "Testing" in n.split("/"))
                   or n.split("/")[-2] in ("Training", "Testing")
                   for n in names):
            sys.exit("ERROR: ZIP does not contain the expected Training/Testing layout")
        for info in z.infolist():
            if info.is_dir():
                continue
            # flatten a single leading top-level folder if present
            parts = info.filename.split("/", 1)
            rel = parts[1] if len(parts) == 2 and parts[0] not in ("Training", "Testing") else info.filename
            target = os.path.join(DATA_DIR, rel)
            if os.path.isfile(target):
                continue  # idempotent: already extracted
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                with z.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
            except Exception:
                corrupted += 1

    if corrupted:
        print(f"WARNING: {corrupted} files failed to extract")

    if not verify(DATA_DIR):
        sys.exit("ERROR: extracted data failed verification")
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
