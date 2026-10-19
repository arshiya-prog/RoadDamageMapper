"""Convert RDD2022 (Pascal VOC XML) into YOLO format for training/train.py --data rdd2022.

1. Download RDD2022 from https://github.com/sekilab/RoadDamageDetector (figshare link in their README)
   — the India subset alone is a good start for Pune roads.
2. Extract it, then:
     python training/prepare_rdd2022.py --src ~/Downloads/RDD2022 --countries India Japan Czech

Class mapping (RDD codes): D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole.
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASS_IDS = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def convert(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path).getroot()
    w = float(tree.findtext("size/width"))
    h = float(tree.findtext("size/height"))
    lines = []
    for obj in tree.iter("object"):
        cid = CLASS_IDS.get(obj.findtext("name", "").strip())
        if cid is None:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="extracted RDD2022 folder")
    ap.add_argument("--countries", nargs="+", default=["India"])
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--keep-empty", type=float, default=0.15, help="fraction of damage-free images to keep as negatives")
    args = ap.parse_args()

    out = ROOT / "datasets" / "rdd2022-yolo"
    shutil.rmtree(out, ignore_errors=True)
    rng = random.Random(0)
    counts = {"train": 0, "val": 0}
    for country in args.countries:
        xml_dir = args.src / country / "train" / "annotations" / "xmls"
        img_dir = args.src / country / "train" / "images"
        if not xml_dir.exists():
            raise SystemExit(f"Not found: {xml_dir}")
        for xml_path in sorted(xml_dir.glob("*.xml")):
            img = img_dir / f"{xml_path.stem}.jpg"
            if not img.exists():
                continue
            lines = convert(xml_path)
            if not lines and rng.random() > args.keep_empty:
                continue
            split = "val" if rng.random() < args.val else "train"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{xml_path.stem}.txt").write_text("\n".join(lines))
            counts[split] += 1
    print(f"Wrote {counts['train']} train / {counts['val']} val images to {out}")


if __name__ == "__main__":
    main()
