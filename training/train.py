"""Train the road-damage detector and install it at weights/road_damage.pt.

Datasets (--data)
  rdd2022         Road Damage Dataset 2022 (~10k Indian road images; potholes + 3 crack types).
                  Run training/prepare_rdd2022.py first.
  path/to/x.yaml  Any YOLO-format dataset, e.g. a pothole set exported from Roboflow Universe or
                  Kaggle. Class names are mapped through roadscan.severity.LABEL_ALIASES
                  ("pothole", "D40", "alligator crack", ...).

Examples
  python training/train.py --data rdd2022 --epochs 80 --model yolo11s.pt
  python training/train.py --data ~/Downloads/pothole-dataset/data.yaml --epochs 40
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


def dataset_yaml(name: str) -> Path:
    if name.endswith((".yaml", ".yml")):
        return Path(name).expanduser().resolve()
    out = ROOT / "training" / f"_{name}.yaml"
    if name == "rdd2022":
        root = DATASETS / "rdd2022-yolo"
        if not (root / "images" / "train").exists():
            raise SystemExit("Run `python training/prepare_rdd2022.py` first.")
        cfg = {"path": str(root), "train": "images/train", "val": "images/val",
               "names": {0: "longitudinal_crack", 1: "transverse_crack", 2: "alligator_crack", 3: "pothole"}}
    else:
        raise SystemExit(f"Unknown dataset {name}")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rdd2022", help="'rdd2022' or a path to a YOLO data.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="starting checkpoint; yolo11s.pt is what weights/road_damage.pt was trained from (n=faster, m=more accurate)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="mps, cuda:0, cpu (auto if omitted)")
    args = ap.parse_args()

    from ultralytics import YOLO

    DATASETS.mkdir(exist_ok=True)
    (ROOT / "weights").mkdir(exist_ok=True)
    model = YOLO(str(ROOT / "weights" / args.model) if (ROOT / "weights" / args.model).exists() else args.model)
    results = model.train(
        data=str(dataset_yaml(args.data)), task="detect", epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(ROOT / "runs"), name=Path(args.data).stem, exist_ok=True,
        # Dashcam footage: lots of brightness/blur variation, rarely flipped vertically.
        hsv_v=0.5, degrees=0, flipud=0.0, fliplr=0.5, mosaic=1.0, patience=15,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = ROOT / "weights" / "road_damage.pt"
    shutil.copy(best, dest)
    print(f"\nInstalled {best} → {dest}")


if __name__ == "__main__":
    main()
