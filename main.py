import os
import random
import torch, cv2
from ultralytics import YOLO
from collections import defaultdict
import yaml
from pathlib import Path
import albumentations as A


YOLO_AUG = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.4),
        A.MotionBlur(blur_limit=5, p=0.15),
        A.GaussNoise(p=0.15),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                 rotate=(-6, 6), shear=(-4, 4), p=0.7),
        A.CLAHE(clip_limit=2.0, p=0.15),
    ],
    bbox_params=A.BboxParams(
        format="yolo",              # (xc, yc, w, h) normalized to [0,1]
        label_fields=["class_labels"],
        min_visibility=0.3,         # drop boxes that get too small/hidden
    ),
)


# 2) Utilities: read/write YOLO labels
def read_yolo_labels(label_path: Path):
    boxes, labels = [], []
    if not label_path.exists():
        return boxes, labels
    with open(label_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 5:
                continue
            cid, xc, yc, w, h = parts
            labels.append(int(float(cid)))
            boxes.append([float(xc), float(yc), float(w), float(h)])
    return boxes, labels

def write_yolo_labels(label_path: Path, boxes, labels):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for (xc, yc, w, h), cid in zip(boxes, labels):
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# 3) Map a stem -> actual image path by extension
_POSSIBLE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def stem_to_image_path(stem: str, img_dir: Path) -> Path | None:
    for ext in _POSSIBLE_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

# 4) Scan train labels into: per-class stems, and stem->class set
def scan_train_split(train_img_dir: Path, train_label_dir: Path, nc: int):
    files_by_cid = [set() for _ in range(nc)]  # stems per class
    stem_to_classes = defaultdict(set)

    for lbl in train_label_dir.rglob("*.txt"):
        stem = lbl.stem
        boxes, labels = read_yolo_labels(lbl)
        if not labels:
            continue
        # add image only once per class
        for cid in set(labels):
            if 0 <= cid < nc:
                files_by_cid[cid].add(stem)
                stem_to_classes[stem].add(cid)

    files_by_cid = [sorted(s) for s in files_by_cid]
    return files_by_cid, stem_to_classes

# 5) Augmentation driver
def augment_yolo_dataset(
    dataset_root: str | Path,
    nc: int,
    target_per_class: int = 1000,
    max_aug_per_source: int = 3,
    seed: int = 41,
    keep_all_boxes_in_aug: bool = True,   # if False, keep only target class boxes
    out_subdir: str | None = None,        # None -> write back into train; else "train_aug"
):
    """
    Oversample minority classes in the YOLO train split using Albumentations.

    Args:
        dataset_root: dataset root that contains images/{train,val} and labels/{train,val}
        nc: number of classes
        target_per_class: minimum # of train images that contain each class after augmentation
        max_aug_per_source: cap augmentations from the same original stem (avoid overfitting one image)
        seed: RNG seed
        keep_all_boxes_in_aug: if True, save all boxes found in augmented image
                               if False, save only boxes of the target class (strict class-specific oversampling)
        out_subdir: if None, write to images/train & labels/train.
                    If set (e.g., "train_aug"), write to images/train_aug & labels/train_aug.
    """
    rng = random.Random(seed)
    root = Path(dataset_root)
    train_img_dir = root / "images" / "train"
    train_label_dir = root / "labels" / "train"

    if out_subdir:
        out_img_dir = root / "images" / out_subdir
        out_lbl_dir = root / "labels" / out_subdir
    else:
        out_img_dir = train_img_dir
        out_lbl_dir = train_label_dir

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Scan current train split
    files_by_cid, stem_to_classes = scan_train_split(train_img_dir, train_label_dir, nc)
    per_class_counts = [len(v) for v in files_by_cid]
    print(f"[info] current per-class image counts: {per_class_counts}")

    total_added = 0

    for cid in range(nc):
        present = per_class_counts[cid]
        need = max(0, target_per_class - present)
        if need == 0:
            print(f"[info] class {cid}: already has {present} >= {target_per_class}, skip")
            continue

        candidates = files_by_cid[cid][:]
        if not candidates:
            print(f"[warn] class {cid}: no source images; cannot augment")
            continue

        rng.shuffle(candidates)
        used_times: dict[str, int] = {}
        augmented_for_cid = 0
        i = 0

        print(f"[info] augmenting class {cid}: need {need} more")

        # Incremental unique naming across class to avoid clashes
        uid_counter = 0

        while augmented_for_cid < need and candidates:
            stem = candidates[i % len(candidates)]
            used_times[stem] = used_times.get(stem, 0)

            if used_times[stem] >= max_aug_per_source:
                i += 1
                # simple escape hatch to avoid infinite loops if all capped
                if i > len(candidates) * (max_aug_per_source + 2):
                    break
                continue

            img_path = stem_to_image_path(stem, train_img_dir)
            if img_path is None:
                i += 1
                continue

            lbl_path = train_label_dir / f"{stem}.txt"
            boxes, labels = read_yolo_labels(lbl_path)
            if not boxes:
                i += 1
                continue

            # Load image (BGR)
            img = cv2.imread(str(img_path))
            if img is None:
                i += 1
                continue

            # Albumentations expects bboxes + label_fields
            class_labels = labels[:]
            aug = YOLO_AUG(image=img, bboxes=boxes, class_labels=class_labels)
            aug_img, aug_boxes, aug_labels = aug["image"], aug["bboxes"], aug["class_labels"]

            if not aug_boxes:
                # nothing survived min_visibility/transform—skip
                i += 1
                continue

            if not keep_all_boxes_in_aug:
                # keep only target class boxes
                keep = [(b, l) for (b, l) in zip(aug_boxes, aug_labels) if l == cid]
                if not keep:
                    i += 1
                    continue
                aug_boxes, aug_labels = list(zip(*keep))
                aug_boxes, aug_labels = list(aug_boxes), list(aug_labels)

            # Save with unique name
            new_name = f"{stem}_aug_c{cid}_{uid_counter}"
            uid_counter += 1

            out_img_path = out_img_dir / f"{new_name}.jpg"
            out_lbl_path = out_lbl_dir / f"{new_name}.txt"

            cv2.imwrite(str(out_img_path), aug_img)
            write_yolo_labels(out_lbl_path, aug_boxes, aug_labels)

            augmented_for_cid += 1
            total_added += 1
            used_times[stem] += 1
            i += 1

        print(f"[info] class {cid}: added {augmented_for_cid} augmented images")

    print(f"[done] augmentation complete. Total new images: {total_added}")
    # Optional: re-scan to show new counts (if writing back into train)
    if out_subdir is None:
        new_files_by_cid, _ = scan_train_split(train_img_dir, train_label_dir, nc)
        print("[info] new per-class image counts:",
              [len(v) for v in new_files_by_cid])
    else:
        # If you wrote to train_aug, show counts for that folder only
        aug_files_by_cid, _ = scan_train_split(out_img_dir, out_lbl_dir, nc)
        print("[info] per-class counts in", out_subdir, ":",
              [len(v) for v in aug_files_by_cid])
        

def generate_data_yaml(label_path: str, output_path: Path | None = None):
    if os.path.exists("data.yaml"):
        print("[info] data.yaml already exists. Skipping generation")
        return

    ids = set()
    root_path = Path(label_path)
    train_label_dir = root_path / "labels" / "train"
    train_img_dir = (root_path / "images" / "train").as_posix()
    val_img_dir = (root_path / "images" / "val").as_posix()
    val_label_dir = root_path / "labels" / "val"

    file_to_class = []

    for f in train_label_dir.rglob("*.txt"):
        try:
            with open(f, "r") as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    cid = int(float(s.split()[0]))
                    ids.add(cid)

        except Exception as e:
            print(f"[Warning] Skipping {f}: {e}")
    
    if not ids:
        print("[info] No class ids found in labels. Assuming 1 class")
        ids = {0}

    max_id = max(ids)
    nc = max_id + 1
    names = [f"class_{i}" for i in range(nc)]


    for f in val_label_dir.rglob("*.txt"):
        try:
            with open(f, "r") as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    cid = int(float(s.split()[0]))
                    
                    file_to_class.append({
                        "file_name": f,
                        "cid": cid
                    })

        except Exception as e:
            print(f"[Warning] Skipping {f}: {e}")

    files_by_cid_sets = [set() for _ in range(nc)]

    for rec in file_to_class:
        cid = rec["cid"]
        if 0 <= cid < nc:
            img_path = Path(rec["file_name"])
            files_by_cid_sets[cid].add(img_path)

    files_by_cid = [sorted(s) for s in files_by_cid_sets]

    print([len(files_by_cid[cid]) for cid in range(nc)])

    data = {
        "train": train_img_dir,
        "val": val_img_dir,
        "nc": nc,
        "names": names
    }

    with open("data.yaml", 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"[info] data.yaml file created. nc = {nc}, ids = {sorted(ids)}")


def train_yolo(model_path: str, imgsz: int, epochs: int):
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f'[info] device: {device}')

    workers = max(2, min(8, os.cpu_count() or 4))
    print(f'[info] # of workers" {workers}')

    model = YOLO(model_path)
    model.train(data='data.yaml', epochs=epochs, imgsz=imgsz, device=device, workers=workers, seed=41, cos_lr = True)

    metrics = model.val(data='data.yaml', imgsz=imgsz, plots=True)
    print(f"[final] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


def finetune_yolo(model_path: str, imgsz: int, epochs: int):
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f'[info] device: {device}')

    workers = max(2, min(8, os.cpu_count() or 4))
    print(f'[info] # of workers" {workers}')

    model = YOLO("yolo11n.pt")
    model.load(str(model_path))

    model.train(data='sign.yaml', epochs=5, imgsz=imgsz, device=device, workers=workers, seed=41, cos_lr = True, resume=False, freeze=10, lr0=5e-4, lrf=5e-4)

    model = YOLO("yolo11n.pt")
    model.load(str("runs/detect/train2/weights/best.pt"))

    model.train(model="yolo11n.pt", data='sign.yaml', epochs=epochs, imgsz=imgsz, device=device, workers=workers, seed=41, cos_lr = True, resume=False, freeze=0, lr0=1e-3, lrf=1e-3)
    metrics = model.val(data='sign.yaml', imgsz=imgsz, plots=True)

    print(f"[final] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


def pred(model_path: str, input_path: str, conf: float):
    """
    Run inference with retries if no detections:
      1) First pass: imgsz=640 (or whatever you trained for), conf as given
      2) Retry if 0 boxes: lower conf a bit, bump imgsz, enable TTA
    Saves annotated outputs to runs/detect/predict*
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    def _run(imgsz: int, conf_thres: float, desc: str):
        res = model.predict(
            source=input_path,
            imgsz=imgsz,
            conf=conf_thres,
            iou=0.7,              # slightly higher NMS IoU can help overlapping small signs
            device=device,
            agnostic_nms=False,   # set True if you want class-agnostic NMS
            augment=False,        # set True only for TTA retry
            save=True,            # save annotated image/video
            verbose=False,
        )
        r = res[0]
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"[{desc}] imgsz={imgsz} conf={conf_thres:.2f} -> boxes={n}")
        return r, res

    # Pass 1: your settings
    r, res = _run(imgsz=640, conf_thres=conf, desc="pass1")

    # If nothing found, retry with lower conf, bigger imgsz, and TTA
    if r.boxes is None or len(r.boxes) == 0:
        conf_retry = max(conf * 0.75, 0.20)  # don’t go too low
        r, res = model.predict(
            source=input_path,
            imgsz=960,            # more pixels for tiny signs
            conf=conf_retry,
            iou=0.7,
            device=device,
            agnostic_nms=True,    # allow overlapping classes to survive NMS
            augment=True,         # simple test-time augmentation
            save=True,
            verbose=False,
        ), None
        # model.predict returns a list; align with earlier shape:
        r = r[0]
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"[retry] imgsz=960 conf={conf_retry:.2f} TTA=True agnostic_nms=True -> boxes={n}")

    # Print a few detections for quick sanity
    if r.boxes is not None and len(r.boxes):
        names = r.names
        for b in r.boxes[:10]:
            x1, y1, x2, y2 = b.xyxy[0].int().tolist()
            cls_id = int(b.cls[0])
            c = float(b.conf[0])
            print(f"  {names[cls_id]}  conf={c:.2f}  box=({x1},{y1},{x2},{y2})")
    else:
        print("[hint] Still no detections. Double-check you're loading your trained best.pt, "
              "try imgsz=1280, or lower conf to ~0.25. Tiny/occluded signs often need higher imgsz.")

    return r  # return the first (or retried) result object


if __name__ == '__main__':
    generate_data_yaml('Traffic_sign_detection_data')
    augment_yolo_dataset(
        dataset_root='Traffic_sign_detection_data',
        nc=43,
        target_per_class=200,
        max_aug_per_source=20,
        seed=41,
        keep_all_boxes_in_aug=True,   # set False to keep only the target class boxes
        out_subdir=None               # or 'train_aug'
    )
    train_yolo("yolo11n.pt", 640, 20)
    finetune_yolo("runs/detect/train/weights/best.pt", 416, 20)
    pred("runs/detect/train3/weights/best.pt", "sample.png", 0.25)