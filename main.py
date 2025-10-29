import os
import random, time, json
import numpy as np
import torch, cv2
from ultralytics import YOLO
from collections import defaultdict
import yaml
from pathlib import Path
import albumentations as A
from sklearn.metrics import (precision_recall_fscore_support, average_precision_score, roc_auc_score)


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


def model_metrics(model_path: str,
                  imgsz: int,
                  data_yaml: str = "sign.yaml",   # ← 원하는 YAML 경로로 바꿔 호출
                  iou_thr: float = 0.50,
                  ece_bins: int = 15,
                  stress_max_imgs: int = 150,
                  conf_infer: float = 0.001,      # low conf to collect *all* candidates
                  conf_eval: float | None = None, # set e.g. 0.45 to match YOLO plot
                  do_sweep: bool = True) -> dict:
    """
    Evaluate on the *val/valid* split defined in `data_yaml` (e.g., sign.yaml).
    Computes:
      - Top-1/Top-3 image-level class presence accuracy
      - Precision/Recall/F1 (micro & macro, fixed IoU threshold)
      - PR-AUC (per-class & macro)
      - ROC-AUC (approx; non-standard for detection)
      - Calibration: ECE, Brier score
      - Efficiency: latency, throughput, model size
      - Stress test: subset mAP50 under corruptions (noise/blur/brightness-contrast)
    Returns: JSON-serializable dict
    """
    import os, time, yaml, cv2
    import numpy as np
    from pathlib import Path
    from collections import defaultdict
    from ultralytics import YOLO
    from sklearn.metrics import average_precision_score, roc_auc_score

    # ---------- small helpers ----------
    def load_yaml(p):
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def list_images(img_dir: Path):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])

    def yolo_to_xyxy(xc, yc, w, h, W, H):
        return [(xc - w/2) * W, (yc - h/2) * H, (xc + w/2) * W, (yc + h/2) * H]

    def read_gt_label(lbl_path: Path, W: int, H: int):
        boxes, classes = [], []
        if not lbl_path.exists():
            return np.zeros((0,4), np.float32), np.zeros((0,), np.int32)
        with open(lbl_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                cid, xc, yc, w, h = s.split()
                boxes.append(yolo_to_xyxy(float(xc), float(yc), float(w), float(h), W, H))
                classes.append(int(float(cid)))
        return np.array(boxes, np.float32), np.array(classes, np.int32)

    def iou_matrix(a, b):
        if a.size == 0 or b.size == 0:
            return np.zeros((len(a), len(b)), np.float32)
        ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
        bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
        inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
        inter_y1 = np.maximum(ay1[:,None], by1[None,:])
        inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
        inter_y2 = np.minimum(ay2[:,None], by2[None,:])
        inter_w  = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h  = np.maximum(0.0, inter_y2 - inter_y1)
        inter    = inter_w * inter_h
        area_a   = (ax2 - ax1) * (ay2 - ay1)
        area_b   = (bx2 - bx1) * (by2 - by1)
        union    = area_a[:,None] + area_b[None,:] - inter
        return np.where(union > 0, inter / union, 0.0)

    def greedy_match(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, thr):
        P, G = len(p_xyxy), len(g_xyxy)
        if P == 0:
            return np.zeros(0, bool), np.zeros(G, bool)
        if G == 0:
            return np.zeros(P, bool), np.zeros(0, bool)
        order = np.argsort(-p_conf)
        p_xyxy = p_xyxy[order]; p_cls = p_cls[order]
        tp = np.zeros(P, bool); gt_taken = np.zeros(G, bool)
        for i in range(P):
            c = p_cls[i]
            m = (g_cls == c)
            if not np.any(m): continue
            ious = iou_matrix(p_xyxy[i:i+1], g_xyxy[m])[0]
            cand_idx = np.where(m)[0]
            free = ~gt_taken[cand_idx]
            if not np.any(free): continue
            ious = ious[free]; cand_idx = cand_idx[free]
            if ious.size == 0: continue
            j = np.argmax(ious)
            if ious[j] >= thr:
                tp[i] = True
                gt_taken[cand_idx[j]] = True
        inv = np.empty_like(order); inv[order] = np.arange(P)
        return tp[inv], gt_taken

    def brier_score(conf, tp_flags):
        if len(conf) == 0: return None
        y = tp_flags.astype(np.float32)
        return float(np.mean((conf - y)**2))

    def expected_calibration_error(conf, tp_flags, n_bins=15):
        if len(conf) == 0: return None
        conf = np.asarray(conf, np.float32); y = tp_flags.astype(np.float32)
        bins = np.linspace(0,1,n_bins+1); N = len(conf); ece = 0.0
        for i in range(n_bins):
            m = (conf >= bins[i]) & (conf < (bins[i+1] if i < n_bins-1 else bins[i+1]))
            if not np.any(m): continue
            acc = np.mean(y[m]); cbar = np.mean(conf[m]); ece += (np.sum(m)/N)*abs(acc-cbar)
        return float(ece)
    
    def prf_at_threshold(per_image_preds, per_image_gts, thr: float, iou_thr: float, nc: int):
        # micro accumulators
        TP=FP=FN=0
        # macro accumulators per class
        tp_c = np.zeros(nc, dtype=int)
        fp_c = np.zeros(nc, dtype=int)
        fn_c = np.zeros(nc, dtype=int)

        for pred, gt in zip(per_image_preds, per_image_gts):
            p_xyxy, p_cls, p_conf = pred["xyxy"], pred["cls"], pred["conf"]
            keep = (p_conf >= thr)
            p_xyxy, p_cls, p_conf = p_xyxy[keep], p_cls[keep], p_conf[keep]

            tp_flags, gt_taken = greedy_match(p_xyxy, p_conf, p_cls, gt["xyxy"], gt["cls"], iou_thr)

            # --- micro ---
            TP += int(tp_flags.sum())
            FP += int((~tp_flags).sum())
            FN += int(max(len(gt["cls"]) - gt_taken.sum(), 0))

            # --- macro per class ---
            for c in range(nc):
                # predictions of class c at this threshold
                m_pred_c = (p_cls == c)
                if m_pred_c.any():
                    tp_c[c] += int(tp_flags[m_pred_c].sum())
                    fp_c[c] += int((~tp_flags[m_pred_c]).sum())
                # ground truths of class c for this image
                m_gt_c = (gt["cls"] == c)
                if m_gt_c.any():
                    # fn = gt of class c not taken
                    fn_c[c] += int((~gt_taken[m_gt_c]).sum())

        # micro P/R/F1
        P_micro = TP/(TP+FP) if TP+FP>0 else 0.0
        R_micro = TP/(TP+FN) if TP+FN>0 else 0.0
        F1_micro = (2*P_micro*R_micro/(P_micro+R_micro)) if (P_micro+R_micro)>0 else 0.0

        # macro P/R/F1 (unweighted mean of per-class)
        P_macro_list, R_macro_list, F1_macro_list = [], [], []
        for c in range(nc):
            Pc = tp_c[c]/(tp_c[c]+fp_c[c]) if (tp_c[c]+fp_c[c])>0 else 0.0
            Rc = tp_c[c]/(tp_c[c]+fn_c[c]) if (tp_c[c]+fn_c[c])>0 else 0.0
            F1c = (2*Pc*Rc/(Pc+Rc)) if (Pc+Rc)>0 else 0.0
            P_macro_list.append(Pc); R_macro_list.append(Rc); F1_macro_list.append(F1c)

        P_macro = float(np.mean(P_macro_list)) if P_macro_list else 0.0
        R_macro = float(np.mean(R_macro_list)) if R_macro_list else 0.0
        F1_macro = float(np.mean(F1_macro_list)) if F1_macro_list else 0.0

        return (P_micro, R_micro, F1_micro, TP, FP, FN,
                P_macro, R_macro, F1_macro)

    def sweep_best_f1(per_image_preds, per_image_gts, iou_thr: float, nc: int):
        thresholds = np.linspace(0.0, 1.0, 201)
        best = {
            "conf": 0.0,
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "TP": 0, "FP": 0, "FN": 0},
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }
        best_f1 = 0.0
        for t in thresholds:
            (Pmi, Rmi, F1mi, TP, FP, FN, Pma, Rma, F1ma) = prf_at_threshold(
                per_image_preds, per_image_gts, t, iou_thr, nc
            )
            if F1mi > best_f1:
                best_f1 = F1mi
                best = {
                    "conf": float(t),
                    "micro": {"precision": float(Pmi), "recall": float(Rmi), "f1": float(F1mi),
                            "TP": int(TP), "FP": int(FP), "FN": int(FN)},
                    "macro": {"precision": float(Pma), "recall": float(Rma), "f1": float(F1ma)}
                }
        return best

    # read YAML (sign.yaml) 
    data_cfg = load_yaml(data_yaml)
    train_img_dir = Path(data_cfg["train"])
    val_img_dir   = Path(data_cfg.get("val") or data_cfg.get("valid") or data_cfg.get("validation"))
    # Roboflow : ".../valid/images" → ".../valid/labels"
    if "images" in str(val_img_dir):
        val_lbl_dir = Path(str(val_img_dir).replace("\\images", "\\labels").replace("/images", "/labels"))
    else:
        # data.yaml: path + images/val ↔ labels/val
        val_lbl_dir = Path(str(val_img_dir).replace("/images/", "/labels/"))
    nc    = int(data_cfg["nc"])
    names = data_cfg.get("names", [f"class_{i}" for i in range(nc)])

    val_imgs = list_images(val_img_dir)
    if len(val_imgs) == 0:
        raise RuntimeError(f"No validation images found under: {val_img_dir}")

    # model 
    device = 0 if torch.cuda.is_available() else "cpu"
    model  = YOLO(model_path)

    per_image_preds = []  # each: {"xyxy": (N,4), "cls": (N,), "conf": (N,)}
    per_image_gts   = []  # each: {"xyxy": (M,4), "cls": (M,)}


    # sweep once over clean val
    per_cls_scores = [[] for _ in range(nc)]
    per_cls_hits   = [[] for _ in range(nc)]
    gt_pos_counts  = np.zeros(nc, int)

    per_img_presence_top = []  # (set(gt classes), {cls: max_conf_in_img})

    lat_ms = []
    TP=FP=FN=0

    for img_path in val_imgs:
        img = cv2.imread(str(img_path))
        if img is None: continue
        H,W = img.shape[:2]
        lbl = val_lbl_dir / img_path.with_suffix(".txt").name
        g_xyxy, g_cls = read_gt_label(lbl, W, H)
        for c in g_cls.tolist():
            if 0 <= c < nc: gt_pos_counts[c] += 1

        t1 = time.time()
        pred = model.predict(source=str(img_path), imgsz=imgsz, conf=conf_infer, iou=0.70,
                             device=device, verbose=False, agnostic_nms=False)[0]
        t2 = time.time()
        lat_ms.append((t2-t1)*1000)

        if pred.boxes is not None and len(pred.boxes):
            p_xyxy = pred.boxes.xyxy.cpu().numpy().astype(np.float32)
            p_cls  = pred.boxes.cls.cpu().numpy().astype(np.int32)
            p_conf = pred.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            p_xyxy = np.zeros((0,4), np.float32)
            p_cls  = np.zeros((0,), np.int32)
            p_conf = np.zeros((0,), np.float32)

        per_image_preds.append({"xyxy": p_xyxy, "cls": p_cls, "conf": p_conf})
        per_image_gts.append({"xyxy": g_xyxy, "cls": g_cls})

        tp_flags, gt_taken = greedy_match(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, iou_thr)

        # detection-level accumulators
        for c in range(nc):
            m = (p_cls == c)
            if np.any(m):
                per_cls_scores[c].extend(p_conf[m].tolist())
                per_cls_hits[c].extend(tp_flags[m].astype(int).tolist())

        TP += int(tp_flags.sum())
        FP += int((~tp_flags).sum())
        FN += int(max(len(g_cls) - gt_taken.sum(), 0))

        # image-level class presence conf map (for top-k)
        confmap = defaultdict(float)
        for c in range(nc):
            if np.any(p_cls == c):
                confmap[c] = float(np.max(p_conf[p_cls==c]))
        per_img_presence_top.append((set(g_cls.tolist()), confmap))

    # efficiency
    avg_latency_ms = float(np.mean(lat_ms)) if lat_ms else None
    throughput_fps = float(1000.0/avg_latency_ms) if avg_latency_ms and avg_latency_ms>0 else None
    try:
        model_size_mb = os.path.getsize(model_path)/(1024*1024)
    except Exception:
        model_size_mb = None

    # micro P/R/F1
    prec_micro = TP/(TP+FP) if TP+FP>0 else 0.0
    rec_micro  = TP/(TP+FN) if TP+FN>0 else 0.0
    f1_micro   = (2*prec_micro*rec_micro/(prec_micro+rec_micro)) if (prec_micro+rec_micro)>0 else 0.0

    # macro P/R/F1 and PR-AUC/ROC-AUC (approx)
    macro_prec_list=[]; macro_rec_list=[]; macro_f1_list=[]
    pr_auc_values=[]; roc_auc_values=[]
    for c in range(nc):
        y_scores = np.array(per_cls_scores[c], np.float32)
        y_true   = np.array(per_cls_hits[c],   np.int32)  # 1=TP / 0=FP
        P        = int(gt_pos_counts[c])

        tp_c = int(y_true.sum())
        fp_c = int(len(y_true)-tp_c)
        fn_c = int(max(P - tp_c, 0))
        prec_c = tp_c/(tp_c+fp_c) if (tp_c+fp_c)>0 else 0.0
        rec_c  = tp_c/(tp_c+fn_c) if (tp_c+fn_c)>0 else 0.0
        f1_c   = (2*prec_c*rec_c/(prec_c+rec_c)) if (prec_c+rec_c)>0 else 0.0
        macro_prec_list.append(prec_c); macro_rec_list.append(rec_c); macro_f1_list.append(f1_c)

        # AP (PR-AUC)
        ap_c = None
        if len(y_scores)>0 and P>0 and len(np.unique(y_true))>1:
            try:
                ap_c = average_precision_score(y_true, y_scores)
            except Exception:
                ap_c = None
        pr_auc_values.append(ap_c)

        # ROC-AUC (approx; non-standard for detection)
        roc_c = None
        if len(np.unique(y_true))>1:
            try:
                roc_c = roc_auc_score(y_true, y_scores)
            except Exception:
                roc_c = None
        roc_auc_values.append(roc_c)

    prec_macro = float(np.mean(macro_prec_list)) if macro_prec_list else 0.0
    rec_macro  = float(np.mean(macro_rec_list))  if macro_rec_list else 0.0
    f1_macro   = float(np.mean(macro_f1_list))   if macro_f1_list else 0.0
    pr_auc_macro = float(np.nanmean([v for v in pr_auc_values if v is not None])) if any(v is not None for v in pr_auc_values) else None
    roc_auc_macro = float(np.nanmean([v for v in roc_auc_values if v is not None])) if any(v is not None for v in roc_auc_values) else None

    # calibration (detections all classes)
    all_scores = np.concatenate([np.array(per_cls_scores[c], np.float32) for c in range(nc)]) if nc>0 else np.array([])
    all_hits   = np.concatenate([np.array(per_cls_hits[c],   np.int32)   for c in range(nc)]) if nc>0 else np.array([])
    ece   = expected_calibration_error(all_scores, all_hits, n_bins=ece_bins) if len(all_scores) else None
    brier = brier_score(all_scores, all_hits) if len(all_scores) else None

    # top-k image-level presence
    def top_k_accuracy(per_img, K):
        correct = 0; total = 0
        for present, confmap in per_img:
            if not present: continue
            total += len(present)
            topK = [] if len(confmap)==0 else [c for c,_ in sorted(confmap.items(), key=lambda x:-x[1])[:K]]
            correct += sum(1 for c in present if c in topK)
        return correct/total if total>0 else 0.0
    top1 = top_k_accuracy(per_img_presence_top, 1)
    top3 = top_k_accuracy(per_img_presence_top, 3)

    # quick stress test on subset
    import albumentations as A
    CORR = {
        "gauss_noise": A.GaussNoise(p=1.0),           # 일부 버전에서 var_limit 경고 → 기본값 사용
        "motion_blur": A.MotionBlur(blur_limit=7, p=1.0),
        "brightness_contrast": A.RandomBrightnessContrast(0.3, 0.3, p=1.0),
    }

    def eval_subset(imgs, tfm=None, max_n=stress_max_imgs):
        from sklearn.metrics import average_precision_score
        per_s = [[] for _ in range(nc)]
        per_h = [[] for _ in range(nc)]
        gt_cnt = np.zeros(nc, int)
        cnt = 0
        for ip in imgs:
            if cnt >= max_n: break
            im = cv2.imread(str(ip))
            if im is None: continue
            H,W = im.shape[:2]
            lbl = val_lbl_dir / ip.with_suffix(".txt").name
            g_xyxy, g_cls = read_gt_label(lbl, W, H)
            for c in g_cls.tolist():
                if 0 <= c < nc: gt_cnt[c] += 1
            if tfm is not None:
                im = tfm(image=im)["image"]
            pred = model.predict(source=im, imgsz=imgsz, conf=0.001, iou=0.70,
                                 device=device, verbose=False, agnostic_nms=False)[0]
            if pred.boxes is not None and len(pred.boxes):
                p_xyxy = pred.boxes.xyxy.cpu().numpy().astype(np.float32)
                p_cls  = pred.boxes.cls.cpu().numpy().astype(np.int32)
                p_conf = pred.boxes.conf.cpu().numpy().astype(np.float32)
            else:
                p_xyxy = np.zeros((0,4), np.float32)
                p_cls  = np.zeros((0,), np.int32)
                p_conf = np.zeros((0,), np.float32)
            tpf, _ = greedy_match(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, iou_thr)
            for c in range(nc):
                m = (p_cls == c)
                if np.any(m):
                    per_s[c].extend(p_conf[m].tolist())
                    per_h[c].extend(tpf[m].astype(int).tolist())
            cnt += 1
        aps = []
        for c in range(nc):
            ys = np.array(per_s[c], np.float32); yt = np.array(per_h[c], np.int32)
            if len(ys)>0 and gt_cnt[c]>0 and len(np.unique(yt))>1:
                try:
                    aps.append(average_precision_score(yt, ys))
                except Exception:
                    pass
        return float(np.mean(aps)) if aps else None

    subset = val_imgs[:min(len(val_imgs), stress_max_imgs)]
    base_map50 = eval_subset(subset, None, stress_max_imgs)
    stress = {}
    for name, aug in CORR.items():
        m = eval_subset(subset, A.Compose([aug]), stress_max_imgs)
        stress[name] = {"mAP50_subset": m, "delta_vs_clean": (m - base_map50) if (m is not None and base_map50 is not None) else None}


    #add operating-point metrics and best-over-sweep
    if conf_eval is not None:
        Pmi, Rmi, F1mi, TP_e, FP_e, FN_e, Pma, Rma, F1ma = prf_at_threshold(
            per_image_preds, per_image_gts, conf_eval, iou_thr, nc
        )
        operating_point = {
            "conf_eval": float(conf_eval),
            "micro": {"precision": Pmi, "recall": Rmi, "f1": F1mi, "TP": TP_e, "FP": FP_e, "FN": FN_e},
            "macro": {"precision": Pma, "recall": Rma, "f1": F1ma}
        }

    sweep_best = sweep_best_f1(per_image_preds, per_image_gts, iou_thr, nc) if do_sweep else None

    return {
        "overall": {"top1_accuracy_presence": top1, "top3_accuracy_presence": top3},
        "precision_recall_f1": {
            "micro": {"precision": prec_micro, "recall": rec_micro, "f1": f1_micro},
            "macro": {"precision": prec_macro, "recall": rec_macro, "f1": f1_macro}
        },
        "pr_auc": {"per_class": pr_auc_values, "macro": (float(np.nanmean([v for v in pr_auc_values if v is not None])) if any(v is not None for v in pr_auc_values) else None)},
        "roc_auc_approx": {"per_class": roc_auc_values, "macro": roc_auc_macro, "note": "ROC on detection confidences is non-standard; compare with PR-AUC."},
        "calibration": {"ece": ece, "brier": brier, "bins": ece_bins},
        "efficiency": {"avg_latency_ms_per_image": avg_latency_ms, "throughput_fps": throughput_fps, "model_size_mb": model_size_mb},
        "stress_test": {"baseline_subset_mAP50": base_map50, "corruptions": stress, "subset_size": len(subset)},
        "operating_point": operating_point,
        "sweep_best": sweep_best,   
        "meta": {"images_evaluated": len(val_imgs), "iou_thr": iou_thr, "imgsz": imgsz, "device": ("cuda" if torch.cuda.is_available() else "cpu"), "nc": nc, "names": names, "data_yaml": str(Path(data_yaml).resolve())}
    }


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
    pred("runs/detect/train3/weights/best.pt", "road.PNG", 0.25)
    metrics_json = model_metrics(
        model_path="runs/detect/train3/weights/best.pt",
        imgsz=416,
        data_yaml="sign.yaml",
        conf_eval=0.49
    )
    with open("metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)