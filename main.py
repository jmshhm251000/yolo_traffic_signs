import os
import torch, cv2
from ultralytics import YOLO
import yaml
from pathlib import Path
import albumentations as A


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
    #train_yolo("yolo11n.pt", 640, 20)
    #finetune_yolo("runs/detect/train/weights/best.pt", 416, 20)
    #pred("runs/detect/train3/weights/best.pt", "sample.png", 0.25)