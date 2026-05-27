from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


# 1. Load Dataset 
DATASET_PATHS = [
    SCRIPT_DIR / "anti_spoofing_split_combined",
    PROJECT_ROOT / "anti_spoofing_split_combined",
]
dataset_path = next((path for path in DATASET_PATHS if path.exists()), None)

if dataset_path is None:
    checked_paths = ", ".join(str(path) for path in DATASET_PATHS)
    raise FileNotFoundError(f"Dataset folder not found. Checked: {checked_paths}")

print("Dataset found:", dataset_path)


# 2. Load YOLO classification model
model = YOLO(SCRIPT_DIR / "yolo11n-cls.pt")


# 3. Train model
results = model.train(
    data=str(dataset_path),
    epochs=30,
    patience=10, #earlystopping
    imgsz=224,
    batch=32,
    # device=0,
    project=str(SCRIPT_DIR),
    name="anti_spoofing_yolo_with_aug_final01",

    
    # Data augmentation
    degrees=10,        # small rotation
    translate=0.1,     # small shifting
    scale=0.2,         # zoom in/out
    shear=5,           # slight tilt
    fliplr=0.5,        # horizontal flip
    flipud=0.0,        # do not flip upside down for faces

    hsv_h=0.015,       # slight colour change
    hsv_s=0.4,         # saturation change
    hsv_v=0.4,         # brightness change

    erasing=0.2,       # random erasing/cutout
)


# 4. Load best trained model
best_model_path = Path(results.save_dir) / "weights" / "best.pt"
trained_model = YOLO(best_model_path)


# 5. Validate model
val_results = trained_model.val(
    data=str(dataset_path),
    imgsz=224,
    batch=32
)

print("Validation results:")
print(val_results.results_dict)


# 6. Test model
test_results = trained_model.val(
    data=str(dataset_path),
    split="test",
    imgsz=224,
    batch=32
)

print("Test results:")
print(test_results.results_dict)
 



