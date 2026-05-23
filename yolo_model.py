from ultralytics import YOLO
from pathlib import Path


# 1. Load Dataset 
dataset_path = "anti_spoofing_split_combined"

if not Path(dataset_path).exists():
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

print("Dataset found:", dataset_path)


# 2. Load YOLO classification model
model = YOLO("yolo11n-cls.pt")


# 3. Train model
results = model.train(
    data=str(dataset_path),
    epochs=30,
    patience=10, #earlystopping
    imgsz=224,
    batch=32,
    # device=0,
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
 



