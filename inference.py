import os
from glob import glob

from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback


wandb.init(
    project="object-detection-bdd",
    name=f"inference/yolov8m",
    entity="reviewco",
    job_type="inference",
)

model_artifact = wandb.use_artifact('reviewco/object-detection-bdd/run_2yvwiulh_model:best', type='model')
model_artifact_dir = model_artifact.download()
model_file = glob(os.path.join(model_artifact_dir, "*.pt"))[0]

# Initialize YOLO Model
model = YOLO(model_file)

# Add W&B callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Perform prediction which automatically logs to a W&B Table
# with interactive overlays for bounding boxes, segmentation masks
model(glob("artifacts/bdd100k-ultralytics-format:v1/images/test/*"))

# Finish the W&B run
wandb.finish()