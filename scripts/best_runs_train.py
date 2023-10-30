import os
import yaml
import shutil

from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback


api = wandb.Api()
sweep = api.sweep("ml-colabs/mlops-odsc/sweeps/8nh5rfkp")

# Get best run parameters
best_run = sweep.best_run(order="metrics/mAP50-95(B)")
best_parameters = best_run.config

wandb.init(project="mlops-odsc", entity="ml-colabs", config=best_parameters, save_code=True, job_type="final/train")
config = wandb.config

artifact = wandb.use_artifact("ml-colabs/mlops-odsc/brackish-underwater-raw:v0", type="dataset")
artifact_dir = artifact.download()

metadata_file = os.path.join(artifact_dir, "data.yaml")
with open(metadata_file, "r") as yaml_file:
    metadata = yaml.safe_load(yaml_file)
print(metadata)
metadata["path"] = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), artifact_dir
)
with open(metadata_file, "w") as yaml_file:
    yaml.dump(metadata, yaml_file)

model = YOLO(f"{config.model}.pt")
add_wandb_callback(model, enable_model_checkpointing=True)
model.train(
    data=metadata_file,
    epochs=30,
    batch=config.batch_size,
    imgsz=config.imgsz,
    lr0=config.lr0,
    lrf=config.lrf,
    momentum=config.momentum,
    weight_decay=config.weight_decay,
    warmup_epochs=config.warmup_epochs,
    warmup_momentum=config.warmup_momentum,
    box=config.box,
    cls=config.cls,
    hsv_h=config.hsv_h,
    hsv_s=config.hsv_s,
    hsv_v=config.hsv_v,
    degrees=config.degrees,
    translate=config.translate,
    scale=config.scale,
    shear=config.shear,
    perspective=config.perspective,
    flipud=config.flipud,
    fliplr=config.fliplr,
    mosaic=config.mosaic,
    mixup=config.mixup,
    copy_paste=config.copy_paste,
)
model.val()

wandb.alert(title="Run Finished", text="Run finished.")
wandb.finish()
shutil.rmtree("./runs")
shutil.rmtree("./artifacts")
