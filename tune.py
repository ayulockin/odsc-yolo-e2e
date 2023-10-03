import shutil

import wandb
from ultralytics import YOLO


wandb.init(project="object-detection-bdd", entity="reviewco")
config = wandb.config

model = YOLO(f"{config.model}.pt")
model.train(
    data="artifacts/bdd100k-ultralytics-format:v1/data.yaml",
    epochs=5,
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

wandb.alert(title="Run Finished", text="Run finished.")
wandb.finish()
shutil.rmtree("./runs")
