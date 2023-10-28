import os
import sys
import yaml

import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO


model_name = sys.argv[1]
epochs = int(sys.argv[2])

wandb.init(
    project="mlops-odsc",
    name=f"baseline/{model_name}",
    entity="ml-colabs",
    job_type="train/baseline",
    save_code=True,
)

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


model = YOLO(f"{model_name}.pt")
add_wandb_callback(model, enable_model_checkpointing=True)
model.train(data=metadata_file, epochs=epochs, imgsz=640)
model.val()

wandb.alert(title="Run Finished", text="Run finished.")