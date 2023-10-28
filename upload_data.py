import wandb

run = wandb.init(
    project="mlops-odsc",
    entity="ml-colabs",
    job_type="upload-raw-data",
)

artifact = wandb.Artifact("brackish-underwater-raw", type="dataset", metadata={"source": "https://public.roboflow.com/object-detection/brackish-underwater"})

artifact.add_dir("data/")

run.log_artifact(artifact)