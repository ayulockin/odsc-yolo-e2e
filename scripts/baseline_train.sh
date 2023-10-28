clear && python baseline_train.py yolov8n 5
rm -rf scripts/artifacts/ scripts/run scripts/wandb/

clear && python baseline_train.py yolov8s 5
rm -rf scripts/artifacts/ scripts/run scripts/wandb/

clear && python baseline_train.py yolov8m 5
rm -rf scripts/artifacts/ scripts/run scripts/wandb/

clear && python baseline_train.py yolov8l 5
rm -rf scripts/artifacts/ scripts/run scripts/wandb/

clear && python baseline_train.py yolov8x 5
rm -rf scripts/artifacts/ scripts/run scripts/wandb/