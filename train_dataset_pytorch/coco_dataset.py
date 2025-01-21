import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil

CLASSES_OF_INTEREST = [
    "laptop", "book", "cell phone", "cup", "bottle",
    "keyboard", "mouse", "remote", "scissors", "clock"
]

def download_coco_dataset():
    if os.path.exists("desk_dataset"):
        shutil.rmtree("desk_dataset")

    os.makedirs("desk_dataset/images/train", exist_ok=True)
    os.makedirs("desk_dataset/images/val", exist_ok=True)
    os.makedirs("desk_dataset/labels/train", exist_ok=True)
    os.makedirs("desk_dataset/labels/val", exist_ok=True)

    train_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=CLASSES_OF_INTEREST,
        max_samples=1000
    )

    val_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=CLASSES_OF_INTEREST,
        max_samples=200
    )

    sorted_classes = sorted(CLASSES_OF_INTEREST)
    create_yaml_config("desk_dataset", sorted_classes)

    train_dataset.export(
        export_dir="desk_dataset",
        dataset_type=fo.types.YOLOv5Dataset,
        split="train",
        classes=sorted_classes
    )

    val_dataset.export(
        export_dir="desk_dataset",
        dataset_type=fo.types.YOLOv5Dataset,
        split="val",
        classes=sorted_classes
    )

def create_yaml_config(dataset_path, classes):
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val

names:
"""
    for idx, class_name in enumerate(classes):
        yaml_content += f"  {idx}: {class_name}\n"

    with open(f"{dataset_path}/dataset.yaml", "w") as f:
        f.write(yaml_content)

if __name__ == "__main__":
    download_coco_dataset()
