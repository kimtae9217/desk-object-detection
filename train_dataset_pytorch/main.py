from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='desk_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='desk_detector',
        patience=20,  # early stopping patience
        save=True,   # save checkpoints
    )

    model.export(format='onnx', simplify=True)

if __name__ == '__main__':
    train_model()
