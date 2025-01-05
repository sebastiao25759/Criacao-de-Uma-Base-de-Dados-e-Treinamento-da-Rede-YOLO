from ultralytics import YOLO


def main():

    model = YOLO("yolov8n.pt")

    model.train(data="dogAndCat.yaml", epochs=30, device='cpu', imgsz=(400, 400))  # train the model
    metrics = model.val()

if __name__ == '__main__':
    main()