from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model="../premodel/yolo11n.pt", verbose=False)
    results = model.train(data="../data.yaml",
                          batch=0.9,
                          cache=True,
                          time=0.2,
                          project="../results/detect_n"
                          )
