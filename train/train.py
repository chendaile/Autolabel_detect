from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r"premodel\yolo11n.pt", verbose=False)
    results = model.train(data=r"data.yaml",
                          batch=0.9,
                          cache=True,
                          time=0.2,
                          project=r"results\detect_n"
                          )
