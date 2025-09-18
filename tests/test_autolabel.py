from autolabel_detect.autolabel import DetectionResult, YOLOAutoLabeler

class DummyBox:
    def __init__(self, xyxy, cls):
        self.xyxy = [list(xyxy)]
        self.cls = [cls]


class DummyResult:
    def __init__(self, boxes):
        self.boxes = boxes


class DummyModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.called = False

    def __call__(self, image):
        self.called = True
        return [DummyResult(self._boxes)]

    @property
    def names(self):
        return {0: "object"}


class FakeImage:
    def __init__(self, shape):
        self.shape = shape


def test_process_image_returns_detections(tmp_path):
    image_array = FakeImage((20, 20, 3))
    loader_calls = []

    def loader(path):
        loader_calls.append(path)
        return image_array

    labeler = YOLOAutoLabeler(
        DummyModel([DummyBox([2, 4, 10, 12], 0)]),
        image_loader=loader,
        image_writer=lambda *_: None,
    )
    detections = labeler.process_image(tmp_path / "image.jpg")
    assert loader_calls, "loader should be called"
    assert detections == [DetectionResult(0, 0.3, 0.4, 0.4, 0.4)]


def test_process_folder_writes_outputs(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.jpg").write_text("placeholder")
    (images_dir / "b.png").write_text("placeholder")

    arrays = {
        images_dir / "a.jpg": FakeImage((10, 10, 3)),
        images_dir / "b.png": FakeImage((10, 10, 3)),
    }
    written = {}

    def loader(path):
        return arrays[path]

    def writer(path, image):
        written[path] = image

    labeler = YOLOAutoLabeler(
        DummyModel([DummyBox([0, 0, 5, 5], 0)]),
        image_loader=loader,
        image_writer=writer,
    )

    output_dir = tmp_path / "output"
    processed = labeler.process_folder(images_dir, output_dir)

    assert processed == 2
    assert (output_dir / "images" / "a.jpg") in written
    label_a = (output_dir / "labels" / "a.txt").read_text().strip()
    assert label_a.startswith("0 ")
