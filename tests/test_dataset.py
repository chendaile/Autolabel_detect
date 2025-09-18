from pathlib import Path

from autolabel_detect.dataset import split_dataset


def create_fake_dataset(base: Path, count: int) -> None:
    base.mkdir()
    images = base / "images"
    labels = base / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(count):
        image_path = images / f"img_{i}.jpg"
        image_path.write_bytes(b"fake")
        label_path = labels / f"img_{i}.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2")


def test_split_dataset_creates_expected_structure(tmp_path):
    dataset_dir = tmp_path / "dataset"
    create_fake_dataset(dataset_dir, 10)

    train, val = split_dataset(dataset_dir, tmp_path / "output", train_ratio=0.7, seed=42)

    assert len(train) == 7
    assert len(val) == 3
    assert (tmp_path / "output" / "train" / "images").is_dir()
    assert (tmp_path / "output" / "validation" / "labels").is_dir()
