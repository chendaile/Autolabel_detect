from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import autolabel_detect.cli as cli
from autolabel_detect.datasets import SplitConfig, split_dataset

@pytest.fixture()
def parser():
    return cli.build_parser()


def test_parser_has_subcommands(parser):
    subcommands = set(parser._subparsers._group_actions[0]._name_parser_map.keys())
    assert {"detect", "autolabel", "train", "split", "capture"}.issubset(subcommands)


def test_detect_command_invokes_detector(monkeypatch, tmp_path):
    called = {}

    class FakeDetector:
        def __init__(self, config):
            called["config"] = config

        def run(self):
            called["run"] = True

    monkeypatch.setattr(cli, "YOLODetector", FakeDetector)
    args = [
        "detect",
        "--model",
        str(tmp_path / "model.pt"),
        "--source",
        "image.jpg",
        "--flip",
        "2",
    ]
    cli.main(args)
    assert called["config"].flip_mode is None
    assert called["run"] is True


def test_autolabel_command_invokes_processor(monkeypatch, tmp_path):
    called = {}

    class FakeLabeler:
        def __init__(self, config):
            called["config"] = config

        def process(self):
            called["processed"] = True

    monkeypatch.setattr(cli, "YOLOAutoLabeler", FakeLabeler)
    args = [
        "autolabel",
        "--model",
        str(tmp_path / "model.pt"),
        "--input",
        str(tmp_path),
    ]
    cli.main(args)
    assert called["processed"] is True


def test_train_command_invokes_training(monkeypatch, tmp_path):
    captured = {}

    def fake_train(config):
        captured["config"] = config

    monkeypatch.setattr(cli, "train_model", fake_train)
    args = [
        "train",
        "--time",
        "0.1",
    ]
    cli.main(args)
    assert captured["config"].time == pytest.approx(0.1)


def test_split_command_outputs_counts(monkeypatch, tmp_path, capsys):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir(parents=True)
    for idx in range(4):
        (dataset / "images" / f"img{idx}.jpg").write_bytes(b"data")
        (dataset / "labels" / f"img{idx}.txt").write_text("0 0.5 0.5 0.5 0.5")

    def fake_split(config):
        assert config.dataset_root == dataset
        return 3, 1

    monkeypatch.setattr(cli, "split_dataset", fake_split)
    args = [
        "split",
        "--dataset",
        str(dataset),
    ]
    cli.main(args)
    output = capsys.readouterr().out
    assert "Train images: 3" in output


def test_capture_command_handles_lifecycle(monkeypatch, tmp_path):
    log = {}

    class FakeCamera:
        def __init__(self, config):
            log["config"] = config

        def initialize(self):
            return True

        def capture_frames(self, frames, interval, show_preview):
            log["frames"] = frames
            log["interval"] = interval
            log["preview"] = show_preview

        def release(self):
            log["released"] = True

    monkeypatch.setattr(cli, "UniversalCamera", FakeCamera)
    args = [
        "capture",
        "--frames",
        "5",
        "--no-preview",
    ]
    cli.main(args)
    assert log["frames"] == 5
    assert log["preview"] is False
    assert log["released"] is True


def test_split_dataset_function(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir(parents=True)
    for idx in range(6):
        (dataset / "images" / f"sample{idx}.jpg").write_bytes(b"img")
        (dataset / "labels" / f"sample{idx}.txt").write_text("0 0.5 0.5 0.5 0.5")

    output = tmp_path / "out"
    train_count, val_count = split_dataset(SplitConfig(dataset_root=dataset, train_ratio=0.5, output_root=output))
    assert train_count + val_count == 6
    assert train_count == 3
    train_images = list((output / "train" / "images").glob("*.jpg"))
    val_images = list((output / "validation" / "images").glob("*.jpg"))
    assert len(train_images) == train_count
    assert len(val_images) == val_count
