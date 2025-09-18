from unittest import mock

from autolabel_detect import cli


def test_cli_autolabel_invokes_labeler(monkeypatch, capsys):
    mock_labeler = mock.MagicMock()
    monkeypatch.setattr(cli, "YOLOAutoLabeler", mock.MagicMock(return_value=mock_labeler))

    cli.main(["autolabel", "model.pt", "input", "output", "--no-overwrite"])

    cli.YOLOAutoLabeler.assert_called_once_with("model.pt", class_names=None)
    mock_labeler.process_folder.assert_called_once_with("input", "output", overwrite=False)
    assert "Processed" in capsys.readouterr().out


def test_cli_train_invokes_trainer(monkeypatch):
    mock_trainer = mock.MagicMock()
    monkeypatch.setattr(cli, "YOLOTrainer", mock.MagicMock(return_value=mock_trainer))

    cli.main(["train", "project", "data.yaml", "--model", "model.pt", "--time", "1"])

    cli.YOLOTrainer.assert_called_once_with("model.pt")
    assert mock_trainer.train.call_args.kwargs == {}
    config = mock_trainer.train.call_args.args[0]
    assert config.project == "project"
    assert config.data == "data.yaml"


def test_cli_split_reports_counts(monkeypatch, capsys):
    monkeypatch.setattr(cli, "split_dataset", mock.MagicMock(return_value=([1, 2], [3])))

    cli.main(["split", "dataset", "output", "--train-ratio", "0.5"])

    cli.split_dataset.assert_called_once()
    assert "train split" in capsys.readouterr().out
