"""Unit tests for dataset loading, directory validation, and statistics scanner."""

from pathlib import Path

from src.data.dataset_loader import MRIDatasetLoader


def test_dataset_loader_initialization():
    """Asserts dataset loader attributes are initialized correctly."""
    dataset_path = "dataset"
    loader = MRIDatasetLoader(dataset_path)

    assert loader.dataset_path == Path(dataset_path)
    assert loader.splits == ["train", "validation", "test"]
    assert loader.classes == []
    assert isinstance(loader.corrupted_files, dict)
