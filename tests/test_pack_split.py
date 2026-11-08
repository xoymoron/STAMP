"""Keep train/validation/test membership inside each recorded sample pack."""

import json
from collections import Counter

import pytest

from stamp.data import AudioWithParametersDataset


@pytest.fixture
def pack_metadata(tmp_path):
    metadata = {}
    for pack, count in (("collection/kit_a", 10), ("collection/kit_b", 20)):
        for index in range(count):
            key = f"{pack}/{index:03d}"
            metadata[key] = {
                "filename": f"audio/{key}.wav",
                "feature_file": f"features/{key}.pt",
                "orig_relpath": f"{pack}/hits/{index:03d}.wav",
                "sample_pack_key": pack,
                "type": "custom",
                "num_samples": 100,
                "analysis_num_samples": 100,
                "modal_backend": "hybrid",
            }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "modal_config.json").write_text(
        json.dumps({"sample_rate": 8000, "hop_length": 16, "modal_backend": "hybrid"})
    )
    return tmp_path, metadata


def test_each_detailed_pack_contributes_to_each_split(pack_metadata):
    root, metadata = pack_metadata
    memberships = {}
    expected = {
        "train": {"collection/kit_a": 8, "collection/kit_b": 16},
        "val": {"collection/kit_a": 1, "collection/kit_b": 2},
        "test": {"collection/kit_a": 1, "collection/kit_b": 2},
    }
    for split in ("train", "val", "test"):
        dataset = AudioWithParametersDataset(
            root,
            "metadata.json",
            8000,
            None,
            split=split,
            seed=42,
        )
        memberships[split] = set(dataset.file_list)
        counts = Counter(metadata[key]["sample_pack_key"] for key in dataset.file_list)
        assert counts == expected[split]

    assert memberships["train"].isdisjoint(memberships["val"])
    assert memberships["train"].isdisjoint(memberships["test"])
    assert memberships["val"].isdisjoint(memberships["test"])
    assert set.union(*memberships.values()) == set(metadata)


def test_small_nested_packs_are_not_merged_before_rounding(pack_metadata):
    root, metadata = pack_metadata
    metadata = {
        key: item for key, item in metadata.items() if key.endswith(("/000", "/001"))
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    dataset = AudioWithParametersDataset(
        root,
        "metadata.json",
        8000,
        None,
        split="train",
        seed=42,
    )
    counts = Counter(metadata[key]["sample_pack_key"] for key in dataset.file_list)
    assert counts == {"collection/kit_a": 1, "collection/kit_b": 1}


def test_detailed_pack_filter_preserves_the_full_dataset_split(pack_metadata):
    root, metadata = pack_metadata
    full = AudioWithParametersDataset(
        root,
        "metadata.json",
        8000,
        None,
        split="test",
        seed=42,
    )
    filtered = AudioWithParametersDataset(
        root,
        "metadata.json",
        8000,
        None,
        split="test",
        seed=42,
        sample_pack_keys=["collection/kit_a"],
    )
    expected = [
        key
        for key in full.file_list
        if metadata[key]["sample_pack_key"] == "collection/kit_a"
    ]
    assert len(filtered) == 1
    assert filtered.file_list == expected


def test_pack_split_is_the_default_and_repeatable(pack_metadata):
    root, _ = pack_metadata
    first = AudioWithParametersDataset(
        root, "metadata.json", 8000, None, split="train", seed=42
    )
    second = AudioWithParametersDataset(
        root, "metadata.json", 8000, None, split="train", seed=42
    )
    counts = Counter(first.metadata[key]["sample_pack_key"] for key in first.file_list)
    assert counts == {"collection/kit_a": 8, "collection/kit_b": 16}
    assert first.file_list == second.file_list
