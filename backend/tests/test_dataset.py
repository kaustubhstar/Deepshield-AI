from app.ml.dataset import Sample, split_by_source


def test_official_split_column_is_preserved():
    from app.ml.dataset import splits_from_manifest

    samples = [
        Sample(path=__import__("pathlib").Path("a.jpg"), label=0, source_id="a", split="train"),
        Sample(path=__import__("pathlib").Path("b.jpg"), label=1, source_id="b", split="valid"),
        Sample(path=__import__("pathlib").Path("c.jpg"), label=0, source_id="c", split="test"),
    ]
    splits = splits_from_manifest(samples)
    assert splits is not None
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    assert splits["train"][0].source_id == "a"


def test_source_ids_never_cross_splits():
    samples = []
    for source in range(20):
        for frame in range(3):
            samples.append(
                Sample(path=__import__("pathlib").Path(f"{source}_{frame}.jpg"), label=source % 2, source_id=f"vid{source}")
            )
    splits = split_by_source(samples, seed=42)
    train_src = {s.source_id for s in splits["train"]}
    val_src = {s.source_id for s in splits["val"]}
    test_src = {s.source_id for s in splits["test"]}
    assert train_src.isdisjoint(val_src)
    assert train_src.isdisjoint(test_src)
    assert val_src.isdisjoint(test_src)


def test_checkpoint_missing_raises():
    from pathlib import Path

    import torch

    from app.ml.checkpoint import CheckpointError, load_checkpoint_bundle

    try:
        load_checkpoint_bundle(Path("missing.pt"), Path("missing.json"), torch.device("cpu"))
        raise AssertionError("expected CheckpointError")
    except CheckpointError:
        pass
