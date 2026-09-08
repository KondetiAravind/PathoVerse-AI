from pathlib import Path

from pathoverse.models.adapters import (
    GigaPathFlashAdapter,
    TimmVisionAdapter,
)


def test_timm_adapter_initialization():

    adapter = TimmVisionAdapter(
        model_id="vit_base_patch16_224",
        device="cpu",
        batch_size=2,
        pretrained=False,
    )

    assert adapter.model_id == "vit_base_patch16_224"
    assert str(adapter.device) == "cpu"


def test_timm_adapter_info():

    adapter = TimmVisionAdapter(
        model_id="vit_base_patch16_224",
        device="cpu",
        batch_size=2,
        pretrained=False,
    )

    info = adapter.info()

    assert info.name == "ViT-B/16"
    assert info.input_size == 224


def test_gigapath_adapter_initialization():

    adapter = GigaPathFlashAdapter(
        device="cpu",
        batch_size=2,
    )

    assert adapter.embedding_dim == 384
    assert adapter.input_size == 224


def test_gigapath_adapter_info():

    adapter = GigaPathFlashAdapter(
        device="cpu",
        batch_size=2,
    )

    info = adapter.info()

    assert info.name == "GigaPath-Flash"
    assert info.embedding_dim == 384
    assert info.input_size == 224