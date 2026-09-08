from pathoverse.models import ModelEngine


def test_model_registry_engine():
    engine = ModelEngine(
        model_name="vit-base",
        device="cpu",
        batch_size=2,
    )

    assert engine.model_name == "vit-base"
    assert engine.batch_size == 2
    assert engine.info().model_id == (
        "google/vit-base-patch16-224-in21k"
    )