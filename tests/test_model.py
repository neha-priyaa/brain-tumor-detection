import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_custom_cnn_shape():
    from src.model import build_custom_cnn

    m = build_custom_cnn()
    assert m.input_shape == (None, 224, 224, 3)
    assert m.output_shape == (None, 4)
    assert m.optimizer is not None


def test_efficientnet_shape_and_freeze():
    from src.model import build_efficientnet

    m = build_efficientnet(dropout=0.2, freeze_base=True)
    assert m.input_shape == (None, 224, 224, 3)
    assert m.output_shape == (None, 4)
    base = [l for l in m.layers if l.name.startswith("efficientnetb0")]
    assert base and base[0].trainable is False

    m2 = build_efficientnet(dropout=0.2, freeze_base=False)
    base2 = [l for l in m2.layers if l.name.startswith("efficientnetb0")]
    assert base2 and base2[0].trainable is True


def test_factory_unknown_raises():
    from src.model import get_model

    try:
        get_model("resnet")
        assert False, "should raise ValueError"
    except ValueError:
        pass
