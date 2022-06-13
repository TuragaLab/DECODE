from unittest import mock

from decode.generic import utils


def test_emitter_composite_attribute_modifier():
    mod = utils.CompositeAttributeModifier(
        {"a": lambda x: x * 2, "b": lambda p: p / 2}
    )

    x = mock.MagicMock()
    x.a = [2]
    x.b = 10

    y = mod.forward(x)

    assert y.a == [2, 2]
    assert y.b == 5
