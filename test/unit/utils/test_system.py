from decode.utils import system


def test_collect_system():
    out = system.collect_system()

    assert "hostname" in out
    assert "os" in out
