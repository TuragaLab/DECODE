from decode.io import param


def test_load_reference():
    ref = param.load_reference()
    assert isinstance(ref, dict)
    assert "calibration" in ref["Paths"]


def test_copy_reference(tmpdir):
    ref, friendly = param.copy_reference(tmpdir)

    assert ref.is_file()
    assert friendly.is_file()
