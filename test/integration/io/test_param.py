from decode.io import param


def test_load_reference():
    ref = param.load_reference()
    assert "calibration_file" in ref.InOut


def test_copy_reference(tmpdir):
    ref, friendly = param.copy_reference(tmpdir)

    assert ref.is_file()
    assert friendly.is_file()
