from unittest import mock
import pytest
from pathlib import Path

from decode.utils import files
from decode.utils import param_io


# @pytest.mark.parametrize("path,suffix,pattern", [
#     ("a.tiff", None, None)
# ])
def test_get_file_list_file(tmpdir):

    # file
    with mock.patch("pathlib.Path.is_file", return_value=True):
        out = files.get_file_list("a.tiff")

    assert out == [Path("a.tiff")]

    # dir
    # create dummy files (easier than mocking ...)
    for i in range(5):
        for suffix in [".tiff", ".csv"]:
            with (tmpdir / (str(i) + suffix)).open("w", encoding="utf-8") as f:
                f.write("")

    out = files.get_file_list(tmpdir, ".tiff")
    assert len(out) == 5
    assert out[0].suffix == ".tiff"

    out = files.get_file_list(tmpdir, None, "*.tiff")
    assert len(out) == 5
    assert out[0].suffix == ".tiff"


def test_load_default_cfg():
    """Test depends on package included fit.yaml"""

    fit_cfg = param_io.RecursiveNamespace(**param_io.load_fit_cfg())
    with mock.patch.object(param_io, 'load_params') as mock_load:
        mock_load.return_value = param_io.RecursiveNamespace(**{
            'Camera': {'em_gain': 100, 'dfg': 42}
        })
        default = files._load_default_cfg(fit_cfg)

    assert default.frame_path is None
    assert default.model_path == fit_cfg.model_path
    assert default.output_path == '.h5'
    assert default.param.Camera.em_gain == 50
    assert default.param.Camera.dfg == 42


def test_compile_fit():
    pass
