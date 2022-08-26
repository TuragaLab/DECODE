import pytest
import pydantic
from unittest import mock
from pathlib import Path

from decode.utils import param_schema


@pytest.mark.parametrize("calib_exists", [True, False])
@pytest.mark.parametrize("exp_exists", [True, False])
def test_inout_training(calib_exists, exp_exists):
    kwargs = {
        "calibration_file": Path("calib.mat"),
        "experiment_out": Path("a/dir")
    }

    with mock.patch.object(Path, "exists", return_value=True):
        with mock.patch.object(Path, "is_file", return_value=calib_exists):
            with mock.patch.object(Path, "is_dir", return_value=exp_exists):
                if calib_exists and exp_exists:
                    param_schema.InOutSchemaTraining(**kwargs)
                else:
                    with pytest.raises(pydantic.ValidationError):
                        param_schema.InOutSchemaTraining(**kwargs)
