import os
import time

import pytest
import deepsmlm.neuralfitter.models.model as model
import deepsmlm.utils.model_io as io_model

deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


@pytest.fixture
def unet():
    """Inits an arbitrary UNet."""
    return model.UNet(3, 5)


@pytest.fixture
def model_interface(unet):
    """Inits the i/o interface."""
    return io_model.LoadSaveModel(unet, deepsmlm_root + 'deepsmlm/test/assets/test_load_save.pt', None,
                                  1, 1e-3)


def test_save(model_interface, unet):
    """Tests saving the model, and tests whether a new suffix is appended."""
    model_interface.save(unet, None)
    exists = os.path.isfile(deepsmlm_root + 'deepsmlm/test/assets/test_load_save_0.pt')
    if not exists:
        pytest.fail("Model could not be found after saving.")

    time.sleep(2)
    model_interface.save(unet, None)
    exists = os.path.isfile(deepsmlm_root + 'deepsmlm/test/assets/test_load_save_1.pt')
    if not exists:
        pytest.fail("Model could not be found after saving.")

    os.remove(deepsmlm_root + 'deepsmlm/test/assets/test_load_save_0.pt')
    os.remove(deepsmlm_root + 'deepsmlm/test/assets/test_load_save_1.pt')


def test_load_init(model_interface):
    model_interface.load_init(deepsmlm_root + 'deepsmlm/test/assets/test_load_save_0.pt')


