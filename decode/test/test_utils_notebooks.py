from pathlib import Path

from decode.test import asset_handler
from decode.utils import notebooks


def test_load_notebooks():
    test_notebook_folder = Path(__file__).resolve().parent / Path('assets/examples')
    test_notebook_folder.mkdir(exist_ok=True)

    with asset_handler.RMAfterTest(test_notebook_folder, True):
        notebooks.load_examples(test_notebook_folder)

        assert len(list(test_notebook_folder.glob('*.ipynb'))) == 4
        assert (test_notebook_folder / 'Introduction.ipynb').exists()
        assert (test_notebook_folder / 'Evaluation.ipynb').exists()
        assert (test_notebook_folder / 'Training.ipynb').exists()
        assert (test_notebook_folder / 'Fit.ipynb').exists()


def test_copy_pkg_file():
    from decode.utils import examples
    copy_dir = Path(__file__).resolve().parent / Path('assets/copy')
    copy_dir.mkdir(exist_ok=True)

    with asset_handler.RMAfterTest(copy_dir, True):
        notebooks.copy_pkg_file(examples, 'Introduction.ipynb', copy_dir)

        assert (copy_dir / 'Introduction.ipynb').exists(), "Copied file does not exist."
