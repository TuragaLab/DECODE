from pathlib import Path

from decode.utils import notebooks


def test_load_notebooks(tmpdir):
    test_notebook_folder = Path(tmpdir)
    notebooks.load_examples(test_notebook_folder)

    assert len(list(test_notebook_folder.glob("*.ipynb"))) == 4
    assert (test_notebook_folder / "Introduction.ipynb").exists()
    assert (test_notebook_folder / "Evaluation.ipynb").exists()
    assert (test_notebook_folder / "Training.ipynb").exists()
    assert (test_notebook_folder / "Fitting.ipynb").exists()


def test_copy_pkg_file(tmpdir):
    from decode.utils import examples

    copy_dir = Path(tmpdir)

    notebooks.copy_pkg_file(examples, "Introduction.ipynb", copy_dir)

    assert (copy_dir / "Introduction.ipynb").exists(), "Copied file does not exist."
