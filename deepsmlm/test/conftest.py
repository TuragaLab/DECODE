import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", default=False, help="run tests that include plotting via matplotlib")
    parser.addoption(
        "--slow", action="store_true", default=False, help="run tests that are indicated as being slow"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "plot: tests with primary purpose of plotting")
    config.addinivalue_line("markers", "slow: tests that could not be made faster")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot") or config.getoption("--slow"):
        # --plot and or --slow given in cli: do not skip slow or plotting tests
        return
    skip = pytest.mark.skip(reason="need --plot / --slow option to run")
    for item in items:
        if "plot" in item.keywords or "slow" in item.keywords:
            item.add_marker(skip)
