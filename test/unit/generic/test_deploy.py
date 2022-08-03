import pytest

from decode.generic import deploy


@pytest.mark.parametrize(
    "deps,deps_available", [
        ("numpy", True),
        ("fantasy_deps_should_not_exist", False)
])
def test_raise_optional_deps(deps, deps_available):
    @deploy.raise_optional_deps(deps, "dep missing")
    def fn_with_dep():
        pass

    if deps_available:
        fn_with_dep()
    else:
        with pytest.raises(ImportError):
            fn_with_dep()
