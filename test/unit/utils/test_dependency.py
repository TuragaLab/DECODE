import pytest
import decode.utils.dependency as dep

run_deps = [
    'pytorch=1.7.1',
    'numpy',
    'tifffile>=2021.1'
]

dev_deps = [
    'pytest'
]

doc_deps = [
    'sphinx'
]

conda_channel = [
    'pytorch',
    'conda-forge'
]

conda_build = {
    'build': ['a'],
    'host': ['b'],
    'run': {
        'abc': None,
        'tifffile>=2021.1': 'xifffile=2021.1'
    }
}

pip = {
    'run': {
        'only_pip': None,
        'pytorch=1.7.1': 'torch=1.7.1',
    }
}


@pytest.mark.parametrize("mode", ['txt', 'env'])
@pytest.mark.parametrize("level", ['run', 'dev', 'doc'])
def test_conda_environment(mode, level):
    out = dep.conda(run_deps, dev_deps, doc_deps, conda_channel, level, mode)

    if mode == 'txt':

        assert out[0] == 'pytorch==1.7.1'
        assert out[1] == 'numpy'
        assert out[2] == 'tifffile>=2021.1'

        if mode == 'dev':
            assert out[3] == 'pytest'

        elif mode == 'doc':
            assert out[3] == 'sphinx'

    elif mode == 'env':

        assert out['name'][:6] == 'decode'
        assert out['channels'] == conda_channel
        assert out['dependencies'][:3] == run_deps

        if mode == 'dev':
            assert out['dependencies'][[3]] == dev_deps

        elif mode == 'docs':
            assert out['dependencies'][[4]] == doc_deps


def test_conda_meta():
    out = dep.conda_meta(run_deps, conda_build)

    assert 'a' in out['build']
    assert 'b' in out['host']

    assert 'abc' in out['run']
    assert 'xifffile=2021.1' in out['run']
    assert 'tifffile>=2021.1' not in out['run']


@pytest.mark.parametrize("level", ['run', 'dev', 'docs'])
def test_pip(level):

    out = dep.pip(run_deps, dev_deps, doc_deps, pip, level)

    if level == 'run':

        assert len(out) == 4

        assert 'torch==1.7.1' in out
        assert 'numpy' in out
        assert 'tifffile>=2021.1' in out
        assert 'only_pip' in out

    elif level == 'dev':

        assert len(out) == 1
        assert 'pytest' in out

    elif level == 'docs':

        assert len(out) == 1
        assert 'sphinx' in out
