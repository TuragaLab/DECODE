import numpy as np
import pytest
import torch

import decode.simulation.sampler as emgen
from decode.emitter.emitter import EmitterSet
from decode.simulation import structures


def test_int_uniform():
    s = emgen._IntUniform(-5.0, 6.0)
    samples = s.sample(10000)

    assert samples.min() == -5
    assert samples.max() == 5


def minimal_structure():
    return structures.RandomStructure((10.0, 20.0), (30.0, 40.0), (1000, 2000.0))


@pytest.fixture
def static_emitter_sampler():
    return emgen.EmitterSamplerStatic(
        structure=minimal_structure(),
        intensity=(100.0, 200.0),
        em_num=100.0,
        frame=(-5, 10),
        frame_range=(-5, 10)
    )


@pytest.fixture
def blinking_emitter_sampler():
    return emgen.EmitterSamplerBlinking(
        structure=minimal_structure(),
        intensity=(100.0, 200.),
        em_num=100.0,
        lifetime=2.,
        frame_range=(-5, 10),
    )


@pytest.mark.parametrize(
    "sampler",
    [
        "static_emitter_sampler",
        "blinking_emitter_sampler",
    ],
)
def test_sample(sampler, request):
    sampler = request.getfixturevalue(sampler)
    assert isinstance(sampler.sample(), EmitterSet), "Expected EmitterSet as output"


@pytest.mark.parametrize(
    "sampler",
    [
        "static_emitter_sampler",
        "blinking_emitter_sampler",
    ],
)
@pytest.mark.parametrize("n", [0, 5, 1000])
def test_sample_n(sampler, n, request):
    sampler = request.getfixturevalue(sampler)
    assert len(sampler.sample_n(n)) == n


@pytest.mark.parametrize(
    "sampler",
    [
        "static_emitter_sampler",
        "blinking_emitter_sampler",
    ],
)
def test_sample_emitter_properties(sampler, request):
    sampler = request.getfixturevalue(sampler)
    em = sampler.sample()

    assert ((-5 <= em.frame_ix) * (em.frame_ix < 10)).all(), "Incorrect frame index"

    if isinstance(sampler, emgen.EmitterSamplerStatic):
        # static sampling should result in unique id
        np.testing.assert_array_equal(em.id, torch.arange(len(em), dtype=torch.long))


@pytest.mark.parametrize(
    "sampler",
    [
        "static_emitter_sampler",
        "blinking_emitter_sampler",
    ],
)
@pytest.mark.parametrize("code", [None, emgen.code.CodeBook({1: 10, 2: 20})])
def test_sample_code(code, sampler, request):
    sampler = request.getfixturevalue(sampler)
    sampler.code_sampler = code
    e = sampler.sample()

    if code is not None:
        assert set(e.code.unique().tolist()).issubset({1, 2})
    else:
        assert e.code is None


@pytest.mark.parametrize(
    "sampler",
    [
        "static_emitter_sampler",
        "blinking_emitter_sampler",
    ],
)
def test_sample_average(sampler, request):
    # tests whether average number returned by EmitterPopper on a single frame matches
    # the specification (approximately)
    sampler = request.getfixturevalue(sampler)

    n_emitter = [len(sampler.sample().iframe[0]) for _ in range(100)]
    avg = torch.tensor(n_emitter).float().mean()

    assert avg == pytest.approx(100., abs=5), "Emitter average seems to be off."


def test_sample_blinking_props(blinking_emitter_sampler):
    sampler = blinking_emitter_sampler

    for t in sampler._time_buffered:
        assert isinstance(t, float)
    assert sampler._time_buffered == (-11., 14.)
    assert sampler._duration_buffered == 25.


@pytest.mark.slow()
def test_sample_blinking_uniformity():
    # tests whether there are approx. equal amount of fluorophores on all frames.
    # tested with a high number for statistical reasons.
    # this test could fail by statistics

    sampler = emgen.EmitterSamplerBlinking(
        structure=minimal_structure(),
        intensity=(0., 1.),
        frame_range=(0, 1000),
        em_num=10000.,
        lifetime=2.,
    )
    em = sampler.sample()

    bin_count, _ = np.histogram(em.frame_ix, bins=np.arange(1001))
    bin_count = torch.from_numpy(bin_count)

    np.testing.assert_allclose(
        bin_count, torch.ones_like(bin_count) * 10000, atol=500
    )
    assert bin_count.float().mean() == pytest.approx(10000, rel=0.05)
