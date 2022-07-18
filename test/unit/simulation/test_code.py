import pytest
import torch

from decode.simulation import code


def test_code():
    codes = [1, 2, 3]
    c = code.Code(codes)

    samples = c.sample(10)

    assert len(samples) == 10
    assert isinstance(samples, torch.LongTensor)
    assert set(codes).issubset(set(samples.tolist()))


@pytest.fixture()
def codebook():
    return code.CodeBook({0: (True, False), 5: (True, True)})


@pytest.mark.parametrize("n", [0, 10])
@pytest.mark.parametrize("sampler_method", ["sample_codes", "sample_bits"])
def test_codebook_sample_codes_bits(n, sampler_method, codebook):
    # tests sample_code and sample_bits
    c = getattr(codebook, sampler_method)(n)
    assert len(c) == n


def test_codebook_invert(codebook):
    # backward transform
    bits = [(True, True), (True, False), (True, True)]
    code_expct = [5, 0, 5]

    code_out = codebook.invert(bits)

    assert code_out == code_expct, "Incorrect output codes"
