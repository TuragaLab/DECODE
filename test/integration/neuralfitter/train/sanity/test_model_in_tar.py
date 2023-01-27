import pytest
import torch

from decode.neuralfitter.train import setup_cfg


# @pytest.fixture
# def sampler(request, cfg_trainable):
#     if request.param == "train":
#         return setup_cfg.setup_sampler(cfg_trainable)[0]
#     elif request.param == "val":
#         return setup_cfg.setup_sampler(cfg_trainable)[1]
#     else:
#         raise ValueError("Invalid request.")


def _sampler_factory(split, code, request):
    if code is None:
        cfg = request.getfixturevalue("cfg_trainable")
    else:
        cfg = request.getfixturevalue("cfg_multi")

    s = setup_cfg.setup_sampler(cfg)
    if split == "train":
        return s[0]
    elif split == "val":
        return s[1]
    else:
        raise ValueError("Invalid split.")


# @pytest.mark("slow")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("code", [None, [0, 1, 2]])
def test_model_input_sanity(split, code, request):
    # test that model input is sane, i.e. no nan, inf and in proper range

    sampler = _sampler_factory(split, code, request)
    sampler.sample()

    sample_input = sampler.input
    sample_input = torch.stack([s for s in sample_input], dim=0)

    assert len(sampler) > 400
    assert not torch.isnan(sample_input).any()
    assert not torch.isinf(sample_input).any()


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("code", [None, [0, 1, 2]])
def test_target_sanity(split, code, request):
    sampler = _sampler_factory(split, code, request)
    sampler.sample()

    tar = sampler.target
    em, mask, bg = [], [], []
    for t in tar:
        em.append(t[0])
        mask.append(t[1])
        bg.append(t[2])

    em = torch.stack(em, dim=0)
    mask = torch.stack(mask, dim=0)
    bg = torch.stack(bg, dim=0)

    mask = mask.max(-1)[0]
    assert not torch.isnan(em[mask]).any()
    assert torch.isnan(em[~mask]).all()
    assert not torch.isnan(bg).any()
