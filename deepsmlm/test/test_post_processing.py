import torch
import pytest

import deepsmlm.neuralfitter.post_processing as post


@pytest.fixture(scope='module')
def cc():
    return post.ConnectedComponents(0.3, 0., 2)


@pytest.fixture(scope='module')
def cc_offset():
    return post.CC5ChModel(0.3, 0., 2)


def test_connected_components(cc):
    p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
    clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
    assert torch.eq(clusix, cc.compute_cix(p_map)).all()


def test_connected_components_offset(cc_offset):
    p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
    clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
    features = torch.cat((
        p_map.unsqueeze(0),
        torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0),
        torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0)
    ), 0)
    out_feat, out_p = cc_offset.average_features(features, clusix, p_map)

    expect_outcome_feat = torch.tensor([[0.5, 0.75, 0.75]])
    expect_p = torch.tensor([1.])

    assert torch.eq(expect_outcome_feat, out_feat).all()
    assert torch.eq(expect_p, out_p).all()

