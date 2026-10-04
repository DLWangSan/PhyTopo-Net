import torch

from model import PhyTopoNet


def test_phytopo_forward_shape():
    model = PhyTopoNet()
    x_met = torch.zeros(1, 7, 64, 64)
    x_geo = torch.zeros(1, 5, 64, 64)
    with torch.no_grad():
        output = model(x_met, x_geo)
    assert output.shape == (1, 4, 64, 64)
