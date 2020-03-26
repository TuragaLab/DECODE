import torch

from deepsmlm.neuralfitter.models import model_offset as models


def trace_network(model):
    traced_net = torch.jit.trace(model, torch.rand(1, 3, 64, 64))
    return traced_net


if __name__ == "__main__":
    print(torch.__version__)
    model = models.OffsetUnet(3)
    # trace = trace_network(model)
    torch.jit.save(model, 'temp.pt')