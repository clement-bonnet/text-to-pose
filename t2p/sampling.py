import torch
from torch.distributions import Categorical, Distribution


def tempered_sampling(
    distribution: Distribution,
    temperature: float,
    softmax_size: int,
    alpha: float = 1.0,
):
    points = distribution.sample(torch.Size((softmax_size,)))
    log_pdfs = distribution.log_prob(points)
    if temperature == 0:
        index = log_pdfs.argmax(0)
    else:
        weights = alpha * (1 / temperature - 1) * log_pdfs
        index = Categorical(logits=weights.T).sample()
    sample = torch.take_along_dim(points, index[None, :, None], dim=0).squeeze(0)
    return sample
