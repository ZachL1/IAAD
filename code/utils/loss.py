import torch

def emd_loss(p_target:torch.Tensor, p_estimate:torch.Tensor, r=2) -> torch.Tensor:
    """
    Earth Mover's Distance on a batch

    Args:
        p_target: true distribution of shape mini_batch_size × num_classes × 1
        p_estimate: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p_target.shape == p_estimate.shape
    cdf_target = torch.cumsum(p_target, dim=1)
    cdf_estimate = torch.cumsum(p_estimate, dim=1)

    cdf_diff = cdf_estimate - cdf_target
    samplewise_emd = torch.pow(torch.mean(torch.pow(torch.abs(cdf_diff), r), dim=1), 1.0/r)

    return samplewise_emd.mean()