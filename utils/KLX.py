import torch
import matplotlib.pyplot as plt

"""
Evaluate the D_stsp to compare the geometric overlap between 
the replicated data and the ture data
"""

def klx_metric(x_gen, x_target, n_bins=30):
    p_gen, p_target = get_pdf_from_timeseries(x_gen, x_target, n_bins)
    return KL_divergence(p_target, p_gen) 

def get_pdf_from_timeseries(x_gen, x_target, n_bins):
    """
    Calculate spatial pdf of time series x1 and x2
    """
    min_, max_ = get_min_max_range(x_target)
    hist_gen = calc_histogram(x_gen, n_bins=n_bins, min_=min_, max_=max_)
    hist_target = calc_histogram(x_target, n_bins=n_bins, min_=min_, max_=max_)

    p_gen = normalize_to_pdf_with_laplace_smoothing(histogram=hist_gen, n_bins=n_bins)
    p_target = normalize_to_pdf_with_laplace_smoothing(histogram=hist_target, n_bins=n_bins)
    return p_gen, p_target

def get_min_max_range(x_target):
    min_ = -2 * x_target.std(0)
    max_ = 2 * x_target.std(0)
    return min_, max_

def calc_histogram(x, n_bins, min_, max_):
    """
    Calculate a multidimensional histogram in the range of min and max
    """
    dim_x = x.shape[1]  # number of dimensions
    device = x.device

    coordinates = torch.LongTensor(x.shape).to(device)
    for dim in range(dim_x):
        span = max_[dim] - min_[dim]
        xd = (x[:, dim] - min_[dim]) / span
        xd = xd * n_bins
        xd = xd.long()
        coordinates[:, dim] = xd

    # discard outliers
    coord_bigger_zero = coordinates > 0
    coord_smaller_nbins = coordinates < n_bins
    inlier = coord_bigger_zero.all(1) * coord_smaller_nbins.all(1)
    coordinates = coordinates[inlier]

    size_ = tuple(n_bins for _ in range(dim_x))
    histogram = torch.sparse.FloatTensor(coordinates.t(), torch.ones(coordinates.shape[0]), size=size_).to_dense()
    return histogram

def normalize_to_pdf_with_laplace_smoothing(histogram, n_bins, smoothing_alpha=10e-6):
    if histogram.sum() == 0:  # if no entries in the range
        pdf = None
    else:
        dim_x = len(histogram.shape)
        pdf = (histogram + smoothing_alpha) / (histogram.sum() + smoothing_alpha * n_bins ** dim_x)
    return pdf

def KL_divergence(p1, p2):
    """
    Calculate Kullback-Leibler divergence
    """
    if p1 is None or p2 is None:
        kl = float('nan')
    else:
        kl = (p1 * torch.log(p1 / p2)).sum()
    return kl