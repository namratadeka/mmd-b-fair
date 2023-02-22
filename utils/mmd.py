import torch
import numpy as np

from utils.utils import zero_diag


def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0

    return Pdist

def gaussian_kernel(X, Y, X_org, Y_org, sigma_phi, sigma_q, epsilon=1e-10, is_smooth=True):
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)

    if is_smooth:
        L = 1 # generalized Gaussian (if L>1)
        Dxx_org = Pdist2(X_org, X_org)
        Dyy_org = Pdist2(Y_org, Y_org)
        Dxy_org = Pdist2(X_org, Y_org)
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma_phi)**L -Dxx_org / sigma_q) + epsilon * torch.exp(-Dxx_org / sigma_q)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma_phi)**L -Dyy_org / sigma_q) + epsilon * torch.exp(-Dyy_org / sigma_q)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma_phi)**L -Dxy_org / sigma_q) + epsilon * torch.exp(-Dxy_org / sigma_q)
    else:
        Kx = torch.exp(-Dxx / sigma_phi)
        Ky = torch.exp(-Dyy / sigma_phi)
        Kxy = torch.exp(-Dxy / sigma_phi)
    return Kx, Ky, Kxy

def mmd_variance(Kx, Ky, Kxy, nx, ny, m, unbiased=False):
    if unbiased:
        # From https://arxiv.org/abs/1906.02104, Eq.(4) for m!=n:
        n = nx
        Kx_ = zero_diag(Kx)
        Ky_ = zero_diag(Ky)
        Ind = torch.ones((n,1)).to(Kx.device)
        
        xdown2 = lambda x: x*(x-1)
        xdown3 = lambda x: xdown2(x)*(x-2)
        xdown4 = lambda x: xdown3(x)*(x-3)
        
        c1 = 4*(m*n + m - 2*n) / (xdown2(m) * xdown4(n))
        c2 = -2*(2*m - n) / (xdown2(m)*n*(n-2)*(n-3))
        c3 = 4*(m*n + m - 2*n - 1) / (xdown2(m) * n**2 * (n-1)**2)
        c4 = -4*(2*m - n - 2) / (xdown2(m)*n*(n-1)**2)
        c5 = -2*(2*m - 3) / (xdown2(m)*xdown4(n))
        c6 = -4*(2*m - 3) / (xdown2(m) * n**2 * (n-1)**2)
        c7 = -8 / (m*xdown3(n))
        c8 = 8 / (m*n*xdown3(n))
        var_est = (
            c1 * (torch.linalg.norm(Kx_ @ Ind, dim=(-2,-1))**2 + torch.linalg.norm(Ky_ @ Ind, dim=(-2,-1))**2)
            + c2 * (torch.linalg.norm(Kx_, dim=(-2,-1))**2 + torch.linalg.norm(Ky_, dim=(-2,-1))**2)
            + c3 * (torch.linalg.norm(Kxy @ Ind, dim=(-2,-1))**2 + torch.linalg.norm(Kxy.permute(0,-1,-2) @ Ind, dim=(-2,-1))**2)
            + c4 * torch.linalg.norm(Kxy, dim=(-2,-1))**2
            + c5 * ((Ind.transpose(1,0) @ Kx_ @ Ind)**2 + (Ind.transpose(1,0) @ Ky_ @ Ind)**2).squeeze()
            + c6 * ((Ind.transpose(1,0) @ Kxy @ Ind)**2).squeeze()
            + c7 * ((Ind.transpose(1,0) @ Kx_ @ Kxy @ Ind) + (Ind.transpose(1,0) @ Ky_ @ Kxy.permute(0,-1,-2) @ Ind)).squeeze()
            + c8 * (((Ind.transpose(1,0) @ Kx_ @ Ind) + (Ind.transpose(1,0) @ Ky_ @ Ind)) * (Ind.transpose(1,0) @ Kxy @ Ind)).squeeze()
        )
    else:
        hh = Kx + Ky - Kxy - Kxy.permute(0, -1, -2)
        v1 = torch.linalg.norm(hh.sum(-1)/ny, dim=(-1))**2 / ny
        v2 = (hh.sum(dim=(-2,-1)) / nx) / nx
        var_est = 4 * (v1 - v2**2)
        
    return var_est

def h1_mean_var_gram(Kx, Ky, Kxy, m, is_var_computed, use_1sample_U=True, unbiased_variance=True):
    """compute value of MMD and variance of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx, Kxy), -1)
    Kyxy = torch.cat((Kxy.permute(0, 2, 1), Ky), -1)
    Kxyxy = torch.cat((Kxxy, Kyxy), 1)
    nx = Kx.shape[1]
    ny = Ky.shape[1]

    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx, dim=(-2, -1)) - torch.sum(torch.diagonal(Kx, dim1=-2, dim2=-1), dim=-1)), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky, dim=(-2, -1)) - torch.sum(torch.diagonal(Ky, dim1=-2, dim2=-1), dim=-1)), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy, dim=(-2, -1)) - torch.sum(torch.diagonal(Kxy, dim1=-2, dim2=-1), dim=-1)), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy, dim=(-2, -1)), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx, dim=(-2, -1))), (nx * nx))
        yy = torch.div((torch.sum(Ky, dim=(-2, -1))), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy, dim=(-2, -1))), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy, dim=(-2, -1)), (nx * ny))
        mmd2 = xx - 2 * xy + yy

    if not is_var_computed:
        return mmd2, None, Kxyxy
        
    var_est = mmd_variance(Kx, Ky, Kxy, nx, ny, m, unbiased=unbiased_variance)
    
    return mmd2, var_est, Kxyxy

def MMDu(features, n_samples, n_population, images, sigma_phi=0.1, sigma_q=0.1, epsilon = 10**(-10), is_smooth=True, 
        is_var_computed=True, use_1sample_U=True, kernel='gaussian', unbiased_variance=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = features[0:n_samples, :] # fetch the sample 1 (features of deep networks)
    Y = features[n_samples:, :] # fetch the sample 2 (features of deep networks)
    X_org = images[0:n_samples, :] # fetch the original sample 1
    Y_org = images[n_samples:, :] # fetch the original sample 2

    if kernel.__eq__('gaussian'):
        Kx, Ky, Kxy = gaussian_kernel(X, Y, X_org, Y_org, sigma_phi, sigma_q, epsilon, is_smooth)

    return h1_mean_var_gram(Kx, Ky, Kxy, n_population, is_var_computed, use_1sample_U, unbiased_variance)

def permutation_test(mmd_value, Kxyxy, n_per, m, n, alpha):
    mmd_vector = torch.zeros(n_per).to(mmd_value.device)
    count = 0
    nxy = Kxyxy.shape[0]
    nx = nxy // 2
    for r in range(n_per):
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)].unsqueeze(0)
        Ky = Kxyxy[np.ix_(indy, indy)].unsqueeze(0)
        Kxy = Kxyxy[np.ix_(indx, indy)].unsqueeze(0)

        tmp = h1_mean_var_gram(Kx, Ky, Kxy, m, is_var_computed=False)
        mmd_vector[r] = tmp[0][0]
        if mmd_vector[r] > mmd_value:
            count += 1
        if count > np.ceil(n_per * alpha):
            h = 0
        else:
            h = 1

    S_mmd_vector = torch.sort(mmd_vector*n).values
    threshold = S_mmd_vector[np.int(np.ceil(n_per * (1 - alpha)))]

    return h, threshold
    
def test_power(mmdu, threshold, mmd_std, M, normal_dist):
    cdf_arg = (mmdu - threshold/M) / mmd_std
    power = normal_dist.cdf(cdf_arg)

    return power, cdf_arg

def block_approx_test_power(mmd2, mmd_std, n_blocks, threshold, normal_dist):
    cdf_arg = ((np.sqrt(n_blocks) * mmd2) / mmd_std) - threshold
    power = normal_dist.cdf(cdf_arg)
    return power, cdf_arg

def strong_test(features, n, m, images, sigma_phi, unbiased_variance, kernel='gaussian', is_smooth=False, n_per=100, alpha=0.05):
    mmd_value, mmd_var, Kxyxy = MMDu(features=features,
                                    n_samples=n, n_population=m, 
                                    images=images, 
                                    sigma_phi=sigma_phi,
                                    kernel=kernel,
                                    is_smooth=is_smooth,
                                    unbiased_variance=unbiased_variance)
    h, threshold = permutation_test(mmd_value[0], Kxyxy[0], n_per, m, n, alpha)
    return h
