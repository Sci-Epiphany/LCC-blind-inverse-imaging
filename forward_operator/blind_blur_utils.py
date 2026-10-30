import math

import torch
import torch.fft as fft


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    return otf

def o2p(otf, shape):
    shape_otf = otf.shape[-2:]
    psf = otf
    for axis, axis_size in enumerate(shape):
        psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
    return psf[:,:,:shape[0], :shape[1]]


def fft_blur(img, k):
    k = p2o(k, img.shape[-2:])
    img_fft = fft.fft2(img)
    ker_fft = fft.fft2(k)

    res_fft = ker_fft.mul(img_fft)
    res = fft.ifft2(res_fft)
    return res.real

def normalize_kernel(k):
    _, _, h, w = k.shape
    k = torch.maximum(torch.zeros_like(k), k)
    k_sum = k.sum(axis=(-2, -1))[:, :, None, None]
    k_sum[k_sum == 0] = 1  # 避免除以零
    k /= k_sum.repeat(1, 1, h, w)
    k[torch.isnan(k)] = 0  # 明确替换 NaN 值为零
    return k


def data_solution_deblur(z, x, y, alpha):
    Fx = fft.fft2(x)
    FxC = Fx.conj()
    Fx_sumC = FxC.sum(axis=0)[None]
    F2x = Fx.mul(FxC)
    F2x_sum = F2x.sum(axis=0)[None]
    Fx_sumFy = Fx_sumC.mul(fft.fft2(y))
    res_fft = (Fx_sumFy + fft.fft2(alpha * z)) / (F2x_sum + alpha)
    return fft.ifft2(res_fft).real

def HQS_ker(y, x, sigma, init_ker, iters=10, ksize=64):
    lamb = 1
    beta = 1e5
    alpha = sigma ** 2 * beta
    z0 = init_ker

    for i in range(iters):
        k0 = data_solution_deblur(p2o(z0, y.shape[-2:]), x, y, alpha).mean(axis=1)[:, None, :, :]

        k0 = o2p(k0, (ksize, ksize))
        z0 = normalize_kernel(k0)

        res = (beta * z0) / (2 * lamb + beta)
        # sigma_d = torch.ones_like(kernel)[:, :1, :, :] * (math.sqrt(lamb / beta))
        # with torch.no_grad():
        #     res = denoiser(torch.cat((kernel, sigma_d), axis=1))

        z0 = normalize_kernel(res)

    return z0
