import tqdm
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt

from cores.trajectory import Trajectory
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE
from cores.mcmc import MCMCSampler
from cores.mcmc_nl import MCMCNLSampler
from forward_operator import LatentWrapper
from model.network_unet import UNetRes  as net_kernel
from forward_operator.blind_blur_utils import normalize_kernel
import cores.utils_agem as agem
# from scripts.DAPS_Demo import measurement


def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        return LatentDAPS(**kwargs)
    nonlinear = kwargs['nonlinear']
    kwargs.pop('nonlinear')
    if nonlinear:
        kwargs.pop('blind')
        return Nonlinear_DAPS(**kwargs)
    blind = kwargs['blind']
    kwargs.pop('blind')
    if blind:
        return BlindDAPS(**kwargs)

    return DAPS(**kwargs)

class DAPS(nn.Module):
    """
    Decoupled Annealing Posterior Sampling (DAPS) implementation.

    Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using the DAPS method.

        Args:
            model (nn.Module): Diffusion model.
            x_start (torch.Tensor): Initial tensor/state.
            operator (nn.Module): Measurement operator.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled tensor/state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        xt = x_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                x0hat = sampler.sample(xt)

            # 2. MCMC update
            x0y = self.mcmc_sampler.sample(xt, model, x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                xt = x0y

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        return xt
    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """Records the intermediate states during sampling."""

        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""

        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion or latent diffusion model.

        Returns:
            torch.Tensor: Random initial tensor.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
        return x_start


class BlindDAPS(nn.Module):
    """
    Decoupled Annealing Posterior Sampling (DAPS) implementation.
    Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)

    def sample(self, model_dict, x_start, operator, measurement=None, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using the DAPS method with dictionary inputs for parallel processing.

        Args:
            model_dict (dict): Dictionary containing models for different components (e.g., 'image', 'kernel').
                               Example: {'image': image_model, 'kernel': kernel_model}.
            operator (nn.Module): Measurement operator to generate measurement y from image and kernel.
            measurement (torch.Tensor, optional): Observed measurement tensor (y). If None, it will be generated.
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (dict, optional): Ground truth data for evaluation. Keys should match model_dict keys.

        Returns:
            dict: Final sampled tensors for each component (e.g., 'image', 'kernel').
        """
        if record:
            self.trajectory = {key: Trajectory() for key in model_dict.keys()}
        # Generate measurement if not provided
        if measurement is None:
            print('No measurement is provided.')
            measurement = self._generate_measurement(x_start['image'], x_start['kernel'], operator)

        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(
            self.annealing_scheduler.num_steps - 1)
        xt = x_start

        reg_path = 'checkpoints/kernel_denoiser_64.pth'
        kernel_denoiser = self.load_network(net_kernel(in_nc=2, out_nc=1, nc=[8, 16, 32, 64], nb=2, bias=False), reg_path, device='cuda')
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]

            # Parallel reverse diffusion for all components
            x0hat = {}
            for key, model in model_dict.items():
                with torch.no_grad():
                    diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                    sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                    x0hat[key] = sampler.sample(xt[key])


            # MCMC update for all components
            x0y = self.mcmc_sampler.sample(
                    xt, model_dict, x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps
                , kernel_denoiser)

            # Forward diffusion for all components
            for key in model_dict.keys():
                if step != self.annealing_scheduler.num_steps - 1:
                    xt[key] = x0y[key] + torch.randn_like(x0y[key]) * self.annealing_scheduler.sigma_steps[step + 1]
                else:
                    xt[key] = x0y[key]

            kernel_tmp = ((x0y['kernel']+ 1) / 2).clamp(min=0, max=1)
            x0y['kernel'] =  kernel_tmp / kernel_tmp.sum((-2, -1), keepdim=True)
            difference = measurement - operator(x0y)
            loss = torch.linalg.norm(difference) + torch.linalg.norm(x0y['kernel'].view(-1), ord=1)


            with open('./loss_record.txt', 'a') as f:
                f.write(f'{loss}\n')
                f.flush()

            # Evaluation
            results = {}
            if evaluator and 'gt' in kwargs:
                gt = kwargs['gt']
                for key in model_dict.keys():
                    with torch.no_grad():
                        if key == 'kernel':
                            results[key] = evaluator(gt[key], measurement, normalize_kernel(x0y[key] * 0.5 + 0.5))
                        else:
                            results[key] = evaluator(gt[key], measurement, x0y[key])

                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    postfix = {
                        f'{key}_{main_eval_fn_name}': f"{results[key][main_eval_fn_name].item():.2f}"
                        for key in model_dict.keys()
                    }
                    pbar.set_postfix(postfix)

            # Record trajectory
            if record:
                for key in model_dict.keys():
                    self._record(key, xt[key], x0y[key], x0hat[key], sigma, results.get(key, {}))




        return {'image':xt['image'], 'kernel':xt['kernel']}

    def _generate_measurement(self, image, kernel, operator):
        """
        Generates the measurement y by applying the operator to image and kernel.

        Args:
            image (torch.Tensor): Image tensor.
            kernel (torch.Tensor): Kernel tensor.
            operator (nn.Module): Measurement operator.

        Returns:
            torch.Tensor: Generated measurement (y).
        """
        return operator(image, kernel)

    def _record(self, key, xt, x0y, x0hat, sigma, results):
        """Records the intermediate states during sampling for a specific component."""
        self.trajectory[key].add_tensor(f'xt', xt)
        self.trajectory[key].add_tensor(f'x0y', x0y)
        self.trajectory[key].add_tensor(f'x0hat', x0hat)
        self.trajectory[key].add_value(f'sigma', sigma)
        for name in results.keys():
            self.trajectory[key].add_value(f'x0hat_{name}', results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, batch_size, model_dict):
        """
        Generates initial random state tensors from the Gaussian prior for all components.

        Args:
            batch_size (int): Number of initial states to generate.
            model_dict (dict): Dictionary containing models for different components.

        Returns:
            dict: Random initial tensors for each component.
        """
        device = next(next(iter(model_dict.values())).parameters()).device
        x_start = {
            key: torch.randn(batch_size, *model.get_in_shape(), device=device) *
                 self.annealing_scheduler.get_prior_sigma()
            for key, model in model_dict.items()
        }
        return x_start

    def load_network(self, net, path, device='cuda'):
        """
        加载去噪网络模型。

        该函数将给定的网络模型加载到指定的设备上。它首先通过提供的路径加载模型的预训练权重，
        然后将模型转移到指定的设备（默认为CUDA）。

        参数:
        - net: 待加载权重的网络模型对象。
        - path: 字符串，预训练模型权重文件的路径。
        - device: 字符串，指定模型运行的设备，默认为'cuda'。另一个常见值可以是'cpu'。

        返回:
        - net: 加载完权重并转移到指定设备上的网络模型对象。
        """
        net.load_state_dict(torch.load(path))

        # 将模型转移到指定的设备上
        return net.to(device)

# class BlindDAPS(nn.Module):
#     """
#     Decoupled Annealing Posterior Sampling (DAPS) implementation.
#
#     Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
#     """
#
#     def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config):
#         """
#         Initializes the DAPS sampler with the provided scheduler and sampler configurations.
#
#         Args:
#             annealing_scheduler_config (dict): Configuration for annealing scheduler.
#             diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
#             mcmc_sampler_config (dict): Configuration for MCMC sampler.
#         """
#         super().__init__()
#         annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
#                                                                              diffusion_scheduler_config)
#         self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
#         self.diffusion_scheduler_config = diffusion_scheduler_config
#         self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)
#         self.beta = mcmc_sampler_config.reg_terms['beta']
#
#     def sample(self, model_dict, x_start, operator, measurement=None, evaluator=None, record=False, verbose=False, **kwargs):
#         """
#         Performs sampling using the DAPS method with dictionary inputs for parallel processing.
#
#         Args:
#             model_dict (dict): Dictionary containing models for different components (e.g., 'image', 'kernel').
#                                Example: {'image': image_model, 'kernel': kernel_model}.
#             operator (nn.Module): Measurement operator to generate measurement y from image and kernel.
#             measurement (torch.Tensor, optional): Observed measurement tensor (y). If None, it will be generated.
#             evaluator (Evaluator, optional): Evaluator for performance metrics.
#             record (bool, optional): If True, records the sampling trajectory.
#             verbose (bool, optional): Enables progress bar and logs.
#             **kwargs:
#                 gt (dict, optional): Ground truth data for evaluation. Keys should match model_dict keys.
#
#         Returns:
#             dict: Final sampled tensors for each component (e.g., 'image', 'kernel').
#         """
#         if record:
#             self.trajectory = {key: Trajectory() for key in model_dict.keys()}
#         # Generate measurement if not provided
#         if measurement is None:
#             measurement = self._generate_measurement(x_start['image'], x_start['kernel'], operator)
#
#         pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(
#             self.annealing_scheduler.num_steps - 1)
#         xt = x_start
#
#         reg_path = 'checkpoints/kernel_denoiser_64.pth'
#         kernel_denoiser = self.load_network(net_kernel(in_nc=2, out_nc=1, nc=[8, 16, 32, 64], nb=2, bias=False), reg_path, device='cuda')
#
#         for step in pbar:
#             sigma = self.annealing_scheduler.sigma_steps[step]
#
#             # Parallel reverse diffusion for all components
#             x0hat = {}
#             for key, model in model_dict.items():
#                 with torch.no_grad():
#                     if key == 'image':
#                         diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
#                         sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
#                         x0hat[key] = sampler.sample(xt[key])
#                     elif key == 'kernel':
#                         if step == 0:
#                            x0hat[key] = xt[key] * 2 - 1
#                         else:
#                             alpha = sigma ** 2 * self.beta
#                             z0 = agem.normalize_kernel(xt[key] * 0.5 + 0.5)
#                             y = ((measurement + 1)/2).clamp(0, 1)
#                             x0 = ((x0hat['image']+ 1)/2).clamp(0, 1)
#                             k0 = agem.data_solution_deblur(agem.p2o(z0,y.shape[-2:]), x0, y, alpha).mean(axis=1)[:,None,:,:]
#                             k0 = agem.o2p(k0, (64, 64))
#                             x0hat[key] = k0 * 2 -1
#
#
#
#             # MCMC update for all components
#             x0y = self.mcmc_sampler.sample(
#                     xt, model_dict, x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps
#                 , kernel_denoiser)
#
#
#             # Forward diffusion for all components
#             for key in model_dict.keys():
#                 if step != self.annealing_scheduler.num_steps - 1 and key != 'kernel':
#                     xt[key] = x0y[key] + torch.randn_like(x0y[key]) * self.annealing_scheduler.sigma_steps[step + 1]
#                 else:
#                     xt[key] = x0y[key]
#
#             # Evaluation
#             # print(torch.max(xt['kernel']))
#             results = {}
#             if evaluator and 'gt' in kwargs:
#                 gt = kwargs['gt']
#                 for key in model_dict.keys():
#                     with torch.no_grad():
#                         if key == 'kernel':
#                             results[key] = evaluator(gt[key], measurement, normalize_kernel(x0y[key]))
#                         else:
#                             results[key] = evaluator(gt[key], measurement, x0y[key])
#
#                 if verbose:
#                     main_eval_fn_name = evaluator.main_eval_fn_name
#                     postfix = {
#                         f'{key}_{main_eval_fn_name}': f"{results[key][main_eval_fn_name].item():.2f}"
#                         for key in model_dict.keys()
#                     }
#                     pbar.set_postfix(postfix)
#
#             # Record trajectory
#             if record:
#                 for key in model_dict.keys():
#                     self._record(key, xt[key], x0y[key], x0hat[key], sigma, results.get(key, {}))
#
#
#
#
#         return {'image':xt['image'], 'kernel':xt['kernel']}
#
#     def _generate_measurement(self, image, kernel, operator):
#         """
#         Generates the measurement y by applying the operator to image and kernel.
#
#         Args:
#             image (torch.Tensor): Image tensor.
#             kernel (torch.Tensor): Kernel tensor.
#             operator (nn.Module): Measurement operator.
#
#         Returns:
#             torch.Tensor: Generated measurement (y).
#         """
#         return operator(image, kernel)
#
#     def _record(self, key, xt, x0y, x0hat, sigma, results):
#         """Records the intermediate states during sampling for a specific component."""
#         self.trajectory[key].add_tensor(f'xt', xt)
#         self.trajectory[key].add_tensor(f'x0y', x0y)
#         self.trajectory[key].add_tensor(f'x0hat', x0hat)
#         self.trajectory[key].add_value(f'sigma', sigma)
#         for name in results.keys():
#             self.trajectory[key].add_value(f'x0hat_{name}', results[name])
#
#     def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
#         """Checks and updates the configurations for the schedulers."""
#         if 'sigma_max' in diffusion_scheduler_config:
#             diffusion_scheduler_config.pop('sigma_max')
#
#         return annealing_scheduler_config, diffusion_scheduler_config
#
#     def get_start(self, batch_size, model_dict):
#         """
#         Generates initial random state tensors from the Gaussian prior for all components.
#
#         Args:
#             batch_size (int): Number of initial states to generate.
#             model_dict (dict): Dictionary containing models for different components.
#
#         Returns:
#             dict: Random initial tensors for each component.
#         """
#         device = next(next(iter(model_dict.values())).parameters()).device
#         x_start = {
#                         key: torch.randn(batch_size, *model.get_in_shape(), device=device) *
#                              self.annealing_scheduler.get_prior_sigma()
#                         for key, model in model_dict.items()
#                     }
#         x_start.update({'kernel': self.init_ker(batch_size=batch_size, ksize=64,device=device)})
#         return x_start
#
#     def init_ker(self, batch_size, ksize=33, device='cuda'):
#         k = torch.zeros((batch_size, 1, ksize, ksize)).to(device)
#         k[:, 0, ksize // 2, ksize // 2] = 1
#         return k
#
#     def load_network(self, net, path, device='cuda'):
#         """
#         加载去噪网络模型。
#
#         该函数将给定的网络模型加载到指定的设备上。它首先通过提供的路径加载模型的预训练权重，
#         然后将模型转移到指定的设备（默认为CUDA）。
#
#         参数:
#         - net: 待加载权重的网络模型对象。
#         - path: 字符串，预训练模型权重文件的路径。
#         - device: 字符串，指定模型运行的设备，默认为'cuda'。另一个常见值可以是'cpu'。
#
#         返回:
#         - net: 加载完权重并转移到指定设备上的网络模型对象。
#         """
#         net.load_state_dict(torch.load(path))
#
#         # 将模型转移到指定的设备上
#         return net.to(device)

class Nonlinear_DAPS(nn.Module):
    """
    Decoupled Annealing Posterior Sampling (DAPS) for Nonlinear-Blind Deblurring implementation.

    Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCNLSampler(**mcmc_sampler_config)

    def sample(self, model_dict, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using the DAPS method.

        Args:
            model (nn.Module): Diffusion model.
            x_start (dict[torch.Tensor]): Initial tensor/state.
            operator (nn.Module): Measurement operator.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled tensor/state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        xt = x_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion and kernel solution
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model_dict, diffusion_scheduler, solver='euler')
                x0_hat = sampler.sample(xt['image'])

            k0hat = operator.call_solution(x0_hat, measurement)

            x0hat = {'image': x0_hat, 'kernel': k0hat}

            # 2. MCMC update
            x0y = self.mcmc_sampler.sample(xt, model_dict, x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps)

            final_step = step == self.annealing_scheduler.num_steps - 2

            # 3. forward diffusion
            for key in xt.keys():
                if final_step and self.mcmc_sampler.return_clean:
                    xt[key] = x0y[key]
                elif key != 'kernel':
                    xt[key] = x0y[key] + torch.randn_like(x0y[key]) * self.annealing_scheduler.sigma_steps[step + 1]
                elif key == 'kernel':
                    kernel_base = (
                        x0y[key]
                        if self.mcmc_sampler.kernel_trajectory == 'mcmc'
                        else x0hat[key]
                    )
                    xt[key] = kernel_base + torch.randn_like(x0y[key]) * self.annealing_scheduler.sigma_steps[step + 1]
                else:
                    xt[key] = x0y[key]

            # 4. evaluation
            results = {}
            if evaluator and 'gt' in kwargs:
                gt = kwargs['gt']
                # for key in xt.keys():
                with torch.no_grad():
                  results['image'] = evaluator(gt['image'], measurement, x0y['image'])

                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    postfix = {
                        'x0' + main_eval_fn_name: f"{results['image'][main_eval_fn_name].item():.2f}"
                    }
                    pbar.set_postfix(postfix)

            # Record trajectory
            if record:
                for key in xt.keys():
                    self._record(key, xt[key], x0y[key], x0hat[key], sigma, results.get(key, {}))
        return xt

    def _record(self, key, xt, x0y, x0hat, sigma, results):
        """Records the intermediate states during sampling for a specific component."""
        self.trajectory[key].add_tensor(f'xt', xt)
        self.trajectory[key].add_tensor(f'x0y', x0y)
        self.trajectory[key].add_tensor(f'x0hat', x0hat)
        self.trajectory[key].add_value(f'sigma', sigma)
        for name in results.keys():
            self.trajectory[key].add_value(f'x0hat_{name}', results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""

        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion or latent diffusion model.

        Returns:
            Dict: Random initial tensors.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
        k_start = torch.randn(batch_size, 512, 2, 2, device=device) * 1.2     # Keep the noise scale unchange
        return {'image':x_start, 'kernel':k_start}




class LatentDAPS(DAPS):
    """
    Latent Decoupled Annealing Posterior Sampling (LatentDAPS).

    Implements posterior sampling using a latent diffusion model combined with MCMC updates
    """
    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using LatentDAPS in latent space, decoding intermediate results.

        Args:
            model (LatentDiffusionModel): Latent diffusion model.
            z_start (torch.Tensor): Initial latent state tensor.
            operator (nn.Module): Measurement operator applied in data space.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for monitoring performance.
            record (bool, optional): Whether to record intermediate states and metrics.
            verbose (bool, optional): Enables progress bar and evaluation metrics.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled data decoded from latent space.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        warpped_operator = LatentWrapper(operator, model)

        zt = z_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                z0hat = sampler.sample(zt)
                x0hat = model.decode(z0hat)

            # 2. MCMC update
            z0y = self.mcmc_sampler.sample(zt, model, z0hat, warpped_operator, measurement, sigma, step / self.annealing_scheduler.num_steps)
            with torch.no_grad():
                x0y = model.decode(z0y)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                zt = z0y + torch.randn_like(z0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                zt = z0y
            with torch.no_grad():
                xt = model.decode(zt)

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        return xt
