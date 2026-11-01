import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from filterpy.examples.bearing_only import xs_scaled
from torch import optim

from forward_operator.blind_blur_utils import normalize_kernel, HQS_ker

from .trajectory import Trajectory


# class MCMCSampler(nn.Module):
#     """
#     Monte Carlo sampler class for diffusion processes.
#
#     Supports Langevin dynamics, Hamiltonian Monte Carlo (HMC) and Metropolis-Hastings (MH) methods.
#
#     Attributes:
#         num_steps (int): Number of sampling steps.
#         lr (float): Initial learning rate.
#         tau (float): Standard deviation for data-fitting term.
#         lr_min_ratio (float): Minimum learning rate ratio.
#         prior_solver (str): Method to compute prior score ('gaussian', 'exact', 'score-t', 'score-min').
#         prior_sigma_min (float): Minimum sigma for prior computation.
#         mc_algo (str): Monte Carlo algorithm ('langevin', 'mh', or 'hmc').
#         momentum (float): Momentum coefficient for HMC.
#     """
#
#     def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01, prior_solver='gaussian', prior_sigma_min=1e-2,
#                  mc_algo='langevin', momentum=0.9):
#         super().__init__()
#         self.num_steps = num_steps
#         self.lr = lr
#         self.tau = tau
#         self.lr_min_ratio = lr_min_ratio
#         self.prior_solver = prior_solver
#         self.prior_sigma_min = prior_sigma_min
#         self.mc_algo = mc_algo
#         self.momentum = momentum
#
#     def score_fn(self, x, x0hat, model, xt, operator, measurement, sigma):
#         """
#         Computes the conditional score function \nabla_x \log p(x_0 = x | x_t, y).
#
#         Returns:
#             Tuple containing:
#                 - Current score estimate.
#                 - Data-fitting loss.
#         """
#         data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
#         data_term = -data_fitting_grad / self.tau ** 2
#         xt_term = (xt - x) / sigma ** 2
#         prior_term = self.get_prior_score(x, x0hat, xt, model, sigma)
#         return data_term + xt_term + prior_term, data_fitting_loss
#
#     def get_prior_score(self, x, x0hat, xt, model, sigma):
#         if self.prior_solver == 'score-min' or self.prior_solver == 'score-t' or self.prior_solver == 'gaussian':
#             prior_score = self.prior_score
#         elif self.prior_solver == 'exact':
#             prior_score = model.score(x, torch.tensor(self.prior_sigma_min).to(x.device)).detach()
#         else:
#             raise NotImplementedError
#         return prior_score
#
#     def prepare_prior_score(self, x0hat, xt, model, sigma):
#         """
#         Precomputes the prior score based on the specified solver method.
#         """
#         if self.prior_solver == 'score-min':
#             self.prior_score = model.score(x0hat, self.prior_sigma_min).detach()
#
#         elif self.prior_solver == 'score-t':
#             self.prior_score = model.score(xt, sigma).detach()
#
#         elif self.prior_solver == 'gaussian':
#             self.prior_score = (x0hat - xt).detach() / sigma ** 2
#
#         elif self.prior_solver == 'exact':
#             pass
#
#         else:
#             raise NotImplementedError
#
#     def mc_prepare(self, x0hat, xt, model, operator, measurement, sigma):
#         """Prepares the sampler state before starting Monte Carlo sampling."""
#         if self.mc_algo == 'hmc':
#             self.velocity = torch.randn_like(x0hat)
#
#     def mc_update(self, x, cur_score, lr, epsilon):
#         """ Performs a single Monte Carlo update step (Langevin or HMC)."""
#         if self.mc_algo == 'langevin':
#             x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
#         elif self.mc_algo == 'hmc':  # (damping) hamiltonian monte carlo
#             step_size = np.sqrt(lr)
#             self.velocity = self.momentum * self.velocity + step_size * cur_score + np.sqrt(2 * (1 - self.momentum)) * epsilon
#             x_new = x + self.velocity * step_size
#         else:
#             raise NotImplementedError
#         return x_new
#
#     def sample_mh(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
#         if record:
#             self.trajectory = Trajectory()
#
#         lr = self.get_lr(ratio)
#         x = x0hat.clone().detach()
#         pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
#         for _ in pbar:
#             # Gaussain proposal
#             x_new = x + torch.randn_like(x) * np.sqrt(lr)
#             # compute acceptance ratio
#             loss_new = operator.loss(x_new, measurement)
#             loss_old = operator.loss(x, measurement)
#             log_data_ratio = (loss_old - loss_new) / (2 * self.tau ** 2)
#             # compute prior p(x_0 | x_t)
#             prior_loss_new = (x_new - x0hat).pow(2).flatten(1).sum(-1)
#             prior_loss_old = (x - x0hat).pow(2).flatten(1).sum(-1)
#             log_prior_ratio = (prior_loss_old - prior_loss_new) / (2 * sigma ** 2)
#             # compute acceptance probability
#             log_accept_prob= log_data_ratio + log_prior_ratio
#             accept = torch.rand_like(log_accept_prob).log() < log_accept_prob
#             accept = accept.view(-1, *[1] * len(x.shape[1:]))
#             # update: accept new sample
#             x = torch.where(accept, x_new, x)
#         return x.detach()
#
#     def sample(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
#         """
#         Main method for performing MCMC sampling.
#
#         Args:
#             xt (torch.Tensor): Current noisy latent tensor.
#             model (nn.Module): Diffusion model providing the score function.
#             x0hat (torch.Tensor): Initial estimate of x0 from PF-ODE.
#             operator (Operator): Measurement operator.
#             measurement (torch.Tensor): Measurement data.
#             sigma (float): Noise scale at current timestep.
#             ratio (float): Ratio to adjust learning rate scheduling.
#             record (bool): Whether to record trajectory.
#             verbose (bool): Verbosity flag.
#
#         Returns:
#             torch.Tensor: Sampled latent tensor.
#         """
#         if self.mc_algo == 'mh':
#             return self.sample_mh(xt, model, x0hat, operator, measurement, sigma, ratio, record, verbose)
#         if record:
#             self.trajectory = Trajectory()
#         lr = self.get_lr(ratio)
#         self.mc_prepare(x0hat, xt, model, operator, measurement, sigma)
#         self.prepare_prior_score(x0hat, xt, model, sigma)
#
#         x = x0hat.clone().detach()
#         pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
#         for _ in pbar:
#             cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma)
#             epsilon = torch.randn_like(x)
#
#             x = self.mc_update(x, cur_score, lr, epsilon)
#
#             # early stopping with NaN
#             if torch.isnan(x).any():
#                 return torch.zeros_like(x)
#
#             # record
#             if record:
#                 self._record(x, epsilon, fitting_loss.sqrt())
#         return x.detach()
#
#     def _record(self, x, epsilon, loss):
#         """
#             Records the intermediate states during sampling.
#         """
#         self.trajectory.add_tensor(f'xi', x)
#         self.trajectory.add_tensor(f'epsilon', epsilon)
#         self.trajectory.add_value(f'loss', loss)
#
#     def get_lr(self, ratio):
#         """
#             Computes the learning rate based on the given ratio.
#         """
#         p = 1
#         multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
#         return multiplier * self.lr
#
#     def summary(self):
#         print('+' * 50)
#         print('MCMC Sampler Summary')
#         print('+' * 50)
#         print(f"Prior Solver    : {self.prior_solver}")
#         print(f"MCMC Algorithm  : {self.mc_algo}")
#         print(f"Num Steps       : {self.num_steps}")
#         print(f"Learning Rate   : {self.lr}")
#         print(f"Tau             : {self.tau}")
#         print(f"LR Min Ratio    : {self.lr_min_ratio}")
#         print('+' * 50)


def regularization_step(k, lamb, beta, denoiser=None, how='l1'):
    _, _, n, p = k.shape
    if how not in {'l0', 'l1', 'l2', 'pnp'}:
        raise ValueError(f"Unsupported kernel regularizer: {how}")
    # k = k / (k.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    if how == 'l0':
        threshold = np.sqrt(lamb / beta)
        mask = (torch.abs(k) <= threshold)
        k[mask] = 0  # hard threshold
        res = k
    if how == 'l1':
        res = k - (lamb / beta)
        res = torch.maximum(torch.zeros_like(res), res)

    elif how == 'l2':
        res = (beta * k) / (2 * lamb + beta)

    elif how == 'pnp':
        if not lamb == 0:
            if denoiser is None:
                raise ValueError("PnP regularization requires a kernel denoiser")
            sigma_d = torch.ones_like(k)[:, :1, :, :] * (np.sqrt(lamb / beta))
            res = denoiser(torch.cat((k, sigma_d), axis=1))
        else:
            res = k
    return res

class MCMCSampler(nn.Module):
    """
    Monte Carlo sampler class for diffusion processes.

    Supports Langevin dynamics, Hamiltonian Monte Carlo (HMC) and Metropolis-Hastings (MH) methods.

    Attributes:
        num_steps (int): Number of sampling steps.
        lr (float): Initial learning rate.
        tau (float): Standard deviation for data-fitting term.
        lr_min_ratio (float): Minimum learning rate ratio.
        prior_solver (str): Method to compute prior score ('gaussian', 'exact', 'score-t', 'score-min').
        prior_sigma_min (float): Minimum sigma for prior computation.
        mc_algo (str): Monte Carlo algorithm ('langevin', 'mh', or 'hmc').
        momentum (float): Momentum coefficient for HMC.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01,
                 prior_solver='gaussian', prior_sigma_min=1e-2,
                 mc_algo='langevin', momentum=0.9, reg_terms='none',
                 kernel_regularizer='pnp', kernel_step_mode='manual',
                 kernel_trust_fraction=1e-2, kernel_line_search_steps=5,
                 kernel_noise_scale=1.0, kernel_data_gradient='raw'):
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
        self.prior_solver = prior_solver
        self.prior_sigma_min = prior_sigma_min
        self.mc_algo = mc_algo
        self.momentum = momentum
        self.lamb = reg_terms['lamb']  # 3e1
        self.beta = reg_terms['beta']  # 8e6
        self.kernel_regularizer = kernel_regularizer
        self.kernel_step_mode = kernel_step_mode
        self.kernel_trust_fraction = kernel_trust_fraction
        self.kernel_line_search_steps = kernel_line_search_steps
        self.kernel_noise_scale = kernel_noise_scale
        self.kernel_data_gradient = kernel_data_gradient

    def score_fn(self, x, x0hat, model_dict, xt, operator, measurement, sigma, kernel_denoiser):
        r"""
        Computes the conditional score function \nabla_x \log p(x_0 = x | x_t, y).

        Args:
            xt (dict): Current noisy latent tensors for each component.
            x0hat (dict): Initial estimates of x0 from PF-ODE for each component.
            model_dict (dict): Dictionary containing models for different components.
            operator (Operator): Measurement operator.
            measurement (torch.Tensor): Measurement data.
            sigma (float): Noise scale at current timestep.w

        Returns:
            Tuple containing:
                - Current score estimate (dict).
                - Data-fitting loss.
        """
        # kernel_tmp = ((x['kernel'] + 1) / 2).clamp(0, 1)
        kernel_positive = F.relu((x['kernel'] + 1) / 2)
        kernel_scale = kernel_positive.sum(
            axis=(-2, -1), keepdim=True
        ).clamp_min(1e-12)
        kernel_tmp = kernel_positive / kernel_scale

        # The measurement standard deviation is known from the forward
        # operator. A null tau therefore removes a redundant hand-tuned scale
        # and ties the likelihood to the physical noise level.
        tau = operator.sigma if self.tau is None else self.tau

        # Compute data-fitting gradient and loss
        data_fitting_grad, data_fitting_loss= operator.gradient(x, measurement, return_loss=True)

        # # --- Step 1: HQS auxiliary variable update ---
        with torch.no_grad():
            res = regularization_step(
                kernel_tmp,
                self.lamb,
                self.beta,
                denoiser=kernel_denoiser,
                how=self.kernel_regularizer,
            )
        # --- Step 2: compute reg gradient from quadratic penalty term ---  k and res have a small gap
        # Pull the quadratic HQS residual back through the simplex
        # normalization and the (k + 1) / 2 latent map. The normalization
        # Jacobian divides by the kernel mass; multiplying by the mass here
        # reverses the chain rule and makes the update strongly scale
        # dependent.
        reg_grad_normalized = -self.beta * (kernel_tmp - res)
        reg_grad_normalized -= (
            reg_grad_normalized * kernel_tmp
        ).sum((-2, -1), keepdim=True)
        reg_grad = (
            0.5 * reg_grad_normalized / kernel_scale
        ) * (kernel_positive > 0).to(reg_grad_normalized.dtype)
        if self.kernel_data_gradient == 'normalized':
            direct_reg_grad = -self.beta * (
                kernel_tmp - res
            ) * kernel_scale
        elif self.kernel_data_gradient == 'raw':
            direct_reg_grad = reg_grad
        else:
            raise ValueError(
                f"Unsupported kernel gradient mode: {self.kernel_data_gradient}"
            )
        self._kernel_prox = res.detach()
        self._kernel_normalized_grad = data_fitting_grad.get(
            'kernel_normalized', data_fitting_grad['kernel']
        )
        # Compute terms for each component
        score_dict = {}

        for key in xt.keys():
            gradient_key = (
                'kernel_normalized'
                if key == 'kernel' and self.kernel_data_gradient == 'normalized'
                else key
            )
            data_term = -data_fitting_grad[gradient_key] / tau ** 2
            xt_term = (xt[key] - x[key]) / sigma ** 2
            prior_term = self.get_prior_score(x[key], x0hat[key], xt[key], model_dict[key], sigma, key)

            if key == 'kernel':
                # Compute kernel huber reg_grad
                score_dict[key] = (
                    data_term + xt_term + prior_term + direct_reg_grad
                )
            elif key == 'image':
                # tv_like = self.hyper_laplace_grad(x[key], p=2/3) * 1e-2
                score_dict[key] = data_term + xt_term + prior_term
            else:
                score_dict[key] = data_term + xt_term + prior_term

        return score_dict, data_fitting_loss

    @staticmethod
    def _normalize_kernel_latent(kernel):
        positive = F.relu((kernel + 1.0) / 2.0)
        return positive / positive.sum(
            (-2, -1), keepdim=True
        ).clamp_min(1e-12)

    @staticmethod
    def _project_simplex(values):
        """Euclidean projection onto the probability simplex per kernel."""
        flat = values.flatten(1)
        ordered, _ = torch.sort(flat, dim=1, descending=True)
        cumulative = ordered.cumsum(dim=1) - 1.0
        indices = torch.arange(
            1, flat.shape[1] + 1, device=flat.device, dtype=flat.dtype
        )
        condition = ordered - cumulative / indices > 0
        rho = condition.sum(dim=1).clamp_min(1)
        theta = (
            cumulative.gather(1, (rho - 1).unsqueeze(1))
            / rho.unsqueeze(1).to(flat.dtype)
        )
        return (flat - theta).clamp_min(0).view_as(values)

    def _kernel_objective(self, kernel, image, x0hat_kernel, measurement,
                          operator, sigma):
        q = self._normalize_kernel_latent(kernel)
        q0 = self._normalize_kernel_latent(x0hat_kernel)
        tau = operator.sigma if self.tau is None else self.tau
        data = operator.loss(
            {'image': image, 'kernel': q}, measurement
        ).sum() / (2.0 * tau ** 2)
        prior = 0.5 * ((q - q0) ** 2).sum() / max(sigma ** 2, 1e-12)
        prox = self._kernel_prox
        regularization = 0.5 * self.beta * ((q - prox) ** 2).sum()
        return data + prior + regularization

    def _adaptive_kernel_update(self, x, score, x0hat, operator,
                                measurement, sigma):
        """Backtrack a simplex-projected kernel step without a fixed lr_k."""
        kernel = x['kernel']
        current = self._kernel_objective(
            kernel, x['image'], x0hat['kernel'], measurement,
            operator, sigma
        )
        direction = score['kernel']
        direction_max = direction.abs().amax(
            dim=tuple(range(1, direction.ndim)), keepdim=True
        ).clamp_min(1e-12)

        # The trust radius is normalized by the kernel-grid resolution. No
        # absolute kernel learning rate or measurement-scale pair is needed.
        _, _, height, width = kernel.shape
        target_delta = self.kernel_trust_fraction / (height * width)
        step = target_delta / direction_max
        for _ in range(self.kernel_line_search_steps):
            candidate = kernel + step * direction
            candidate_q = self._normalize_kernel_latent(candidate)
            candidate = 2.0 * candidate_q - 1.0
            candidate_objective = self._kernel_objective(
                candidate, x['image'], x0hat['kernel'], measurement,
                operator, sigma
            )
            if torch.isfinite(candidate_objective) and candidate_objective <= current:
                x['kernel'] = candidate
                return
            step = step * 0.5

    def _block_simplex_update(self, x, x0hat, operator, measurement,
                              sigma, image_lr, image_noise):
        """Update the kernel on the simplex before refreshing image feedback."""
        tau = operator.sigma if self.tau is None else self.tau
        kernel = x['kernel']
        q = self._normalize_kernel_latent(kernel)
        q0 = self._normalize_kernel_latent(x0hat['kernel'])
        data_grad = -self._kernel_normalized_grad / tau ** 2
        prior_grad = (q0 - q) / max(sigma ** 2, 1e-12)
        regularizer_grad = -self.beta * (q - self._kernel_prox)
        kernel_lr = self.get_lr(0.0)['kernel']
        q = self._project_simplex(
            q + kernel_lr * (data_grad + prior_grad + regularizer_grad)
        )
        x['kernel'] = 2.0 * q - 1.0

        # The image correction must use the kernel accepted in this block.
        refreshed_data_grad = operator.gradient(
            {'image': x['image'], 'kernel': x['kernel']}, measurement
        )['image']
        image_prior_grad = (
            x0hat['image'] - x['image']
        ) / max(sigma ** 2, 1e-12)
        image_score = -refreshed_data_grad / tau ** 2 + image_prior_grad
        x['image'] = (
            x['image']
            + image_lr * image_score
            + np.sqrt(2.0 * image_lr) * image_noise
        )

    def get_prior_score(self, x, x0hat, xt, model, sigma, key):
        if self.prior_solver[key] == 'score-min' or self.prior_solver[key] == 'score-t' or self.prior_solver[key] == 'gaussian':
                    prior_score = self.prior_scores[key]
        elif self.prior_solver[key] == 'exact':
            prior_score = model.score(x, torch.tensor(self.prior_sigma_min).to(x.device)).detach()
        else:
            raise NotImplementedError
        return prior_score


    def prepare_prior_score(self, x0hat, xt, model_dict, sigma):
        """
        Precomputes the prior score based on the specified solver method.
        """
        self.prior_scores = {}
        for key, model in model_dict.items():
            if self.prior_solver[key] == 'score-min':
                self.prior_scores[key] = model.score(x0hat[key], self.prior_sigma_min).detach()

            elif self.prior_solver[key] == 'score-t':
                self.prior_scores[key] = model.score(xt[key], sigma).detach()

            elif self.prior_solver[key] == 'gaussian':
                self.prior_scores[key] = (x0hat[key] - xt[key]).detach() / sigma ** 2

            elif self.prior_solver[key] == 'exact':
                pass

            else:
                raise NotImplementedError


    def mc_prepare(self, x0hat, xt, model_dict, operator, measurement, sigma):
        """Prepares the sampler state before starting Monte Carlo sampling."""
        if self.mc_algo == 'hmc':
            self.velocity = {key: torch.randn_like(x0hat[key]) for key in x0hat.keys()}

    def mc_update(self, x, cur_score, lr, epsilon):
        """ Performs a single Monte Carlo update step (Langevin or HMC)."""
        if self.mc_algo == 'langevin':
            x_new = {}
            for key in x.keys():
                noise_scale = (
                    self.kernel_noise_scale if key == 'kernel' else 1.0
                )
                x_new[key] = (
                    x[key]
                    + lr[key] * cur_score[key]
                    + noise_scale * np.sqrt(2 * lr[key]) * epsilon[key]
                )
        elif self.mc_algo == 'hmc':  # (damping) hamiltonian monte carlo
            step_size = np.sqrt(lr)
            self.velocity = {key: self.momentum * self.velocity[key] + step_size * cur_score[key] + np.sqrt(
                2 * (1 - self.momentum)) * epsilon[key] for key in x.keys()}
            x_new = {key: x[key] + self.velocity[key] * step_size for key in x.keys()}
        else:
            raise NotImplementedError
        return x_new

    def sample_mh(self, xt, model_dict, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        if record:
            self.trajectory = {key: Trajectory() for key in xt.keys()}

        lr = self.get_lr(ratio)
        x = {key: x0hat[key].clone().detach() for key in x0hat.keys()}
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        for _ in pbar:
            # Gaussian proposal
            x_new = {key: x[key] + torch.randn_like(x[key]) * np.sqrt(lr) for key in x.keys()}
            # compute acceptance ratio
            loss_new = operator.loss(x_new, measurement)
            loss_old = operator.loss(x, measurement)
            log_data_ratio = (loss_old - loss_new) / (2 * self.tau ** 2)
            # compute prior p(x_0 | x_t)
            prior_loss_new = {key: (x_new[key] - x0hat[key]).pow(2).flatten(1).sum(-1) for key in x.keys()}
            prior_loss_old = {key: (x[key] - x0hat[key]).pow(2).flatten(1).sum(-1) for key in x.keys()}
            log_prior_ratio = {key: (prior_loss_old[key] - prior_loss_new[key]) / (2 * sigma ** 2) for key in x.keys()}
            # compute acceptance probability
            log_accept_prob = {key: log_data_ratio[key] + log_prior_ratio[key] for key in x.keys()}
            accept = {key: torch.rand_like(log_accept_prob[key]).log() < log_accept_prob[key] for key in x.keys()}
            accept = {key: accept[key].view(-1, *[1] * len(x[key].shape[1:])) for key in x.keys()}
            # update: accept new sample
            x = {key: torch.where(accept[key], x_new[key], x[key]) for key in x.keys()}
        return x


    def sample(self, xt, model_dict, x0hat, operator, measurement, sigma, ratio,
               kernel_denoiser, record=False, verbose=False):
        if self.mc_algo == 'mh':
            return self.sample_mh(xt, model_dict, x0hat, operator, measurement, sigma, ratio, record, verbose)

        if record:
            self.trajectory = {key: Trajectory() for key in xt.keys()}

        lr = self.get_lr(ratio)
        self.mc_prepare(x0hat, xt, model_dict, operator, measurement, sigma)
        self.prepare_prior_score(x0hat, xt, model_dict, sigma)

        x = {key: x0hat[key].clone().detach() for key in x0hat.keys()}
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)


        for i in pbar:
            # 统一计算整体score，避免多次前向传播
            cur_score, fitting_loss = self.score_fn(x, x0hat, model_dict, xt, operator, measurement, sigma, kernel_denoiser)
            epsilon = {key: torch.randn_like(x[key]) for key in x.keys()}

            if self.kernel_step_mode == 'manual':
                x = self.mc_update(x, cur_score, lr, epsilon)
            elif self.kernel_step_mode == 'block_simplex':
                self._block_simplex_update(
                    x, x0hat, operator, measurement, sigma,
                    lr['image'], epsilon['image']
                )
            elif self.kernel_step_mode == 'trust_region':
                image_score = {'image': cur_score['image']}
                image = self.mc_update(
                    {'image': x['image']}, image_score,
                    {'image': lr['image']}, {'image': epsilon['image']}
                )['image']
                x['image'] = image
                self._adaptive_kernel_update(
                    x, cur_score, x0hat, operator, measurement, sigma
                )
            else:
                raise ValueError(
                    f"Unsupported kernel step mode: {self.kernel_step_mode}"
                )

            # 早停检查
            if any(torch.isnan(x[key]).any() for key in x.keys()):
                return {key: torch.zeros_like(x[key]) for key in x.keys()}

            # 记录轨迹
            if record:
                self._record('image', x['image'], epsilon['image'], fitting_loss.sqrt())
                self._record('kernel', x['kernel'], epsilon['kernel'], fitting_loss.sqrt())
    #
        return x

    def _record(self, key, x, epsilon, loss):
        """
            Records the intermediate states during sampling for a specific component.
        """
        self.trajectory[key].add_tensor(f'xi', x)
        self.trajectory[key].add_tensor(f'epsilon', epsilon)
        self.trajectory[key].add_value(f'loss', loss)

    def huber_grad(self, x, delta=1.0, eps=1e-12):
        k_orig = (x + 1.0) / 2.0
        # 非负
        k_orig = torch.clamp(k_orig, min=0.0)
        # 归一化（按每个 kernel 的空间维度求和）
        s = k_orig.sum(dim=(-2, -1), keepdim=True)
        s = s.clamp_min(eps)
        k = k_orig / s
        grad_orig = torch.where(k <= delta, k, delta * torch.sign(k))
        # 链式法则：dk_orig/dk_scaled = 1/2
        grad_scaled = 0.5 * grad_orig
        return grad_scaled

    def tv_grad(self, img, eps=1e-12):
        """
        Compute ∇x for TV regularization.
        img: (B, C, H, W)
        Returns:
            gx, gy with same shape
        """
        gx = img[:, :, :, 1:] - img[:, :, :, :-1]
        gy = img[:, :, 1:, :] - img[:, :, :-1, :]

        # pad back to original size
        gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
        gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
        return gx, gy

    def tv_score(self, img, weight=1.0, eps=1e-12):
        gx, gy = self.tv_grad(img)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + eps)

        # normalized gradient direction
        nx = gx / mag
        ny = gy / mag

        # divergence (reverse differences)
        div_x = nx[:, :, :, :-1] - nx[:, :, :, 1:]
        div_y = ny[:, :, :-1, :] - ny[:, :, 1:, :]
        div_x = torch.nn.functional.pad(div_x, (1, 0, 0, 0))
        div_y = torch.nn.functional.pad(div_y, (0, 0, 1, 0))

        tv_grad = -(div_x + div_y)
        return weight * tv_grad

    def hyper_laplace_grad(self, x, p=0.6, eps=1e-6):
        # x: (B,C,H,W)
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
        gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))

        mag = torch.sqrt(gx ** 2 + gy ** 2 + eps)  # (B,C,H,W)
        coeff = p * (mag + eps) ** (p - 1)  # shape (B,C,H,W)

        nx = gx * coeff
        ny = gy * coeff

        # div (backward differences)
        div_x = nx[:, :, :, :-1] - nx[:, :, :, 1:]
        div_y = ny[:, :, :-1, :] - ny[:, :, 1:, :]

        div_x = torch.nn.functional.pad(div_x, (1, 0, 0, 0))
        div_y = torch.nn.functional.pad(div_y, (0, 0, 1, 0))
        grad = -(div_x + div_y)  # negative divergence
        return grad  # same shape as x

    def get_lr(self, ratio):
        """改进的学习率计算方法，为不同组件设置独立参数"""
        p_image, p_kernel = 1, 1  # 不同组件使用不同衰减指数
        multiplier = {
            'image': (1 ** (1 / p_image) + ratio * (
                        self.lr_min_ratio['image'] ** (1 / p_image) - 1 ** (1 / p_image))) ** p_image,
            'kernel': (1 ** (1 / p_kernel) + ratio * (
                        self.lr_min_ratio['kernel'] ** (1 / p_kernel) - 1 ** (1 / p_kernel))) ** p_kernel
        }
        return {
            'image': multiplier['image'] * self.lr['image'],
            'kernel': multiplier['kernel'] * self.lr['kernel']
        }


    def summary(self):
        print('+' * 50)
        print('MCMC Sampler Summary')
        print('+' * 50)
        print(f"Prior Solver    : {self.prior_solver}")
        print(f"MCMC Algorithm  : {self.mc_algo}")
        print(f"Num Steps       : {self.num_steps}")
        print(f"Learning Rate   : {self.lr}")
        print(f"Tau             : {self.tau}")
        print(f"LR Min Ratio    : {self.lr_min_ratio}")
        print('+' * 50)
