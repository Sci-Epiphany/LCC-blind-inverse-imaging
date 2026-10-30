import tqdm
import torch
import numpy as np
import torch.nn as nn
from .trajectory import Trajectory


class MCMCNLSampler(nn.Module):
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

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01, prior_solver='gaussian', prior_sigma_min=1e-2,
                 mc_algo='langevin', momentum=0.9, reg_terms='none'):
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

    def score_fn(self, x, x0hat, model_dict, xt, operator, measurement, sigma):
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

        # Compute data-fitting gradient and loss
        data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)

        # # --- Step 1: HQS auxiliary variable update ---
        with torch.no_grad():
            res = regularization_step(x['kernel'], self.lamb, self.beta, how='l2')
        # --- Step 2: compute reg gradient from quadratic penalty term ---
        reg_grad = self.beta * (x['kernel'] - res)

        # Compute terms for each component
        score_dict = {}

        for key in xt.keys():
            data_term = -data_fitting_grad[key] / self.tau ** 2

            if key == 'kernel':
                # Compute kernel huber reg_grad
                x0_term = (x0hat[key] - x[key]) * 1e5
                score_dict[key] = data_term - reg_grad  - x0_term
            else:
                prior_term = self.get_prior_score(x[key], x0hat[key], xt[key], model_dict, sigma)
                xt_term = (xt[key] - x[key]) / sigma ** 2

                score_dict[key] = data_term + xt_term + prior_term

        return score_dict, data_fitting_loss

    def get_prior_score(self, x, x0hat, xt, model, sigma):
        if self.prior_solver == 'score-min' or self.prior_solver == 'score-t' or self.prior_solver == 'gaussian':
            prior_score = self.prior_score
        elif self.prior_solver== 'exact':
            prior_score = model.score(x, torch.tensor(self.prior_sigma_min).to(x.device)).detach()
        else:
            raise NotImplementedError
        return prior_score

    def prepare_prior_score(self, x0hat, xt, model, sigma):
        """
        Precomputes the prior score based on the specified solver method.
        """
        if self.prior_solver == 'score-min':
            self.prior_score = model.score(x0hat, self.prior_sigma_min).detach()

        elif self.prior_solver == 'score-t':
            self.prior_score = model.score(xt, sigma).detach()

        elif self.prior_solver == 'gaussian':
            self.prior_score = (x0hat - xt).detach() / sigma ** 2

        elif self.prior_solver == 'exact':
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
            x_new = {key: x[key] + lr[key] * cur_score[key] + np.sqrt(2 * lr[key]) * epsilon[key] for key in x.keys()}
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

    def sample(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        if self.mc_algo == 'mh':
            return self.sample_mh(xt, model, x0hat, operator, measurement, sigma, ratio, record, verbose)

        if record:
            self.trajectory = {key: Trajectory() for key in xt.keys()}

        lr = self.get_lr(ratio)
        # self.mc_prepare(x0hat, xt, model, operator, measurement, sigma)
        self.prepare_prior_score(x0hat['image'], xt['image'], model, sigma)

        x = {key: x0hat[key].clone().detach() for key in x0hat.keys()}
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)

        for i in pbar:
            # 统一计算整体score，避免多次前向传播
            cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma)
            epsilon = {key: torch.randn_like(x[key]) for key in x.keys()}

            x_new = self.mc_update(x, cur_score, lr, epsilon)

            # 早停检查
            if any(torch.isnan(x_new[key]).any() for key in x_new.keys()):
                return {key: torch.zeros_like(x[key]) for key in x.keys()}

            x = x_new

            # 记录轨迹
            if record:
                self._record('image', x['image'], epsilon['image'], fitting_loss.sqrt())
                self._record('kernel', x['kernel'], epsilon['kernel'], fitting_loss.sqrt())

        return x

    def _record(self, x, epsilon, loss):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xi', x)
        self.trajectory.add_tensor(f'epsilon', epsilon)
        self.trajectory.add_value(f'loss', loss)

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


def regularization_step(k, lamb, beta, denoiser=None, how='l1'):
    _, _, n, p = k.shape
    # k = k / (k.sum(dim=(-2, -1), keepdim=True) + 1e-12)
    if how == 'l0':
        threshold = (2 * lamb / beta) ** 0.5
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
            sigma_d = torch.ones_like(k)[:, :1, :, :] * (np.sqrt(lamb / beta))
            res = denoiser(torch.cat((k, sigma_d), axis=1))
        else:
            res = k
    return res
