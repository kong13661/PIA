import torch
from typing import Callable


class EpsGetter:
    def __init__(self, model):
        self.model = model

    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        raise NotImplementedError


class Attacker:
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize: Callable = None, denormalize: Callable = None):
        self.eps_getter = eps_getter
        self.betas = betas
        self.noise_level = torch.cumprod(1 - betas, dim=0).float()
        self.interval = interval
        self.attack_num = attack_num
        self.normalize = normalize
        self.denormalize = denormalize
        self.T = len(self.noise_level)

    def __call__(self, x0, xt, condition):
        raise NotImplementedError

    def get_xt_coefficient(self, step):
        return self.noise_level[step] ** 0.5, (1 - self.noise_level[step]) ** 0.5

    def get_xt(self, x0, step, eps):
        a_T, b_T = self.get_xt_coefficient(step)
        return a_T * x0 + b_T * eps

    def _normalize(self, x):
        if self.normalize is not None:
            return self.normalize(x)
        return x

    def _denormalize(self, x):
        if self.denormalize is not None:
            return self.denormalize(x)
        return x


class DDIMAttacker(Attacker):
    def get_y(self, x, step):
        return (1 / self.noise_level[step] ** 0.5) * x

    def get_x(self, y, step):
        return y * self.noise_level[step] ** 0.5

    def get_p(self, step):
        return (1 / self.noise_level[step] - 1) ** 0.5

    def get_reverse_and_denoise(self, x0, condition, step=None):
        x0 = self._normalize(x0)
        intermediates = self.ddim_reverse(x0, condition)
        intermediates_denoise = self.ddim_denoise(x0, intermediates, condition)
        return torch.stack(intermediates), torch.stack(intermediates_denoise)

    def __call__(self, x0, condition=None):
        intermediates, intermediates_denoise = self.get_reverse_and_denoise(x0, condition)
        return self.distance(intermediates, intermediates_denoise)

    def distance(self, x0, x1):
        return ((x0 - x1).abs()**2).flatten(2).sum(dim=-1)

    def ddim_reverse(self, x0, condition):
        raise NotImplementedError

    def ddim_denoise(self, x0, intermediates, condition):
        raise NotImplementedError


class SecMIAttacker(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        x = x0
        intermediates.append(x0)

        for step in range(0, terminal_step, self.interval):
            y_next = self.eps_getter(x, condition, self.noise_level, step) * (self.get_p(step + self.interval) - self.get_p(step)) + self.get_y(x, step)
            x = self.get_x(y_next, step + self.interval)
            intermediates.append(x)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        ternimal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, ternimal_step + self.interval, self.interval), 1):
            x = intermediates[idx]
            y_prev = self.eps_getter(x, condition, self.noise_level, step) * (self.get_p(step - self.interval) - self.get_p(step)) + self.get_y(x, step)
            x_prev = self.get_x(y_prev, step - self.interval)
            x = x_prev
            intermediates_denoise.append(x_prev)

            if idx == len(intermediates) - 1:
                del intermediates[-1]
        return intermediates_denoise

    def get_prev_from_eps(self, x0, eps_x0, eps, t):
        t = t + self.interval
        xta1 = self.get_xt(x0, t, eps_x0)

        y_prev = eps * (self.get_p(t - self.interval) - self.get_p(t)) + self.get_y(xta1, t)
        x_prev = self.get_x(y_prev, t - self.interval)
        return x_prev


class PIA(DDIMAttacker):
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize: Callable = None, denormalize: Callable = None, lp=4):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        self.lp = lp

    def distance(self, x0, x1):
        return ((x0 - x1).abs()**self.lp).flatten(2).sum(dim=-1)

    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, self.noise_level, 0)
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise


class PIAN(DDIMAttacker):

    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize: Callable = None, denormalize: Callable = None, lp=4):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        self.lp = lp

    def distance(self, x0, x1):
        return ((x0 - x1).abs()**self.lp).flatten(2).sum(dim=-1)

    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, self.noise_level, 0)
        eps = eps / eps.abs().mean(list(range(1, eps.ndim)), keepdim=True) * (2 / torch.pi) ** 0.5
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise


class NaiveAttacker(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        # x = x0
        terminal_step = self.interval * self.attack_num
        for _ in reversed(range(0, terminal_step, self.interval)):
            eps = torch.randn_like(x0)
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise
