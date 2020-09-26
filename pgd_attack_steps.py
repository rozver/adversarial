import torch


class AttackStep:
    def __init__(self, orig_x, eps, step_size):
        self.orig_x = orig_x
        self.eps = eps
        self.step_size = step_size

    def project(self, x):
        raise NotImplementedError

    def step(self, x, g):
        raise NotImplementedError

    def random_perturb(self, x):
        raise NotImplementedError


class LinfStep(AttackStep):
    def project(self, x):
        diff = x - self.orig_x
        diff = torch.clamp(diff, -self.eps, self.eps)
        new_x = torch.clamp(diff + self.orig_x, 0, 1)
        return new_x

    def step(self, x, grads):
        step = torch.sign(grads)*self.step_size
        new_x = x + step
        return new_x

    def random_perturb(self, x):
        perturbation = torch.rand_like(x)
        new_x = x + 2*(perturbation-0.5)*self.eps
        return new_x
