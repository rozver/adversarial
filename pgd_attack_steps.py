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

    def random_perturb(self, x, mask):
        raise NotImplementedError


class LinfStep(AttackStep):
    def project(self, x):
        diff = x - self.orig_x
        diff = torch.clamp(diff, -self.eps, self.eps)
        new_x = torch.clamp(self.orig_x+diff, 0, 1)
        return new_x

    def step(self, x, grad):
        step = torch.sign(grad)*self.step_size
        new_x = x + step
        return new_x

    def random_perturb(self, x, mask):
        perturbation = mask*torch.rand_like(x)
        new_x = x + 2*(perturbation-0.5)*self.eps
        return new_x


class L2Step(AttackStep):
    def project(self, x):
        diff = x - self.orig_x
        diff_normed = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        new_x = torch.clamp(self.orig_x+diff_normed, 0, 1)
        return new_x

    def step(self, x, grad):
        norm = torch.norm(grad, p=2, keepdim=True).detach()
        grad_normed = grad.div(norm)
        new_x = x + grad_normed*self.eps
        return new_x

    def random_perturb(self, x, mask):
        perturbation = mask*torch.rand_like(x)
        new_x = self.project(self.orig_x+perturbation)
        return new_x
