from torchattacks import APGDT
from torchattacks.attack import Attack
from torchattacks.wrappers.multiattack import MultiAttack
from torchattacks.attacks.fab import FAB
from torchattacks.attacks.square import Square
import numpy as np
import torch
import torch.nn.functional as F
import time


def loss_y(prob, labels):
    cost = -prob[np.arange(len(labels)), labels].mean()

    return cost


def loss_y_(prob, labels):
    cost = -prob[np.arange(len(labels)), labels]

    return cost


def loss_dpdr(prob, labels, k):
    prob_s, ind_s = prob.sort(dim=-1, descending=True)
    ind = (ind_s[:, 0] == labels).float()
    prob_tau = prob_s[:, 1] * ind + prob_s[:, 0] * (1 - ind)
    prob_y = prob[np.arange(len(labels)), labels]
    prob_diff_1 = prob_tau - prob_y
    prob_diff_2 = prob_tau - prob_s[:, k]
    offset = prob_diff_2.max().detach() / 2
    cost = (prob_diff_1 / (offset - prob_diff_1.sign() * (prob_diff_2 - offset) + 1e-10)).mean()

    return cost


def loss_dpdr_(prob, labels, k):
    prob_s, ind_s = prob.sort(dim=-1, descending=True)
    ind = (ind_s[:, 0] == labels).float()
    prob_tau = prob_s[:, 1] * ind + prob_s[:, 0] * (1 - ind)
    prob_y = prob[np.arange(len(labels)), labels]
    prob_diff_1 = prob_tau - prob_y
    prob_diff_2 = prob_tau - prob_s[:, k]
    offset = prob_diff_2.max().detach() / 2
    cost = prob_diff_1 / (offset - prob_diff_1.sign() * (prob_diff_2 - offset) + 1e-10)

    return cost


class SDM(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, total_steps=20, random_start=False):
        super().__init__("SDM", model)
        config = {
            10: [1, 5, 2],
            20: [1, 5, 4],
            50: [2, 5, 5],
            100: [2, 5, 10],
            200: [4, 5, 10],
            500: [4, 5, 25],
            1000: [5, 5, 40],
        }
        assert total_steps in config.keys()
        self.eps = eps
        self.alpha = alpha
        self.cycles, self.stages, self.steps = config[total_steps]
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.cycles):
            for k in range(self.stages):
                for _ in range(self.steps):
                    adv_images.requires_grad = True
                    prob = F.log_softmax(self.get_logits(adv_images), dim=-1)
                    if k == 0:
                        cost = loss_y(prob, labels)
                    else:
                        cost = loss_dpdr(prob, labels, k)
                    adv_images = self.update(cost, adv_images, images)

        return adv_images

    def update(self, cost, adv_images, images):
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False)[0]
        adv_images = adv_images.detach() + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class SDML2(Attack):
    def __init__(self, model, eps=1.0, alpha=0.2, total_steps=20, random_start=False, eps_for_division=1e-10):
        super().__init__("SDML2", model)
        config = {
            10: [1, 5, 2],
            20: [1, 5, 4],
            50: [2, 5, 5],
            100: [2, 5, 10],
            200: [4, 5, 10],
            500: [4, 5, 25],
            1000: [5, 5, 40],
        }
        assert total_steps in config.keys()
        self.eps = eps
        self.alpha = alpha
        self.cycles, self.stages, self.steps = config[total_steps]
        self.random_start = random_start
        self.eps_for_division = eps_for_division

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for _ in range(self.cycles):
            for k in range(self.stages):
                for _ in range(self.steps):
                    adv_images.requires_grad = True
                    prob = F.log_softmax(self.get_logits(adv_images), dim=-1)
                    if k == 0:
                        cost = loss_y(prob, labels)
                    else:
                        cost = loss_dpdr(prob, labels, k)
                    adv_images = self.update(cost, adv_images, images)

        return adv_images

    def update(self, cost, adv_images, images):
        batch_size = len(images)
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False)[0]
        grad_norms = (torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division)  # nopep8
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        adv_images = adv_images.detach() + self.alpha * grad
        delta = adv_images - images
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = self.eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class ASDM(Attack):
    def __init__(self, model, norm="Linf", eps=8 / 255, total_steps=50, n_restarts=1, seed=0,
                 eot_iter=1, rho=0.75, verbose=False):
        super().__init__("ASDM", model)
        config = {
            50: [2, 5, 5],
            100: [2, 5, 10],
            200: [2, 5, 20],
            500: [2, 5, 50],
            1000: [2, 5, 100],
        }
        assert total_steps in config.keys()
        self.norm = norm
        self.eps = eps
        self.cycles, self.stages, self.steps = config[total_steps]
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)

        return adv_images

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(
                x[np.arange(x.shape[0]), y]
                - x_sorted[:, -2] * ind
                - x_sorted[:, -1] * (1.0 - ind)
        ) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(self, x_in, y_in, kk):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )
        if self.verbose:
            print(
                "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
            )

        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                        t.reshape([t.shape[0], -1])
                        .abs()
                        .max(dim=1, keepdim=True)[0]
                        .reshape([-1, 1, 1, 1])
                    )  # nopep8
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                            (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # 1 forward pass (eot_iter = 1)
                logits = self.get_logits(x_adv)
                prob = F.log_softmax(logits, dim=-1)
                if kk == 0:
                    loss_indiv = loss_y_(prob, y)
                else:
                    loss_indiv = loss_dpdr_(prob, y, kk)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
                self.eps
                * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
                * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  # nopep8
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # n_reduced = 0
        for i in range(self.steps):
            # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        0.0,
                        1.0,
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * grad / (
                            (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                                + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                                + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                x_adv = x_adv_1 + 0.0

            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # 1 forward pass (eot_iter = 1)
                    logits = self.get_logits(x_adv)
                    prob = F.log_softmax(logits, dim=-1)
                    if kk == 0:
                        loss_indiv = loss_y_(prob, y)
                    else:
                        loss_indiv = loss_dpdr_(prob, y, kk)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                    x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )  # nopep8
            if self.verbose:
                print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

            # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )
                    fl_reduce_no_impr = (~reduced_last_check) * (
                            loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
                    )  # nopep8
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ["Linf", "L2"]
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        acc = self.get_logits(x).max(1)[1] == y
        # loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print(
                "-------------------------- running {}-attack with epsilon {:.4f} --------------------------".format(
                    self.norm, self.eps
                )
            )
            print("initial accuracy: {:.2%}".format(acc.float().mean()))
        startt = time.time()

        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not cheap:
                raise ValueError("not implemented yet")

            else:
                for _ in range(self.cycles):
                    for kk in range(self.stages):
                        for counter in range(self.n_restarts):
                            ind_to_fool = acc.nonzero().squeeze()
                            if len(ind_to_fool.shape) == 0:
                                ind_to_fool = ind_to_fool.unsqueeze(0)
                            if ind_to_fool.numel() != 0:
                                x_to_fool, y_to_fool = (
                                    x[ind_to_fool].clone(),
                                    y[ind_to_fool].clone(),
                                )  # nopep8
                                (
                                    best_curr,
                                    acc_curr,
                                    loss_curr,
                                    adv_curr,
                                ) = self.attack_single_run(
                                    x_to_fool, y_to_fool, kk
                                )  # nopep8
                                ind_curr = (acc_curr == 0).nonzero().squeeze()
                                #
                                acc[ind_to_fool[ind_curr]] = 0
                                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                                if self.verbose:
                                    print(
                                        "restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s".format(
                                            counter, acc.float().mean(), time.time() - startt
                                        )
                                    )

            return acc, adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (
                -float("inf")
            )  # nopep8
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.0
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.0

                if self.verbose:
                    print("restart {} - loss: {:.5f}".format(counter, loss_best.sum()))

            return loss_best, adv_best


class AutoAttackSDM(Attack):
    def __init__(self, model, norm="Linf", eps=8 / 255, n_classes=10, seed=None, verbose=False):
        super().__init__("AutoAttack", model)
        assert norm == "Linf" or norm == "L2"

        self.norm = norm
        self.eps = eps
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose

        if norm == "Linf":
            sdm = SDM(model, eps=eps, total_steps=200)
        else:
            sdm = SDML2(model, eps=eps, total_steps=200)

        # [‘sdm-200’, 'fab', 'square', 'apgd-t', 'fab-t']
        self._autoattack = MultiAttack(
            [
                sdm,
                FAB(
                    model,
                    eps=eps,
                    norm=norm,
                    seed=self.get_seed(),
                    verbose=verbose,
                    n_classes=n_classes,
                    n_restarts=5,
                ),
                Square(
                    model,
                    eps=eps,
                    norm=norm,
                    seed=self.get_seed(),
                    verbose=verbose,
                    n_queries=5000,
                    n_restarts=1,
                ),
                APGDT(
                    model,
                    eps=eps,
                    norm=norm,
                    seed=self.get_seed(),
                    verbose=verbose,
                    n_classes=n_classes,
                    n_restarts=1,
                ),
                FAB(
                    model,
                    eps=eps,
                    norm=norm,
                    seed=self.get_seed(),
                    verbose=verbose,
                    multi_targeted=True,
                    n_classes=n_classes,
                    n_restarts=1,
                ),
            ]
        )

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed
