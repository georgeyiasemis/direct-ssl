# coding=utf-8
# Copyright (c) DIRECT Contributors

# Code copied and slightly adapted from https://github.com/kyegomez/Sophia/tree/main
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.optim.optimizer import Optimizer


class Sophia(Optimizer):
    def __init__(
        self,
        model,
        input_data,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        k=10,
        estimator="Hutchinson",
        rho=1,
    ):
        self.model = model
        self.input_data = input_data
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k, estimator=estimator, rho=rho)
        super(Sophia, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")

                state = self.state[p]

                # state init
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["h"] = torch.zeros_like(p.data)

                m, h = state["m"], state["h"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # update biased first moment estimate
                m.mul_(beta1).add_(1 - beta1, grad)

                # update hessian estimate
                if state["step"] % group["k"] == 1:
                    if group["estimator"] == "Hutchinson":
                        hessian_estimate = self.hutchinson(p, grad)
                    elif group["estimator"] == "Gauss-Newton-Bartlett":
                        hessian_estimate = self.gauss_newton_bartlett(p, grad)
                    else:
                        raise ValueError("Invalid estimator choice")
                    h.mul_(beta2).add_(1 - beta2, hessian_estimate)

                # update params
                p.data.add_(-group["lr"] * group["weight_decay"], p.data)
                p.data.addcdiv_(-group["lr"], m, h.add(group["eps"]).clamp(max=group["rho"]))

        return loss

    def hutchinson(self, p, grad):
        u = torch.randn_like(grad)
        grad_dot_u = torch.sum(grad * u)
        hessian_vector_product = torch.autograd.grad(grad_dot_u, p, retain_graph=True)[0]
        return u * hessian_vector_product

    def gauss_newton_bartlett(self, p, grad):
        B = len(self.input_data)
        logits = [self.model(xb) for xb in self.input_data]
        y_hats = [torch.softmax(logit, dim=0) for logit in logits]
        g_hat = torch.autograd.grad(
            sum([self.loss_function(logit, y_hat) for logit, y_hat in zip(logits, y_hats)]) / B, p, retain_graph=True
        )[0]
        return B * g_hat * g_hat