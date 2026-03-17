import math
from typing import Callable, Optional, Union

import torch

from solver import solve_isotropic_covariance, symKL_objective


class _MarvellGradientPerturbFunction(torch.autograd.Function):
    """Autograd function that keeps forward identity and perturbs backward gradients.

    This mirrors TensorFlow `KL_gradient_perturb_function_creator` semantics in
    `custom_gradients_masking.py`.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        p_frac: Union[str, float],
        dynamic: bool,
        sumkl_threshold: Optional[float],
        init_scale: float,
        uv_choice: str,
        summary_writer=None,
        step_getter: Optional[Callable[[], int]] = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(y)
        ctx.p_frac = p_frac
        ctx.dynamic = dynamic
        ctx.sumkl_threshold = sumkl_threshold
        ctx.init_scale = init_scale
        ctx.uv_choice = uv_choice
        ctx.summary_writer = summary_writer
        ctx.step_getter = step_getter
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (y,) = ctx.saved_tensors
        g_original_shape = grad_output.shape
        g = grad_output.reshape(g_original_shape[0], -1)

        y_bool = y == 1
        pos_g = g[y_bool]
        neg_g = g[~y_bool]

        pos_g_mean = torch.mean(pos_g, dim=0, keepdim=True)
        neg_g_mean = torch.mean(neg_g, dim=0, keepdim=True)

        pos_coordinate_var = torch.mean((pos_g - pos_g_mean) ** 2, dim=0)
        neg_coordinate_var = torch.mean((neg_g - neg_g_mean) ** 2, dim=0)

        avg_pos_coordinate_var = torch.mean(pos_coordinate_var)
        avg_neg_coordinate_var = torch.mean(neg_coordinate_var)

        g_diff = pos_g_mean - neg_g_mean
        g_diff_norm = float(torch.norm(g_diff).detach().cpu().item())

        if ctx.uv_choice == "uv":
            u = float(avg_neg_coordinate_var.detach().cpu().item())
            v = float(avg_pos_coordinate_var.detach().cpu().item())
            if u == 0.0:
                print("neg_g")
                print(neg_g)
            if v == 0.0:
                print("pos_g")
                print(pos_g)
        elif ctx.uv_choice == "same":
            uv = float((avg_neg_coordinate_var + avg_pos_coordinate_var).detach().cpu().item()) / 2.0
            u, v = uv, uv
        elif ctx.uv_choice == "zero":
            u, v = 0.0, 0.0
        else:
            raise AssertionError(f"unsupported uv_choice: {ctx.uv_choice}")

        d = float(g.shape[1])

        if ctx.p_frac == "pos_frac":
            p = float(torch.sum(y).detach().cpu().item() / len(y))
        else:
            p = float(ctx.p_frac)

        scale = ctx.init_scale
        lam10, lam20, lam11, lam21 = None, None, None, None

        while True:
            p_power = scale * g_diff_norm**2
            lam10, lam20, lam11, lam21, sumkl = solve_isotropic_covariance(
                u=u,
                v=v,
                d=d,
                g=g_diff_norm**2,
                p=p,
                P=p_power,
                lam10_init=lam10,
                lam20_init=lam20,
                lam11_init=lam11,
                lam21_init=lam21,
            )

            if (not ctx.dynamic) or (sumkl <= ctx.sumkl_threshold):
                break

            scale *= 1.5

        if ctx.summary_writer is not None:
            step = ctx.step_getter() if ctx.step_getter is not None else 0
            ctx.summary_writer.add_scalar("solver/u", u, step)
            ctx.summary_writer.add_scalar("solver/v", v, step)
            ctx.summary_writer.add_scalar("solver/g", g_diff_norm**2, step)
            ctx.summary_writer.add_scalar("solver/p", p, step)
            ctx.summary_writer.add_scalar("solver/scale", scale, step)
            ctx.summary_writer.add_scalar("solver/P", p_power, step)
            ctx.summary_writer.add_scalar("solver/lam10", lam10, step)
            ctx.summary_writer.add_scalar("solver/lam20", lam20, step)
            ctx.summary_writer.add_scalar("solver/lam11", lam11, step)
            ctx.summary_writer.add_scalar("solver/lam21", lam21, step)
            ctx.summary_writer.add_scalar(
                "sumKL_before",
                symKL_objective(
                    lam10=0.0,
                    lam20=0.0,
                    lam11=0.0,
                    lam21=0.0,
                    u=float(avg_neg_coordinate_var.detach().cpu().item()),
                    v=float(avg_pos_coordinate_var.detach().cpu().item()),
                    d=d,
                    g=g_diff_norm**2,
                ),
                step,
            )
            ctx.summary_writer.add_scalar("sumKL_after", sumkl, step)
            ctx.summary_writer.add_scalar("error prob lower bound", 0.5 - math.sqrt(sumkl) / 4, step)

        perturbed_g = g.clone()
        y_float = y.to(dtype=g.dtype).reshape(-1)

        pos_random = torch.randn(y.shape, device=g.device, dtype=g.dtype)
        perturbed_g = perturbed_g + (
            (pos_random * y_float).reshape(-1, 1)
            * g_diff
            * (math.sqrt(lam11 - lam21) / g_diff_norm)
        )

        if lam21 > 0.0:
            perturbed_g = perturbed_g + (
                torch.randn(g.shape, device=g.device, dtype=g.dtype)
                * y_float.reshape(-1, 1)
                * math.sqrt(lam21)
            )

        neg_random = torch.randn(y.shape, device=g.device, dtype=g.dtype)
        perturbed_g = perturbed_g + (
            (neg_random * (1 - y_float)).reshape(-1, 1)
            * g_diff
            * (math.sqrt(lam10 - lam20) / g_diff_norm)
        )

        if lam20 > 0.0:
            perturbed_g = perturbed_g + (
                torch.randn(g.shape, device=g.device, dtype=g.dtype)
                * (1 - y_float).reshape(-1, 1)
                * math.sqrt(lam20)
            )

        return perturbed_g.reshape(g_original_shape), None, None, None, None, None, None, None, None


class MarvellGradientPerturbLayer(torch.nn.Module):
    """PyTorch Marvell layer equivalent to TF sumKL gradient perturbation.

    Forward pass is identity. Backward pass perturbs the incoming gradients.

    Usage:
        layer = MarvellGradientPerturbLayer(...)
        x = layer(x, y_batch)
    """

    def __init__(
        self,
        p_frac: Union[str, float] = "pos_frac",
        dynamic: bool = False,
        error_prob_lower_bound: Optional[float] = None,
        sumkl_threshold: Optional[float] = None,
        init_scale: float = 1.0,
        uv_choice: str = "uv",
        summary_writer=None,
        step_getter: Optional[Callable[[], int]] = None,
    ):
        super().__init__()
        print("p_frac", p_frac)
        print("dynamic", dynamic)

        if dynamic and (error_prob_lower_bound is not None):
            sumkl_threshold = (2 - 4 * error_prob_lower_bound) ** 2
            print("error_prob_lower_bound", error_prob_lower_bound)
            print("implied sumKL_threshold", sumkl_threshold)
        elif dynamic:
            print("using sumKL_threshold", sumkl_threshold)

        print("init_scale", init_scale)
        print("uv_choice", uv_choice)

        self.p_frac = p_frac
        self.dynamic = dynamic
        self.sumkl_threshold = sumkl_threshold
        self.init_scale = init_scale
        self.uv_choice = uv_choice
        self.summary_writer = summary_writer
        self.step_getter = step_getter

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _MarvellGradientPerturbFunction.apply(
            x,
            y,
            self.p_frac,
            self.dynamic,
            self.sumkl_threshold,
            self.init_scale,
            self.uv_choice,
            self.summary_writer,
            self.step_getter,
        )
