from typing import Optional, List

import torch
import numpy as np

from tabicl.prior.graph_lib._base import RandomTensorTransformer, Context
from tabicl.prior.graph_lib._activation import RandomActivation
from tabicl.prior.graph_lib._matrix import RandomMatrix, RandomGaussianMatrix
from tabicl.prior.graph_lib._weights import RandomWeights


class RandomFunction(RandomTensorTransformer):
    """
    Base class for random functions that take one tensor (n_samples, in_features) as input.
    """
    def __init__(self, context: Context, in_features: int, out_features: int):
        """
        :param context: Context.
        :param in_features: Input dimensionality.
        :param out_features: Output dimensionality.
        """
        super().__init__(context)
        self.in_features = in_features
        self.out_features = out_features

    def _fit(self, x: torch.Tensor):
        fct_types_dict = {
            'mlp': RandomMLPFunction,
            'tree': RandomTreeFunction,
            'disc': RandomDiscretizationFunction,
            'lin': RandomLinearFunction,
            'quad': RandomQuadraticFunction,
            'gp': RandomGPFunction,
            'em': RandomEMAssignmentFunction,
            'prod': RandomProductFunction,
            # ----- Physics-informed additions (manufacturing) -----
            'arrhenius': RandomArrheniusFunction,
            'archard': RandomArchardWearFunction,
            'hollomon': RandomHollomonHardeningFunction,
            'basquin': RandomBasquinFatigueFunction,
            'fourier_heat': RandomFourierHeatFunction,
            'taylor_wear': RandomTaylorToolWearFunction,
            'newton_cooling': RandomNewtonCoolingFunction,
            'tol_stackup': RandomToleranceStackupFunction,
            'preston': RandomPrestonFunction,
        }
        presets_dict = {
            'default': 'mlp,tree,disc,lin,quad,gp,em,prod',
            'tabpfnv2': 'mlp,tree,disc',
            # New preset: physics formulas only, useful for pure ablations
            'physics': 'arrhenius,archard,hollomon,basquin,fourier_heat,taylor_wear,newton_cooling,tol_stackup,preston',
            # New preset: original prior + physics formulas mixed together
            'mix_physics': (
                'mlp,tree,disc,lin,quad,gp,em,prod,'
                'arrhenius,archard,hollomon,basquin,fourier_heat,taylor_wear,newton_cooling,tol_stackup,preston'
            ),
        }
        fct_types: List[type] = []
        for name in self.config.fct_types.split(','):
            if name in presets_dict:
                fct_types.extend([fct_types_dict[type_name] for type_name in presets_dict[name].split(',')])
            else:
                fct_types.append(fct_types_dict[name])

        fct_type = self.sampler.choice("random_function_fct_type", fct_types)
        self.fct_ = fct_type(self.context, self.in_features, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fct_(x)



class CheapRandomFunction(RandomFunction):
    """
    Only samples from a subset of random function types,
    to be faster and to avoid infinite recursion in some settings.
    """
    def _fit(self, x: torch.Tensor):
        fct_types = [
            RandomGPFunction,
            RandomTreeFunction,
            RandomDiscretizationFunction,
            RandomLinearFunction,
            RandomQuadraticFunction,
            # Physics functions are all closed-form / non-recursive, so they are cheap by construction
            RandomArrheniusFunction,
            RandomArchardWearFunction,
            RandomHollomonHardeningFunction,
            RandomBasquinFatigueFunction,
            RandomFourierHeatFunction,
            RandomTaylorToolWearFunction,
            RandomNewtonCoolingFunction,
            RandomToleranceStackupFunction,
            RandomPrestonFunction,
        ]
        fct_type = self.sampler.choice("cheap_random_function_fct_type", fct_types)
        self.fct_ = fct_type(self.context, self.in_features, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fct_(x)


class RandomMLPFunction(RandomFunction):
    """
    Random multilayer perceptron (MLP). Can have activations at the beginning and end.
    """
    def _fit(self, x: torch.Tensor):
        width = self.sampler.randint("nn_width", 1, 128, use_log=True)
        # n_layers = np.random.randint(2, 5)
        # count linear and activation as separate layers
        n_layers = self.sampler.randint("nn_n_layers", 1, 4, use_log=True)
        self.start_act_ = self.sampler.boolean("nn_start_act")
        self.end_act_ = self.sampler.boolean("nn_end_act")
        layer_sizes = [x.shape[-1]] + ([width] * (n_layers - 1)) + [self.out_features]
        self.linears_ = [
            RandomLinearFunction(self.context, n_in, n_out)
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.acts_ = [
            RandomActivation(self.context)
            for _ in range(n_layers - 1 + int(self.start_act_) + int(self.end_act_))
        ]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.acts_
        if self.start_act_:
            x = acts[0](x)
            acts = acts[1:]

        if self.end_act_:
            for lin, act in zip(self.linears_, acts):
                x = act(lin(x))
        else:
            for lin, act in zip(self.linears_[:-1], acts):
                x = act(lin(x))
            x = self.linears_[-1](x)

        return x


class RandomTreeFunction(RandomFunction):
    """
    An ensemble of oblivious (=symmetric) decision trees (as in CatBoost) with random leaf values.
    """
    def _fit(self, x: torch.Tensor):
        from tabicl.prior.graph_lib._points import RandomGaussianPoints

        self.feature_imp_ = torch.clamp(x.std(dim=0, correction=0), 1e-8)
        self.feature_imp_[~torch.isfinite(self.feature_imp_)] = 1e-8
        # self.feature_imp_ /= self.feature_imp_.sum()
        self.n_trees_ = self.sampler.numerical("rt_n_trees", 1, 128, use_log=True, use_int=True)
        self.depth_ = self.sampler.randint("rt_depth", 1, 8)
        self.n_leaves_ = 2**self.depth_
        self.n_splits_ = self.n_trees_ * self.depth_
        self.n_leaf_values_ = self.n_trees_ * self.n_leaves_
        # print(f'{self.n_trees_=}, {self.depth_=}')
        self.split_dims_ = torch.multinomial(self.feature_imp_, self.n_splits_, replacement=True).to(self.device)
        self.split_points_ = x[torch.randint(x.shape[0], size=(self.n_splits_,), device=self.device), self.split_dims_]
        # use Gaussian points for now to avoid too strong recursion
        self.leaf_values_ = (
            RandomGaussianPoints(self.context)
            .sample(self.n_leaf_values_, self.out_features)
            .reshape(self.n_trees_, self.n_leaves_, self.out_features)
        )
        self.idx_multipliers_ = 2 ** torch.arange(self.depth_, dtype=torch.long, device=self.device)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input tensor into leaf node indices for each tree in the ensemble.

        Args:
            x: Input tensor of shape (n_samples, n_features) to be transformed

        Returns:
            Tensor of shape (n_samples, n_trees) containing leaf node indices for each tree

        Note:
            Uses float32 casting for boolean split decisions to enable CUDA operations, followed
            by conversion to long integer indices. This resolves the "baddbmm_cuda not implemented
            for Long" error while maintaining index integrity.
        """
        split_dims = self.split_dims_.to(x.device)
        split_points = self.split_points_.to(x.device)
        split_sides = x[:, split_dims] > split_points
        split_sides = split_sides.reshape(x.shape[0], self.n_trees_, self.depth_)
        # tree_idxs has the same shape (n_batch, n_trees) as leaf_idxs
        leaf_idxs = torch.einsum("btd,d->bt", split_sides.float(), self.idx_multipliers_.float()).long()
        tree_idxs = torch.arange(self.n_trees_, dtype=torch.long, device=x.device).expand(x.shape[0], self.n_trees_)

        # this will do
        # leaf_values[batch, tree, out] = self.leaf_values[tree_idxs[batch, tree], leaf_idxs[batch, tree], out]
        # = self.leaf_values[tree, leaf_idxs[batch, tree], out]
        leaf_values = self.leaf_values_.to(x.device)
        result = leaf_values[tree_idxs, leaf_idxs]

        return result.mean(dim=1)


class RandomDiscretizationFunction(RandomFunction):
    """
    Discretizes to the closest point from a subset of points, then applies a linear function.
    """
    def __init__(self, context: Context, in_features: int, out_features: int, n_centers: Optional[int] = None):
        super().__init__(context, in_features, out_features)
        self.n_centers = n_centers

    def _fit(self, x: torch.Tensor):
        if x.shape[0] < 2:
            assert x.shape[0] == 1
            self.n_centers = 1
            self.centers_ = x
        else:
            if self.n_centers is None:
                self.n_centers = self.sampler.randint(
                    "rand_discr_n_centers", 2, min(x.shape[0], self.config.max_discretization_cardinality), use_log=True
                )
            else:
                self.n_centers = min(self.n_centers, x.shape[0])
            self.centers_ = x[torch.randperm(len(x), device=x.device)[: self.n_centers]]

        self.p_ = self.sampler.numerical("cat_disc_p", 0.5, 4.0, use_log=True)

        # pass the centers through a random linear function to get something with self.out_features features
        # (which is equivalent to composing this with a random linear function, but not equally fast)
        self.targets_ = RandomLinearFunction(self.context, self.in_features, self.out_features)(self.centers_)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.centers_, self.p_)  # this is faster than manually calculating the distance matrix
        closest_idx = dists.argmin(dim=-1)
        return self.targets_[closest_idx]


class RandomGPFunction(RandomFunction):
    # Gaussian process function with extra linear transform and random kernel
    def __init__(self, context: Context, in_features: int, out_features: int, n_frequencies: int = 256):
        super().__init__(context, in_features, out_features)
        self.n_frequencies = n_frequencies

    def _fit(self, x: torch.Tensor):
        # global decay rate a > 1
        a = self.sampler.numerical("gp_a", 2.0, 20.0, use_log=True)

        if self.sampler.boolean("generalized_gp_use_product_kernel"):
            # uniform samples for inverse CDF sampling
            u = torch.rand(self.in_features, self.n_frequencies, device=x.device)
            u = torch.clamp(u, 1e-6, 1 - 1e-6)  # avoid too much trouble with inverse CDF
            # get distribution with density f(x) = (a-1)(1+x)^{-a} on [0, \infty)
            # CDF is F(x) = 1 - (1 + x)^{1-a}
            # inverse CDF is F^{-1}(u) = (1 - u)^{1/(1-a)} - 1
            # since u is uniform on [0, 1], we can use u instead of 1-u
            # We don't have a random input matrix here because we want the kernel to stay axis-aligned
            invcdf = torch.pow(u, 1 / (1 - a)) - 1.0
            self.freqs_ = invcdf
        else:
            # uniform samples for inverse CDF sampling, as above
            u = torch.rand(self.n_frequencies, device=x.device)
            u = torch.clamp(u, 1e-6, 1 - 1e-6)  # avoid too much trouble with inverse CDF
            invcdf = torch.pow(u, 1 / (1 - a)) - 1.0
            self.freqs_ = torch.randn(self.in_features, self.n_frequencies, device=x.device)
            self.freqs_ *= invcdf[None, :] / self.freqs_.norm(dim=0, keepdim=True)
            input_tfm = RandomGaussianMatrix(self.context).sample(1, self.in_features, self.in_features)[0]
            # multiply the output with random weights, so the function can vary less in certain directions
            input_tfm = input_tfm * RandomWeights(self.context).sample(1, self.in_features).t()
            input_tfm *= self.sampler.numerical("rand_gen_gp_scale", 0.5, 10.0, use_log=True)
            self.freqs_ = input_tfm @ self.freqs_

        self.bias_ = 2 * np.pi * torch.rand(1, self.n_frequencies, device=x.device)
        self.weights_ = torch.randn(self.n_frequencies, self.out_features, device=x.device) / np.sqrt(
            self.n_frequencies
        )  # could use RandomMatrix here but it's expensive

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x @ self.freqs_ + self.bias_) @ self.weights_


class RandomLinearFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        # RandomMatrix is not a symmetric distribution and assumes the input to be on the right side
        # (e.g., for RandomWeightMatrix), so we transpose
        self.matrix_ = RandomMatrix(self.context).sample(1, self.out_features, self.in_features).squeeze(0).t()

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix_


class RandomQuadraticFunction(RandomFunction):
    """
    Random quadratic function (avoids quadratic complexity in the input dimension by subsampling to 20 features).
    """
    def _fit(self, x: torch.Tensor):
        max_n_features = 20
        if x.shape[1] > max_n_features:
            self.idxs_ = torch.randperm(x.shape[1])[:max_n_features]
        else:
            self.idxs_ = torch.arange(x.shape[1])
        self.n_feat_ = len(self.idxs_) + 1  # +1 for constant term
        self.tensor_3d_ = RandomMatrix(self.context).sample(self.out_features, self.n_feat_, self.n_feat_)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        # append ones to get the affine + linear part
        x = torch.cat([x[:, self.idxs_], torch.ones(x.shape[0], 1)], dim=-1)
        return torch.einsum("oij,bi,bj->bo", self.tensor_3d_, x, x)


class RandomEMAssignmentFunction(RandomFunction):
    """
    Function roughly inspired by soft cluster assignment in the EM algorithm.
    """
    def _fit(self, x: torch.Tensor):
        # need to have at least two outputs, otherwise softmax is a constant function
        n_ind = self.sampler.randint("random_em_components", 2, max(16, 2 * self.out_features) + 1, use_log=True)
        self.x_ind_ = x[torch.randint(x.shape[0], size=(n_ind,))]
        self.x_ind_ += 1.0 * torch.randn_like(self.x_ind_)
        # self.stds_ = RandomWeights(self.context).sample(1, n_ind).squeeze(0) + 1e-30
        self.stds_ = torch.exp(self.sampler.numerical("gen_em_assignment_std_factor", 0.0, 1.0) * torch.randn(n_ind))
        # todo: technically we would have to multiply this by the dimension?
        self.consts = -torch.log(2 * torch.pi * self.stds_[None, :] ** 2) / 2
        self.out_func_ = RandomLinearFunction(self.context, in_features=n_ind, out_features=self.out_features)
        self.p_ = self.sampler.numerical("gen_em_assignment_p_inv", 1.0, 4.0, use_log=True)
        self.q_ = self.sampler.numerical("gen_em_assignment_q", 1.0, 2.0)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.x_ind_, p=self.p_)
        logits = self.consts - torch.clamp(dists / self.stds_[None, :], min=0.0) ** self.q_
        result = torch.softmax(logits, dim=-1)
        return self.out_func_(result)


class RandomProductFunction(RandomFunction):
    """
    Product of two (cheap) random functions.
    """
    def _fit(self, x: torch.Tensor):
        self.fcts_ = [
            CheapRandomFunction(self.context, in_features=self.in_features, out_features=self.out_features)
            for _ in range(2)
        ]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcts_[0](x) * self.fcts_[1](x)


# =====================================================================
# Physics-informed random functions (manufacturing-relevant)
#
# Each function picks 1-2 (or a variable number of) input columns, maps
# them via a sigmoid into a physically plausible positive range (since
# incoming columns are standardized, roughly N(0, 1)-distributed), and
# applies a classical manufacturing / materials-science formula with
# randomized physical constants. The resulting scalar per sample is then
# broadcast to `out_features` dimensions via a RandomLinearFunction, the
# same mechanism used by RandomDiscretizationFunction and RandomEMAssignmentFunction
# above. All of these functions are closed-form (no recursion), so they
# are safe to use in CheapRandomFunction as well.
# =====================================================================

def _pick_col_idxs(x: torch.Tensor, k: int) -> torch.Tensor:
    """Randomly picks k column indices from x (with replacement if in_features < k)."""
    return torch.randint(0, x.shape[1], (k,), device=x.device)


class RandomArrheniusFunction(RandomFunction):
    """
    Arrhenius equation: rate = A * exp(-Ea / (R*T)).
    Relevant for thermally activated processes: curing, diffusion, chemical
    reaction rates, temperature-driven material fatigue.
    """
    def _fit(self, x: torch.Tensor):
        self.t_idx_ = _pick_col_idxs(x, 1)
        self.t_offset_ = 300.0  # Kelvin baseline
        self.t_scale_ = self.sampler.numerical("arrhenius_t_scale", 50.0, 500.0, use_log=True)
        self.ea_over_r_ = self.sampler.numerical("arrhenius_ea_over_r", 500.0, 15000.0, use_log=True)
        self.log_a_ = float(np.log(self.sampler.numerical("arrhenius_a", 1e-3, 1e3, use_log=True)))
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        T = self.t_offset_ + self.t_scale_ * torch.sigmoid(x[:, self.t_idx_[0]])
        log_rate = self.log_a_ - self.ea_over_r_ / T
        return self.out_proj_(log_rate[:, None])


class RandomArchardWearFunction(RandomFunction):
    """
    Archard's wear law: W = K * F * d (K encodes hardness).
    Relevant for tool wear, sliding/friction wear, bearing wear.
    """
    def _fit(self, x: torch.Tensor):
        self.idxs_ = _pick_col_idxs(x, 2)
        self.f_scale_ = self.sampler.numerical("archard_f_scale", 0.5, 20.0, use_log=True)
        self.d_scale_ = self.sampler.numerical("archard_d_scale", 0.5, 20.0, use_log=True)
        self.k_ = self.sampler.numerical("archard_k", 1e-5, 1e-2, use_log=True)
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        F = self.f_scale_ * torch.sigmoid(x[:, self.idxs_[0]])
        d = self.d_scale_ * torch.sigmoid(x[:, self.idxs_[1]])
        wear = self.k_ * F * d
        return self.out_proj_(wear[:, None])


class RandomHollomonHardeningFunction(RandomFunction):
    """
    Hollomon hardening law: sigma = K * epsilon^n.
    Relevant for forming processes (deep drawing, rolling, forging).
    """
    def _fit(self, x: torch.Tensor):
        self.eps_idx_ = _pick_col_idxs(x, 1)
        self.eps_scale_ = self.sampler.numerical("hollomon_eps_scale", 0.01, 1.0, use_log=True)
        self.strength_coef_ = self.sampler.numerical("hollomon_k", 200.0, 2000.0, use_log=True)
        self.hardening_exp_ = self.sampler.numerical("hollomon_n", 0.05, 0.5, use_log=True)
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps_scale_ * torch.sigmoid(x[:, self.eps_idx_[0]]) + 1e-4
        sigma = self.strength_coef_ * eps ** self.hardening_exp_
        return self.out_proj_(sigma[:, None])


class RandomBasquinFatigueFunction(RandomFunction):
    """
    Basquin equation (S-N curve): sigma_a = sigma_f' * (2N)^b.
    Relevant for fatigue strength / lifetime prediction.
    """
    def _fit(self, x: torch.Tensor):
        self.n_idx_ = _pick_col_idxs(x, 1)
        self.n_offset_ = 1e2
        self.n_scale_ = self.sampler.numerical("basquin_n_scale", 1e2, 1e6, use_log=True)
        self.fatigue_coef_ = self.sampler.numerical("basquin_sigma_f", 300.0, 1500.0, use_log=True)
        self.fatigue_exp_ = -self.sampler.numerical("basquin_b", 0.05, 0.15, use_log=True)
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        N = self.n_offset_ + self.n_scale_ * torch.sigmoid(x[:, self.n_idx_[0]])
        sigma_a = self.fatigue_coef_ * (2 * N) ** self.fatigue_exp_
        return self.out_proj_(sigma_a[:, None])


class RandomFourierHeatFunction(RandomFunction):
    """
    Fourier's law of heat conduction (steady-state, 1D): q = k*A*(T1-T2)/L.
    Relevant for thermal processes: welding, heat treatment, casting.
    """
    def _fit(self, x: torch.Tensor):
        self.idxs_ = _pick_col_idxs(x, 2)
        self.t_offset_ = 20.0
        self.t1_scale_ = self.sampler.numerical("fourier_t1_scale", 10.0, 300.0, use_log=True)
        self.t2_scale_ = self.sampler.numerical("fourier_t2_scale", 10.0, 300.0, use_log=True)
        self.conductivity_ = self.sampler.numerical("fourier_k", 0.1, 400.0, use_log=True)
        self.area_ = self.sampler.numerical("fourier_area", 1e-4, 1.0, use_log=True)
        self.length_ = self.sampler.numerical("fourier_length", 1e-3, 1.0, use_log=True)
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        T1 = self.t_offset_ + self.t1_scale_ * torch.sigmoid(x[:, self.idxs_[0]])
        T2 = self.t_offset_ + self.t2_scale_ * torch.sigmoid(x[:, self.idxs_[1]])
        q = self.conductivity_ * self.area_ * (T1 - T2) / self.length_
        return self.out_proj_(q[:, None])


class RandomTaylorToolWearFunction(RandomFunction):
    """
    Taylor's tool life equation: V * T^n = C  =>  T = (C/V)^(1/n).
    Relevant for tool life in machining/turning/milling.
    """
    def _fit(self, x: torch.Tensor):
        self.v_idx_ = _pick_col_idxs(x, 1)
        self.v_scale_ = self.sampler.numerical("taylor_v_scale", 10.0, 500.0, use_log=True)
        self.taylor_c_ = self.sampler.numerical("taylor_c", 50.0, 500.0, use_log=True)
        self.taylor_n_ = self.sampler.numerical("taylor_n", 0.1, 0.4, use_log=True)
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        V = self.v_scale_ * torch.sigmoid(x[:, self.v_idx_[0]]) + 1.0
        log_T = (np.log(self.taylor_c_) - torch.log(V)) / self.taylor_n_
        return self.out_proj_(log_T[:, None])


class RandomNewtonCoolingFunction(RandomFunction):
    """
    Newton's law of cooling: T(t) = T_env + (T0 - T_env) * exp(-k*t).
    Relevant for cooling processes: casting, heat treatment, weld cooling.
    """
    def _fit(self, x: torch.Tensor):
        self.idxs_ = _pick_col_idxs(x, 2)
        self.t0_offset_ = 20.0
        self.t0_scale_ = self.sampler.numerical("newton_t0_scale", 50.0, 300.0, use_log=True)
        self.time_scale_ = self.sampler.numerical("newton_time_scale", 1.0, 500.0, use_log=True)
        self.t_env_ = self.sampler.numerical("newton_t_env", 10.0, 30.0, use_log=True)
        self.k_ = self.sampler.numerical("newton_k", 1e-3, 1.0, use_log=True)
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        T0 = self.t0_offset_ + self.t0_scale_ * torch.sigmoid(x[:, self.idxs_[0]])
        t = self.time_scale_ * torch.sigmoid(x[:, self.idxs_[1]])
        T = self.t_env_ + (T0 - self.t_env_) * torch.exp(-self.k_ * t)
        return self.out_proj_(T[:, None])


class RandomToleranceStackupFunction(RandomFunction):
    """
    Statistical tolerance stack-up (root-sum-square): sigma_total = sqrt(sum(sigma_i^2)).
    Relevant for dimensional accuracy in multi-stage manufacturing processes.
    """
    def _fit(self, x: torch.Tensor):
        n_dims = self.sampler.randint("tol_stackup_n_dims", 2, min(x.shape[1], 8), use_log=True)
        self.idxs_ = _pick_col_idxs(x, n_dims)
        self.sigma_scale_ = self.sampler.numerical("tol_stackup_sigma_scale", 0.01, 1.0, use_log=True)
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        sigmas = self.sigma_scale_ * torch.sigmoid(x[:, self.idxs_])
        total = torch.sqrt((sigmas ** 2).sum(dim=-1) + 1e-12)
        return self.out_proj_(total[:, None])


class RandomPrestonFunction(RandomFunction):
    """
    Preston's equation: MRR = Kp * P * V.
    Relevant for material removal rate in polishing/lapping/CMP.
    """
    def _fit(self, x: torch.Tensor):
        self.idxs_ = _pick_col_idxs(x, 2)
        self.p_scale_ = self.sampler.numerical("preston_p_scale", 0.5, 10.0, use_log=True)
        self.v_scale_ = self.sampler.numerical("preston_v_scale", 0.5, 10.0, use_log=True)
        self.kp_ = self.sampler.numerical("preston_kp", 1e-4, 1e-1, use_log=True)
        self.out_proj_ = RandomLinearFunction(self.context, 1, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        P = self.p_scale_ * torch.sigmoid(x[:, self.idxs_[0]])
        V = self.v_scale_ * torch.sigmoid(x[:, self.idxs_[1]])
        mrr = self.kp_ * P * V
        return self.out_proj_(mrr[:, None])
