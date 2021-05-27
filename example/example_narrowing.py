import functools
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "../")

from bayes_optim import EI, NarrowingBO, RealSpace, SearchSpace
from bayes_optim.bayes_opt import partial_argument
from bayes_optim.surrogate import GaussianProcess, trend

np.random.seed(123)
dim = 25  # Should be greater than 2
lb, ub = -5, 15


def specifiy_dummy_vars(d_eff: int):
    def wrapper(func):
        @functools.wraps(func)
        def inner(x):
            x = np.asarray(x[:d_eff])
            return func(x)

        return inner

    return wrapper


@specifiy_dummy_vars(2)
def branin(x):
    """Branin function (https://www.sfu.ca/~ssurjano/branin.html)
    Global minimum 0.397881 at (-Pi, 12.275), (Pi, 2.275), and (9.42478, 2.475)
    """
    x1 = x[0]
    x2 = x[1]
    g_x = (
        (x2 - (5.1 * x1 ** 2) / (4 * np.pi ** 2) + 5 * x1 / np.pi - 6) ** 2
        + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1)
        + 10
    )
    return np.abs(0.397881 - g_x)


@specifiy_dummy_vars(2)
def fitness(x):
    return np.sum([x[1] ** x[0] for x in enumerate(x)])


def compute_func_std(
    funcs: List[callable], search_space: SearchSpace, N: int = 1e3, p: int = 2
) -> Tuple[float, int]:
    """Standard deviation of a set of function, where the mean function is taken as the pointwise
    mean and the distance between functions is computed by approximating the L^p norm of
    the difference of them.

    Parameters
    ----------
    funcs : List[callable]
        a list of functions
    search_space : SearchSpace
        the domain of the integration
    N : int, optional
        sample size for the approximation, by default 1e3
    p : int, optional
        parameter p as in L^p norm, by default 2

    Returns
    -------
    Tuple[float, int]
        (approximated std., index of the function which gives the largest acquisition value)
    """
    samples = search_space.sample(N)
    X = np.empty((len(funcs), N))
    for i, func in enumerate(funcs):
        X[i, :] = list(map(func, samples))

    mean_func = np.mean(X, axis=0)  # the mean function
    norms = [np.sum((X[i, :] - mean_func) ** 2.0) for i in range(len(funcs))]
    return np.std(norms), np.argmax(np.max(X, axis=1))


def acquisition_var_selector(
    model, inactive: Dict, search_space: SearchSpace
) -> Tuple[str, float]:
    """A variable is deemed unimportant if the marginal acquisition function (by fixing the value
    of this variable) does not change much when varying the value of it.

    Parameters
    ----------
    model : Surrogate
        The current surrogate model
    inactive : Dict
        A dictionary of inactive variables and their default values
    search_space : SearchSpace
        The original search space

    Returns
    -------
    Tuple[str, float]
        (name of the selected variable, its default value)
    """
    mask = np.array([v in inactive.keys() for v in search_space.var_name])
    values = list(inactive.values())
    score = np.empty(len(search_space) - len(inactive))
    best_v = np.empty(len(search_space) - len(inactive))
    var_names = np.array([v for v in search_space.var_name if v not in inactive.keys()])

    for i, var in enumerate(search_space.data):
        if var.name not in inactive.keys():
            masks_ = mask.copy()  # deactive ith variable
            masks_[i] = True
            idx = np.nonzero(~masks_)[0]
            # take 10 evenly-spaced values for this variable and slice original acquisition
            # function at those values
            V = np.linspace(var.bounds[0], var.bounds[1], 10)
            pre = sum(masks_[:i])
            funcs = []
            for v in V:
                values_ = values[:pre] + [v] + values[pre:]
                # NOTE: we take the EI criterion for now
                funcs.append(partial_argument(EI(model), masks_, values_))

            score[i], idx = compute_func_std(funcs, search_space[idx])
            best_v[i] = V[idx]

    _ = np.argmin(score)
    return var_names[_], best_v[_]


def corr_fsel(data, model, active_fs):
    """Pearson correlation-based feature selection
    Considering the points evaluated and the active features,
    the correlation is calculated. Then, the feature with the smallest
    correlation is discarded.
    """
    if len(active_fs) == 1:
        return {}
    df = pd.DataFrame(data.tolist(), columns=data.var_name.tolist())
    df["f"] = data.fitness
    df = df[active_fs + ["f"]]
    cor = df.corr()
    cor_fitness = abs(cor["f"])
    # TODO is the name of the variable influencing the sort?
    fs = cor_fitness.sort_values(ascending=True).index[0]
    # TODO set the value for the discarded feature
    return fs, 0


def mean_improvement(data, model, metrics):
    """Mean fitness improvement criteria.

    The mean fitness is calculated for all the available data
    points. If the mean is improving (ge) in relation to the
    previous one (i.e., metrics in the input). Then, a True value
    is returned, along with the new calculated mean.
    """
    _mean = np.mean([x.fitness for x in data])
    if ("mean" in metrics and _mean >= metrics["mean"]) or "mean" not in metrics:
        return True, {"mean": _mean}
    return False, {"mean": _mean}


space = RealSpace([lb, ub]) * dim
mean = trend.constant_trend(dim, beta=None)
thetaL = 1e-10 * (ub - lb) * np.ones(dim)
thetaU = 10 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(
    theta0=theta0,
    thetaL=thetaL,
    thetaU=thetaU,
    nugget=0,
    noise_estim=False,
    optimizer="BFGS",
    wait_iter=3,
    random_start=dim,
    likelihood="concentrated",
    eval_budget=100 * dim,
)

opt = NarrowingBO(
    search_space=space,
    obj_fun=branin,
    model=model,
    DoE_size=30,
    max_FEs=500,
    verbose=True,
    n_point=1,
    minimize=True,
    narrowing_fun=corr_fsel,
    narrowing_improving_fun=mean_improvement,
    narrowing_FEs=5,
)

print(opt.run())
