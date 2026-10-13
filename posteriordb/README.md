# posteriorDB

FlatPPL ports of sixteen models from
[posteriorDB revision 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4).
The original JSON files live in [`data/`](data). Each model loads its data
relative to its own file with `load_data`.

The playground lists the fourteen models with at most ten parameters. The larger
`GLMM_Poisson_model` and `Mh_model` remain available in this repository.

| Model | Description | Parameters |
|---|---|---:|
| [Rate_1_model](Rate_1_model.flatppl) | Binomial rate | 1 |
| [GLM_Binomial_model](GLM_Binomial_model.flatppl) | Binomial regression with a quadratic trend | 3 |
| [GLM_Poisson_model](GLM_Poisson_model.flatppl) | Poisson regression with a cubic trend | 4 |
| [GLMM_Poisson_model](GLMM_Poisson_model.flatppl) | Poisson regression with year effects | 45 |
| [kidscore_interaction](kidscore_interaction.flatppl) | Regression with an interaction | 5 |
| [blr](blr.flatppl) | Bayesian linear regression | 6 |
| [arK](arK.flatppl) | Autoregression on five observed lags | 7 |
| [eight_schools_centered](eight_schools_centered.flatppl) | Centered hierarchical Normal model | 10 |
| [Mh_model](Mh_model.flatppl) | Capture-recapture with individual effects | 388 |
| [gp_regr](gp_regr.flatppl) | Gaussian process regression | 3 |
| [dogs_hierarchical](dogs_hierarchical.flatppl) | Avoidance learning with observed trial histories | 2 |
| [low_dim_gauss_mix](low_dim_gauss_mix.flatppl) | Two-component Gaussian mixture with ordered means | 5 |
| [M0_model](M0_model.flatppl) | Capture-recapture with a shared capture probability | 2 |
| [Mb_model](Mb_model.flatppl) | Capture-recapture with a behavioral response | 3 |
| [arma11](arma11.flatppl) | ARMA(1,1) with observed history | 4 |
| [Survey_model](Survey_model.flatppl) | Binomial surveys with a shared, marginalized population size | 1 |

Every model exposes `Pi_prior`, `Pi_post`, `Pi_post_domain`, and a starting
point `Pi_post_input`. For example, save this query beside the models:

```flatppl
m = load_module("eight_schools_centered.flatppl")
point = elementof(m.Pi_post_domain)
inputs = point
outputs = logdensityof(m.Pi_post, point)
```

Compile the query with `flatppl stablehlo query.flatppl`, or load it through
[flatppl-python](https://flatppl.org/flatppl-python/).
`query.compile_transformed(logdensity=True)` supplies the unconstrained
target and both coordinate transforms for BlackJAX, including the Jacobian
inside StableHLO. The centered eight-schools model has funnel geometry.
The supplied starting points do not establish sampler convergence.

In the playground, select RAM, slice or ESS for `kidscore_interaction`.
Its unbounded `Flat` coefficient priors have no sampler, so prior-sampling
methods such as IS, AMIS and SMC refuse it. The other fifteen models have
sampleable priors, including Uniform priors on fixed finite intervals.

AMIS works well for the Gaussian mixture. RAM can remain in a secondary mode
from prior initialization, so compare chains before trusting its summaries.

These preserve the original posteriorDB posterior laws, including the GP
covariance's `sigma` diagonal term. They do not adopt
Wren's additional Kidscore/GLMM priors or its wider GLMM coefficient bound.

FlatPPL normalizes the half-Normal and half-Cauchy priors. The resulting
log density exceeds Stan's by `log(2)` for `arK`, `blr`,
`eight_schools_centered`, `kidscore_interaction`, and `arma11`, and by `2*log(2)` for
`gp_regr`. Normalizing the bounded Uniform priors adds
`-log(40) - 3*log(20)` for `GLM_Poisson_model` and `-log(5)` for `Mh_model`.
The ordered mixture adds `3*log(2)` from its ordered means and two half-Normal scales.
These fixed offsets leave the posterior laws and gradients unchanged.
All other offsets are zero.

The [testsuite corpus](https://github.com/flatppl/flatppl-testsuite/tree/main/corpora/posteriordb)
holds original Stan sources and independent density/gradient checks at
three points per model. Data retain posteriorDB's [BSD license](LICENSE).
