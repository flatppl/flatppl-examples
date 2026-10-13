# posteriorDB

FlatPPL ports of ten models from
[posteriorDB revision 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4).
The original JSON files live in [`data/`](data). Each model loads its data
relative to its own file with `load_data`.

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

In the playground, select RAM, slice or ESS for `GLM_Poisson_model`,
`kidscore_interaction`, and `Mh_model`. Their `Flat` priors have no sampler,
so prior-sampling methods such as IS, AMIS and SMC refuse them.

These preserve the original posteriorDB targets, including bounded flat
priors and the GP covariance's `sigma` diagonal term. They do not adopt
Wren's additional Kidscore/GLMM priors or its wider GLMM coefficient bound.

FlatPPL normalizes the half-Normal and half-Cauchy priors. The resulting
log density exceeds Stan's by `log(2)` for `arK`, `blr`,
`eight_schools_centered`, and `kidscore_interaction`, and by `2*log(2)` for
`gp_regr`. These fixed offsets leave the posterior laws and gradients
unchanged. All other offsets are zero.

The [testsuite corpus](https://github.com/flatppl/flatppl-testsuite/tree/main/corpora/posteriordb)
holds original Stan sources and independent density/gradient checks at
three points per model. Data retain posteriorDB's [BSD license](LICENSE).
