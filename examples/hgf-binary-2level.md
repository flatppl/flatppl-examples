# Two-level binary HGF with softmax choices

The [FlatPPL model](hgf-binary-2level.flatppl) ports the approximate belief filter in
HierarchicalGaussianFiltering.jl's `binary_2level` model. The [Julia model](hgf-binary-2level.jl)
uses the original package at revision `c0404294082fdc358d680a04a63eb9bfa9182ccc`.

| Part | Julia | FlatPPL |
|---|---|---|
| Filter | `premade_hgf("binary_2level", config)` | `scan(update, s0, u_data)` |
| Initial belief | Default mean 0, precision 1 | `record(mean = 0, precision = 1)` |
| Choice | `HGFSoftmax(target_state = :xprob_posterior_mean)` | `Bernoulli.(invlogit.(means ./ action_noise))` |
| Likelihood | Sum of original action-distribution `logpdf` values | `logdensityof(L, pars)` |
| Posterior | Likelihood plus two Normal prior scores | `logdensityof(posterior, pars)` |

The six outcomes `[1, 0, 1, 1, 0, 0]` are fixed inputs. The choices
`[1, 0, 1, 0, 1, 0]` are scored observations. Choices use the updated log-odds
mean, explicitly overriding the action model's default target.

For previous mean $m$, precision $\pi$, and outcome $u$:

$$V^- = 1/\pi + e^\omega, \qquad p^- = \operatorname{logistic}(m),$$
$$\pi' = 1/V^- + p^-(1-p^-), \qquad m' = m + (u-p^-)/\pi',$$
$$q = \operatorname{logistic}(m'/e^{\ell}), \qquad a \sim \mathrm{Bernoulli}(q).$$

`scan` carries the mean and precision record through the outcomes. This update
is deterministic. `markovchain` and `kscan` instead describe stochastic transitions.
The probability node uses classic updates: precision updates before the mean.
Drift is zero, coupling and autoconnection are one, and time steps are one.
The priors are `Normal(-2, 1)` on `omega` and `Normal(0, 0.5)` on `log_noise`.
The posterior density uses those coordinates, so no noise-transform Jacobian applies.
This example implements the package's approximate filter rather than exact latent-state inference.

## Score comparison

The original Julia package, FlatPPL JS, and an independent variance-form recurrence
agree within `2e-12` at all three parameter points. Every trial's mean, precision,
and choice probability also agrees. The stored reference uses Julia 1.13.1 and HGF 0.7.0.

| omega | action noise | Julia / FlatPPL log likelihood | Julia / FlatPPL log posterior |
|---:|---:|---:|---:|
| -2 | 1 | -3.970393704144 | -5.115123589993 |
| -3 | 0.7 | -3.975463546691 | -5.874627463808 |
| -0.5 | 1.4 | -3.945275969554 | -6.441432987437 |

From the repository root, run the checker with Node 24 or newer and a current
`flatppl-js` checkout. The engine must include the structured scan fixes in
[PR 327](https://github.com/flatppl/flatppl-js/pull/327):

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js node reference/hgf-binary-2level/verify.mjs
```

The four checks cover the independent recurrence and scores, outcome order,
choice conditioning, scores at actual posterior draws, and the stored Julia reference.
Rust scoring has not been verified.

To regenerate the Julia reference, use Julia 1.11 or newer and the isolated,
revision-pinned project:

```sh
julia --project=reference/hgf-binary-2level -e 'using Pkg; Pkg.instantiate()'
julia --project=reference/hgf-binary-2level examples/hgf-binary-2level.jl
```

The Julia script writes `reference/hgf-binary-2level/julia-reference.json`.
The project pins the HGF source and direct dependency versions. It resolves
transitive dependencies when instantiated; its local manifest is ignored.

## Original sources

- [Binary two-level constructor](https://github.com/ComputationalPsychiatry/HierarchicalGaussianFiltering.jl/blob/c0404294082fdc358d680a04a63eb9bfa9182ccc/src/premade_models/premade_hgfs/premade_binary_2level.jl)
- [Continuous belief update](https://github.com/ComputationalPsychiatry/HierarchicalGaussianFiltering.jl/blob/c0404294082fdc358d680a04a63eb9bfa9182ccc/src/update_hgf/node_updates/continuous_state_node.jl)
- [Softmax action model](https://github.com/ComputationalPsychiatry/HierarchicalGaussianFiltering.jl/blob/c0404294082fdc358d680a04a63eb9bfa9182ccc/src/premade_models/premade_action_models/premade_softmax.jl)
