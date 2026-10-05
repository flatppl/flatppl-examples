using HierarchicalGaussianFiltering
using Distributions
using JSON3

const HGF_SOURCE_REV = "c0404294082fdc358d680a04a63eb9bfa9182ccc"
const HGF_INPUTS = [1, 0, 1, 1, 0, 0]
const HGF_CHOICES = [1, 0, 1, 0, 1, 0]
const HGF_POINTS = [
    (; omega = -2.0, log_noise = 0.0),
    (; omega = -3.0, log_noise = log(0.7)),
    (; omega = -0.5, log_noise = log(1.4)),
]

function hgf_score(omega, log_noise; inputs = HGF_INPUTS, choices = HGF_CHOICES)
    length(inputs) == length(choices) || throw(DimensionMismatch("one choice per input"))
    hgf = premade_hgf(
        "binary_2level",
        Dict(("xprob", "volatility") => omega);
        verbose = false,
    )
    model = ActionModel(HGFSoftmax(;
        HGF = hgf,
        action_noise = exp(log_noise),
        target_state = :xprob_posterior_mean,
    ))
    agent = init_agent(model)
    means, precisions, probabilities, trial_scores = (Float64[] for _ in 1:4)
    for (input, choice) in zip(inputs, choices)
        # The same action-model call as ActionModels.simulate!, without sampling.
        distribution = agent.action_model(agent.model_attributes, input)
        belief = agent.model_attributes.submodel
        push!(means, get_states(belief, :xprob_posterior_mean))
        push!(precisions, get_states(belief, :xprob_posterior_precision))
        push!(probabilities, mean(distribution))
        push!(trial_scores, logpdf(distribution, choice))
    end
    likelihood = sum(trial_scores)
    posterior = likelihood + logpdf(Normal(-2, 1), omega) + logpdf(Normal(0, 0.5), log_noise)
    return (; omega, log_noise, means, precisions, probabilities,
        trialScores = trial_scores, likelihood, posterior)
end

function write_hgf_reference(path = joinpath(@__DIR__, "..", "reference", "hgf-binary-2level", "julia-reference.json"))
    reference = (;
        source_revision = HGF_SOURCE_REV,
        hgf_version = string(pkgversion(HierarchicalGaussianFiltering)),
        julia_version = string(VERSION),
        inputs = HGF_INPUTS,
        choices = HGF_CHOICES,
        cases = [hgf_score(point.omega, point.log_noise) for point in HGF_POINTS],
    )
    open(path, "w") do io
        JSON3.write(io, reference)
        write(io, '\n')
    end
    return reference
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_hgf_reference(isempty(ARGS) ? joinpath(@__DIR__, "..", "reference", "hgf-binary-2level", "julia-reference.json") : only(ARGS))
end
