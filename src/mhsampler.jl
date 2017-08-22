# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct MHState{
    Q<:AbstractProposalDist,
    S<:MCMCSample,
} <: AbstractMCMCState
    pdist::Q

    current_sample::S
    proposed_sample::S
    proposal_accepted::Bool
    current_nreject::Int64

    rng::R

    nsteps::Int64
    naccept::Int64
end


struct MetropolisHastings <: MCMCAlgorithm{MHState} end
export MetropolisHastings



acceptance_ratio(state::MHState) = chain.state.naccept / state.nsteps




mcmc_iterate(
    callback,
    chain::MCMCChain{<:MetropolisHastings},
    exec_context::ExecContext = ExecContext();
    granularity::Int = 1
    max_nsamples::Int = 1
    max_nsteps::Int = 1000
)
    target = chain.target
    state = chain.state
    rng = chain.rng

    tfunc = target.tfunc
    bounds = target.bounds

    pdist = state.pdist
    current_sample = state.current_sample
    proposed_sample = state.proposed_sample

    current_params = current_sample.params
    current_log_value = current_sample.log_value

    proposed_params = proposed_sample.params

    if state.proposal_accepted
        copy!(current_sample, proposed_sample)
        state.current_nreject = 0
        state.proposal_accepted = false
    end

    accepted = false
    nsteps = 0
    nsamples = 0
    while nsamples < max_nsamples && nsteps < max_nsteps
        # TODO: mofify/tag counter(s) for counter-based rng
        proposal_rand!(rng, pdist, proposed_params, current_params)
        apply_bounds!(proposed_params, bounds)

        proposed_log_value = target_logval(tfunc, params_next, exec_context)

        log_tp_fwd = proposal_logpdf(, params_next, current_params)

        # TODO: mofify/tag counter(s) for counter-based rng
        accepted = rand(rng) < exp(log_value_next - log_value_last)

        nsteps += 1
        state.nsteps += 1
        if accepted
            state.proposal_accepted = true
            state.naccept += 1

            sample.weight = state.current_nreject + 1

            nsamples += 1
            chain.nsamples += 1
        else
            state.current_nreject += 1
            ...
        end

        if accepted || (granularity > 2)
            callback(chain)
        end
    end
end



current_sample.weight += 1


function mcmc_update!(state::MHState, new_params::Vector{<:Real}, new_log_value::Real, log_tpr::Real)::Bool
    rng = state.rng
    isnan(log_value) && error("Encountered NaN log_value")
    accepted = rand(rng) < exp(log_value - state.log_value)
    if accepted
        copy!(state.params, params)
        state.log_value = log_value
        state.nsamples += 1
        state.multiplicity = 1
    else
        state.multiplicity += 1
    end
    accepted
end








function mh_propose_eval!(
    params_new::AbstractVector{P},
    target::AbstractTargetFunction,
    pdist::AbstractProposalDist,
    params_old::AbstractVector{P},
    bounds::AbstractParamBounds,
    executor::SerialExecutor
)::Real where {P<:Real}
    proposal_rand!(executor.rng, pdist, params_new, params_old)
    apply_bounds!(params_new, bounds)
    target_logval(target, params_new, executor.ec)
end



mutable struct MHChainState{
    P<:Real,
    T<:Real,
    S<:MCMCSubject,
    R<:AbstractRNG
} <: AbstractMCMCState
    subject::S
    rng::R
    params::Vector{P}
    log_value::T
    nsamples::Int64
    multiplicity::Int

end






function mcmc_step!(state::MHChainState, exec_context::ExecContext = ExecContext())
    rng = state.rng
    exec_context = state.exec_context
    params_old = state.params
    params_new = similar(params_old) # TODO: Avoid memory allocation

    # TODO: mofify/tag counter(s) for counter-based rng
    proposal_rand!(rng, state.pdist, params_new, params_old)
    apply_bounds!(params_new, state.bounds)
    log_value_new = target_logval(state.target, params_new, exec_context)

    # TODO: mofify/tag counter(s) for counter-based rng
    mcmc_update!(state, params_new, log_value_new, rng)

    state
end


#=
function mcmc_step(states::AbstractVector{MHChainState{P, R}}, exec_context::ExecContext = ExecContext()) where {P,R}
    rng = state.rng
    exec_context = state.exec_context

    # TODO: Avoid memory allocation:
    params_old = hcat((s.params for s in states)...)
    params_new = similar(params_old)
    log_values_new = Vector{R}(length(states))

    proposal_rand!(rng, state.pdist, params_new, params_old)
    apply_bounds!(params_new, state.bounds)
    log_values_new = target_logval!(log_values_new, state.target, params_new, exec_context)

    for i in eachindex(log_values_new)
        # TODO: Run multithreaded if length(states) is large?
        p_new = view(params_new, :, i) # TODO: Avoid memory allocation
        push!(states[i], p_new, log_values_new[i], rng)
    end
    states
end
=#





#=



function MHSampler(
    log_f::Any, # target function, log_f(params::AbstractVector, aux_values::)
    param_bounds::Union{AbstractParamBounds, Vector{NTuple{2}}},
    q::AbstractProposalFunction = MvNormal(...),     # proposal distribution
    tune_q::Any # tune_q(q, history::MCSamplerOutput) -> q', tune_q may mutate it's state
    callback::Any # sampling loop callback: callback(state)
    ;
    n_chains::Integer = 1,
    max_iterations::Nullable{Int} = Nullable{Int}(),
    max_runtime::Nullable{Float64} = Nullable{Float64}()
)
    
    bounds, transformed_log_f = _param_bounds(param_bounds)
end


abstract AbstractMCSamplerOutput

# Single chain output (same type after merge?):
mutable struct MCSamplerOutput{T,Arr<:AbstractArray} <: AbstractMCSamplerOutput
    log_f::Arr{T,1} # Target function may be factorized
    weight::Arr{T,1}
    params::Arr{T, 2}
    aux::Arr{T, 2} # Auxiliary values like likelihood, prior, observables, etc.
end


mutable struct SigmaDistTuner{T}
    iteration::Int # initially 1
    lambda::T # e.g. 0.5
    scale::T # initially 2.38^2/ndims
end

function tuning_init(::Type{StudentTProposalFunction}, tuner::SigmaDistTuner, bounds::HyperCubeBounds)
    flat_var = (bounds.to - bounds.from).^2 / 12
    ndims = length(flat_var)
    new_Σ_unscal_pd = PDiagMat(flat_var)
    tuner.scale = 2.38^2/ndims
    StudentTProposalFunction(new_Σ_unscal_pd * tuner.scale)
end

function tuning_adapt(tuner::SigmaDistTuner, q::StudentTProposalFunction, history::MCSamplerOutput)
    t = tuner.iteration
    λ = tuner.lambda
    c = tuner.scale
    Σ = q.Σ

    S = cov(history.params, 1)
    a_t = 1/t^λ
    new_Σ_unscal = (1 - a_t) * (Σ/c) + a_t * S
    new_Σ_unscal_pd = PDMat(cholfact(Hermitian(new_Σ_unscal_pd)))

    α_min = 0.15
    α_max = 0.35

    c_min = 1e-4
    c_max = 1e2

    β = 1.5

    α = 1 / mean(history.weight) # acceptance

    if α > α_max && c < c_max
        new_c = c * β
    elseif α < α_min && c > c_min
        new_c /=  c / β
    else
        new_c = c
    end

    tuner.iteration += 1
    tuner.scale = new_c

    StudentTProposalFunction(new_Σ_unscal_pd * tuner.scale)
end


# User:

sampler = MHSampler(x -> -x^2/2, [(-4, 4)], n_chains = 4)
output = rand(sampler, 1000000) = ...::SamplerOutput

=#
