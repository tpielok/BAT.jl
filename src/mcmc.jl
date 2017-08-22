# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type AbstractMCMCState end


mutable struct MCMCSubject{
    F<:AbstractTargetFunction,
    Q<:ProposalDist,
    B<:AbstractParamBounds
}
    target::F
    pdist::Q
    bounds::B
end

export MCMCSubject

Base.length(subject::MCMCSubject) = length(subject.bounds)


mutable struct MCMCSample{
    P<:Real,
    T<:Real,
    W<:Real
} <: AbstractMCMCState
    params::Vector{P}
    log_value::T
    weight::W
    nsamples::Int64
end

Base.length(sample::MCMCSample) = length(sample.params)


function MCMCSample(
    P<:Real,
    T<:Real,
    W<:Real
) where {
    P<:Real,
    T<:Real,
    S<:MCMCSubject,
    R<:AbstractRNG
}
    MHChainState{P,T,S,R}(
        subject,
        rng,
        params,
        log_value,
        nsamples,
        multiplicity
    )
end


"""
    mcmc_step(state::AbstractMCMCState, rng::AbstractRNG, exec_context::ExecContext = ExecContext())
    mcmc_step(states::AbstractVector{<:AbstractMCMCState}, rng::AbstractRNG, exec_context::ExecContext = ExecContext()) where {P,R}
"""
function  mcmc_step end
export mcmc_step


"""
    exec_context(state::AbstractMCMCState)
"""
function exec_context end
export exec_context
