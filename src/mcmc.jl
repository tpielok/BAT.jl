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
