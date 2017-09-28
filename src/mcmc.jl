# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCState end

abstract type MCMCAlgorithm{S<:AbstractMCMCState} end


abstract type AbstractMCMCSample end
export AbstractMCMCSample



mutable struct MCMCSample{
    P<:Real,
    T<:Real,
    W<:Real
} <: AbstractMCMCSample
    params::Vector{P}
    log_value::T
    weight::W
end

export MCMCSample


Base.length(s::MCMCSample) = length(s.params)

Base.similar(s::MCMCSample{P,T,W}) where {P,T,W} =
    MCMCSample{P,T,W}(oob(s.params), convert(T, NaN), zero(W))

import Base.==
==(A::MCMCSample, B::MCMCSample) =
    A.params == B.params && A.log_value == B.log_value && A.weight == B.weight


function Base.copy!(dest::MCMCSample, src::MCMCSample) 
    copy!(dest.params, src.params)
    dest.log_value = src.log_value
    dest.weight = src.weight
    dest
end


nparams(s::MCMCSample) = length(s)



struct MCMCChainInfo
    id::Int
    cycle::Int
    tuned::Bool
    converged::Bool
end

export MCMCChainInfo

MCMCChainInfo(id::Int, cycle::Int = 0) = MCMCChainInfo(id, cycle, false, false)


next_cycle(info::MCMCChainInfo) =
    MCMCChainInfo(info.id, info.cycle + 1, info.tuned, info.converged)

set_tuned(info::MCMCChainInfo, value::Bool) =
    MCMCChainInfo(info.id, info.cycle, value, info.converged)

set_converged(info::MCMCChainInfo, value::Bool) =
    MCMCChainInfo(info.id, info.cycle, info.tuned, value)



mutable struct MCMCChain{
    A<:MCMCAlgorithm,
    T<:AbstractTargetSubject,
    S<:AbstractMCMCState
}
    algorithm::A
    target::T
    state::S
    info::MCMCChainInfo
end

export MCMCChain


nparams(chain::MCMCChain) = nparams(chain.target)



abstract type AbstractMCMCStats end
export AbstractMCMCStats

Base.push!(stats::AbstractMCMCStats, chain::MCMCChain) = push!(stats, chain.state)



struct MCMCNullStats <: AbstractMCMCStats end
export MCMCNullStats

Base.push!(stats::MCMCNullStats, s::MCMCSample) = stats



struct MCMCBasicStats{L<:Real,P<:Real} <: AbstractMCMCStats
    param_stats::BasicMvStatistics{P,FrequencyWeights}
    logtf_stats::BasicUvStatistics{L,FrequencyWeights}
    mode::Vector{P}

    function MCMCBasicStats{L,P}(m::Integer) where {L<:Real,P<:Real}
        param_stats = BasicMvStatistics{P,FrequencyWeights}(m)
        logtf_stats = BasicUvStatistics{L,FrequencyWeights}()
        mode = Vector{P}(size(param_stats.mean, 1))

        new{L,P}(
            BasicMvStatistics{P,FrequencyWeights}(m),
            BasicUvStatistics{L,FrequencyWeights}(),
            fill(oob(P), m)
        )
    end
end

export MCMCBasicStats


MCMCBasicStats(chain::MCMCChain) = MCMCBasicStats(chain.state)


function Base.push!(stats::MCMCBasicStats, s::MCMCSample)
    push!(stats.param_stats, s.params, s.weight)
    if s.log_value > stats.logtf_stats.maximum
        stats.mode .= s.params
    end
    push!(stats.logtf_stats, s.log_value, s.weight)
    stats
end

nparams(stats::MCMCBasicStats) = stats.param_stats.m



struct MCMCSampleVector{P<:Real,T<:AbstractFloat,W<:Real} <: DenseVector{MCMCSample{P,T,W}}
    params::ExtendableArray{P, 2, 1}
    log_values::Vector{T}
    weights::Vector{W}
end

export MCMCSampleVector

function MCMCSampleVector(chain::MCMCChain)
    P = eltype(chain.state.current_sample.params)
    T = typeof(chain.state.current_sample.log_value)
    W = typeof(chain.state.current_sample.weight)

    m = size(chain.state.current_sample.params, 1)
    MCMCSampleVector(ExtendableArray{P}(m, 0), Vector{T}(0), Vector{W}(0))
end


Base.size(xs::MCMCSampleVector) = size(xs.log_values)

Base.getindex(xs::MCMCSampleVector{P,T,W}, i::Integer) where {P,T,W} =
    MCMCSample{P,T,W}(xs.params[:,i], xs.log_values[i], xs.weights[i])


function Base.push!(xs::MCMCSampleVector, x::MCMCSample)
    append!(xs.params, x.params)
    push!(xs.log_values, x.log_value)
    push!(xs.weights, x.weight)
    xs
end
