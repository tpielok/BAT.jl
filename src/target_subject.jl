# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type AbstractTargetSubject end



mutable struct TargetSubject{
    F<:AbstractTargetDensity,
    B<:AbstractParamBounds
} <: AbstractTargetSubject
    tdensity::F
    bounds::B
end

export TargetSubject

Base.length(subject::TargetSubject) = length(subject.bounds)

target_function(subject::TargetSubject) = subject.tdensity
param_bounds(subject::TargetSubject) = subject.bounds
nparams(subject::TargetSubject) = nparams(subject.bounds)


#=

# ToDo:

mutable struct TransformedTargetSubject{
    SO<:AbstractTargetSubject,
    SN<:TargetSubject
} <: AbstractTargetSubject
   before::SO
   after::SN
   # ... transformation, Jacobi matrix of transformation, etc.
end

export TransformedTargetSubject

...

=#