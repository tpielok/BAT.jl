# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

@testset "rng" begin
    @testset "Philox4xSeed" begin
        philox = @inferred Philox4xSeed()
        @test typeof(philox) <: AbstractRNGSeed
        @test typeof(philox.seed) <: NTuple{2, UInt}
        #philox = @inferred Philox4xSeed((2,3))

    end

end
