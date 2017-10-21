# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const standard_confidence_vals = [0.68, 0.95, 0.997]


function rectangle_path(lo::Vector{<:Real}, hi::Vector{<:Real})
    [
        lo[1] lo[2];
        hi[1] lo[2];
        hi[1] hi[2];
        lo[1] hi[2];
        lo[1] lo[2];
    ]
end


function err_ellipsis_path(μ::Vector{<:Real}, Σ::Matrix{<:Real}, confidence::Real = 0.68, npts = 256)
    σ_sqr, A = eig(Hermitian(Σ))
    σ = sqrt.(σ_sqr)
    ϕ = linspace(0, 2π, 100)
    σ_scaled = σ .* sqrt(invlogcdf(Chisq(2), log(confidence)))
    xy = hcat(σ_scaled[1] * cos.(ϕ), σ_scaled[2] * sin.(ϕ)) * [A[1,1] A[1,2]; A[2,1] A[2,2]]
    xy .+ μ'
end


@recipe function f(samples::DensitySampleVector, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    acc = find(x -> x > 0, samples.weight)
    rej = find(x -> x <= 0, samples.weight)

    base_markersize = get(d, :markersize, 1.5)
    seriestype = get(d, :seriestype, :scatter)

    plot_bounds = get(d, :bounds, true)
    delete!(d, :bounds)

    if seriestype == :scatter
        color = get(d, :seriescolor, :green)
        label = get(d, :label, isempty(rej) ? "samples" : "accepted")

        @series begin
            seriestype := :scatter
            label := label
            markersize := base_markersize * sqrt.(samples.weight[acc])
            markerstrokewidth := 0
            color := color
            xlabel --> "\$\\theta_$(pi_x)\$"
            ylabel --> "\$\\theta_$(pi_y)\$"
            (samples.params[pi_x, acc], samples.params[pi_y, acc])
        end

        if !isempty(rej)
            @series begin
                seriestype := :scatter
                label := "rejected"
                markersize := base_markersize
                markerstrokewidth := 0
                color := :red
                (samples.params[pi_x, rej], samples.params[pi_y, rej])
            end
        end
    elseif seriestype == :histogram2d
        @series begin
            seriestype := :histogram2d
            label --> "samples"
            xlabel --> "\$\\theta_$(pi_x)\$"
            ylabel --> "\$\\theta_$(pi_y)\$"
            weights := samples.weight[:]
            (samples.params[pi_x, :], samples.params[pi_y, :])
        end
    else
        error("seriestype $seriestype not supported")
    end

    nothing
end


@recipe function f(bounds::HyperRectBounds, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    vhi = bounds.vol.hi[[pi_x, pi_y]]; vlo = bounds.vol.lo[[pi_x, pi_y]]
    rect_xy = rectangle_path(vlo, vhi)
    # bext = 0.1 * (vhi - vlo)
    # xlims = (vlo[1] - bext[1], vhi[1] + bext[1])
    # ylims = (vlo[2] - bext[2], vhi[2] + bext[2])

    @series begin
        seriestype := :path
        label --> "bounds"
        linecolor --> :darkred
        linewidth --> 2
        linealpha --> 0.5
        # xlims --> xlims
        # ylims --> ylims
        (rect_xy[:,1], rect_xy[:,2])
    end

    nothing
end


@recipe function f(stats::MCMCBasicStats, parsel::NTuple{2,Integer})
    pi_x, pi_y = parsel

    Σ_all = stats.param_stats.cov
    Σ = [Σ_all[pi_x, pi_x] Σ_all[pi_x, pi_y]; Σ_all[pi_y, pi_x] Σ_all[pi_y, pi_y]]

    μ = stats.param_stats.mean[[pi_x, pi_y]]
    mode_xy = stats.mode[[pi_x, pi_y]]

    conf = standard_confidence_vals

    linecolor --> :darkviolet
    linewidth --> 2
    linealpha --> 0.68
    
    for i in eachindex(conf)
        xy = err_ellipsis_path(μ, Σ, conf[i])
        @series begin
            seriestype := :path
            label --> "$(100 *conf[i])%"
            (xy[:,1], xy[:,2])
        end
    end

    markercolor --> :darkviolet
    markersize --> 7
    markeralpha --> 0
    markerstrokewidth --> 2
    markerstrokecolor --> :black
    markerstrokealpha --> 1

    @series begin
        seriestype := :scatter
        label := "mean"
        markershape := :circle
        color --> :black
        ([μ[1]], [μ[2]])
    end

    @series begin
        seriestype := :scatter
        label := "mode"
        markershape := :rect
        color --> :black
        ([mode_xy[1]], [mode_xy[2]])
    end

    vlo = stats.param_stats.minimum[[pi_x, pi_y]]
    vhi = stats.param_stats.maximum[[pi_x, pi_y]]
    rect_xy = rectangle_path(vlo, vhi)
    # bext = 0.1 * (vhi - vlo)
    # xlims = (vlo[1] - bext[1], vhi[1] + bext[1])
    # ylims = (vlo[2] - bext[2], vhi[2] + bext[2])

    @series begin
        seriestype := :path
        label --> "bbox"
        linecolor --> :green
        linewidth --> 2
        linealpha --> 0.5
        # xlims --> xlims
        # ylims --> ylims
        (rect_xy[:,1], rect_xy[:,2])
    end

    nothing
end
