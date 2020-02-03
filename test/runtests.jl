using Test, Mill, Flux
using Random

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

function mngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in eachindex(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return grads
end

function mgradcheck(f, xs...)
    correct = true
    for (ng, ag) in zip(mngradient(f, xs...), gradient(f, xs...))
        if isnothing(ag)
            if any(ng .!= 0)
                correct = false
                @show ng
            end
        elseif !all(isapprox.(ng, ag, rtol = 1e-5, atol = 1e-5))
            correct = false
            grad_dif = [abs.(x) for x in (ng .- ag)]
            @show ng
            @show ag
            @show grad_dif
        end
    end
    correct
end

mgradtest(f, xs::AbstractArray...) = mgradcheck((xs...) -> sum(sin.(f(xs...))), xs...)

Random.seed!(42)

include("datanode.jl")
include("modelnode.jl")
include("missing.jl")
include("aggregation.jl")
include("gradtests.jl")
include("conv.jl")
include("bags.jl")
include("ngrams.jl")
include("activations.jl")
