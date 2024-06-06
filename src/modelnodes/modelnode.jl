"""
    AbstractMillModel

Supertype for any model defined in [`Mill.jl`](https://github.com/CTUAvastLab/Mill.jl).
"""
abstract type AbstractMillModel end

(m::AbstractMillModel)(x::AbstractVector{<:AbstractMillNode}) = ChainRulesCore.ignore_derivatives() do
    reduce(catobs, x)
end |> m

include("arraymodel.jl")
include("bagmodel.jl")
include("productmodel.jl")
include("lazymodel.jl")
include("reflectinmodel.jl")
