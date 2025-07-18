# We document types/constructors/functors in one docstring until
# https://github.com/JuliaDocs/Documenter.jl/issues/558 is resolved
"""
    AbstractAggregation

Supertype for any aggregation operator.

See also: [`AggregationStack`](@ref), [`SegmentedSum`](@ref), [`SegmentedMax`](@ref),
    [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
abstract type AbstractAggregation end

@inline _bagnorm(w::Nothing, b) = length(b)
@inline _bagnorm(w::AbstractVector, b) = @views sum(w[b])
@inline _bagnorm(w::AbstractMatrix, b) = @views vec(sum(w[:, b], dims=2))

@inline _weight(w::Nothing, _, _, ::Type{T}) where T = one(T)
@inline _weight(w::AbstractVector, _, j, _) = w[j]
@inline _weight(w::AbstractMatrix, i, j, _) = w[i, j]

@inline _weightsum(ws::Real, _) = ws
@inline _weightsum(ws::AbstractVector, i) = ws[i]

# more stable definitions for r_map and p_map
ChainRulesCore.rrule(::typeof(softplus), x) = softplus.(x), Δ -> (NoTangent(), unthunk(Δ) .* σ.(x))

# our definition of type min for Maybe{...} types
_typemin(t::Type) = typemin(t)
_typemin(::Type{Missing}) = missing
_typemin(::Type{Maybe{T}}) where T = typemin(T)

function _check_agg(::AbstractAggregation, ::Missing) end
function _check_agg(a::AbstractAggregation, X::AbstractMatrix)
    if size(X, 1) ≠ length(a.ψ)
        DimensionMismatch(
            "Different number of rows in input ($(size(X, 2))) and aggregation ($(length(a.ψ)))"
        ) |> throw
    end
end

include("segmented_sum.jl")
include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")

include("aggregation_stack.jl")

Base.vcat(as::AbstractAggregation...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:AbstractAggregation})
    AggregationStack(tuple(as...))
end

include("bagcount.jl")

# definitions for mixed aggregations
const names = ["Sum", "Mean", "Max", "PNorm", "LSE"]
for p in filter(p -> length(p) > 1, collect(powerset(collect(1:length(names)))))
    s = Symbol("Segmented", names[p]...)
    @eval begin
        """
            $($(s))([t::Type, ]d::Integer)

        Construct [`AggregationStack`](@ref) consisting of $($(
            join("[`Segmented" .* names[p] .* "`](@ref)", ", ", " and ")
       )) operator$($(length(p) > 1 ? "s" : "")).

        $($(
            all(in(["Sum", "Mean", "Max"]), names[p]) ? """
            # Examples
            ```jldoctest
            julia> $(s)(4)
            AggregationStack:
            $(join(" Segmented" .* names[p] .* "(ψ = Float32[0.0, 0.0, 0.0, 0.0])", "\n"))

            julia> $(s)(Float64, 2)
            AggregationStack:
            $(join(" Segmented" .* names[p] .* "(ψ = [0.0, 0.0])", "\n"))
            ```
            """ : ""
        ))

        See also: [`AbstractAggregation`](@ref), [`AggregationStack`](@ref), [`SegmentedSum`](@ref),
            [`SegmentedMax`](@ref), [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
        """
        function $s(d::Integer)
            AggregationStack($((Expr(:call, Symbol("Segmented", n), :d) for n in names[p])...))
        end
    end
    @eval function $s(::Type{T}, d::Integer) where T
        AggregationStack($((Expr(:call, Symbol("Segmented", n), :T, :d) for n in names[p])...))
    end
    @eval export $s
end
