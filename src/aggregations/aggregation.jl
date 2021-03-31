# We document types/constructors/functors in one docstring until
# https://github.com/JuliaDocs/Documenter.jl/issues/558 is resolved
"""
    AggregationOperator{T <: Number}

Supertype for any aggregation operator. `T` is the type of parameters of the operator.

See also: [`Aggregation`](@ref), [`SegmentedSum`](@ref), [`SegmentedMax`](@ref),
    [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
abstract type AggregationOperator{T <: Number} end

"""
    Aggregation{T, U <: Tuple{Vararg{AggregationOperator{T}}}}

A container that implements a concatenation of one or more `AggregationOperator`s.

Construct with e.g. `mean_aggregation([t::Type, ]d)`, `max_aggregation([t::Type, ]d)` for single
operators and with e.g. `pnormlse_aggregation([t::Type, ]d)` for concatenations. With these calls
all parameters inside operators are initialized randomly as `Float32` arrays, unless type `t` is
further specified. It is also possible to call the constructor directly, see Examples.

Intended to be used as a functor:

    (a::Aggregation)(x, bags[, w])

where `x` is either `Missing`, `AbstractMatrix` or [`ArrayNode`](@ref),
`bags` is [`AbstractBags`](@ref) structure and optionally `w` is an `AbstractVector` of weights.

If [`Mill.bagcount`](@ref) is on, one more row is added to the result containing bag size
after ``x ↦ \\log(x + 1)`` transformation.

# Examples
```jldoctest
julia> a = mean_aggregation(5)
Aggregation{Float32}:
 SegmentedMean(ψ = Float32[0.0, 0.0, 0.0, 0.0, 0.0])

julia> a = meanmax_aggregation(Int64, 4)
Aggregation{Int64}:
 SegmentedMean(ψ = [0, 0, 0, 0])
 SegmentedMax(ψ = [0, 0, 0, 0])

julia> Aggregation(mean_aggregation(4), max_aggregation(4))
Aggregation{Float32}:
 SegmentedMean(ψ = Float32[0.0, 0.0, 0.0, 0.0])
 SegmentedMax(ψ = Float32[0.0, 0.0, 0.0, 0.0])

julia> mean_aggregation(5)(Float32[0 1 2; 3 4 5], bags([1:1, 2:3]))
3×2 Matrix{Float32}:
 0.0       1.5
 3.0       4.5
 0.693147  1.09861
```

See also: [`AggregationOperator`](@ref), [`SegmentedSum`](@ref), [`SegmentedMax`](@ref),
    [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
struct Aggregation{T, U <: Tuple{Vararg{AggregationOperator{T}}}}
    fs::U
    Aggregation(fs::Union{Aggregation, AggregationOperator}...) = Aggregation(fs)
    function Aggregation(fs::Tuple{Vararg{Union{Aggregation{T}, AggregationOperator{T}}}}) where {T}
        ffs = _flatten_agg(fs)
        new{T, typeof(ffs)}(ffs)
    end
end

_flatten_agg(t) = tuple(vcat(map(_flatten_agg, t)...)...)
_flatten_agg(a::Aggregation) = vcat(map(_flatten_agg, a.fs)...)
_flatten_agg(a::AggregationOperator) = [a]

Flux.@functor Aggregation

function (a::Aggregation{T})(x::Union{AbstractArray, Missing}, bags::AbstractBags, args...) where T
    o = reduce(vcat, (f(x, bags, args...) for f in a.fs))
    bagcount() ? vcat(o, Zygote.@ignore permutedims(log.(one(T) .+ length.(bags)))) : o
end
(a::Union{AggregationOperator, Aggregation})(x::ArrayNode, args...) = ArrayNode(a(x.data, args...))
Flux.@forward Aggregation.fs Base.getindex, Base.firstindex, Base.lastindex, Base.first, Base.last, Base.iterate, Base.eltype

Base.length(a::Aggregation) = sum(length.(a.fs))
Base.size(a::Aggregation) = tuple(sum(only, size.(a.fs)))
Base.vcat(as::Aggregation...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:Aggregation})
    Aggregation(tuple(vcat((collect(a.fs) for a in as)...)...))
end

function Base.show(io::IO, a::Aggregation)
    if get(io, :compact, false)
        print(io, "Aggregation(", join(length.(a.fs), ", "), ")")
    else
        print(io, "⟨" * join(repr(f; context=:compact => true) for f in a.fs ", ") * "⟩")
    end
end

function Base.show(io::IO, ::MIME"text/plain", a::Aggregation{T}) where T
    print(io, "Aggregation{$T}:\n")
    Base.print_array(io, a.fs |> collect)
end

function Base.show(io::IO, a::T) where T <: AggregationOperator
    if get(io, :compact, false)
        print(io, nameof(T), "(", length(a), ")")
    else
        _show(io, a)
    end
end

@inline _bagnorm(w::Nothing, b) = length(b)
@inline _bagnorm(w::AbstractVector, b) = @views sum(w[b])
@inline _bagnorm(w::AbstractMatrix, b) = @views vec(sum(w[:, b], dims=2))

@inline _weight(w::Nothing, _, _, ::Type{T}) where T = one(T)
@inline _weight(w::AbstractVector, _, j, _) = w[j]
@inline _weight(w::AbstractMatrix, i, j, _) = w[i, j]

@inline _weightsum(ws::Real, _) = ws
@inline _weightsum(ws::AbstractVector, i) = ws[i]

# more stable definitions for r_map and p_map
ChainRulesCore.rrule(::typeof(softplus), x) = softplus.(x), Δ -> (NO_FIELDS, Δ .* σ.(x))

# our definition of type min for Maybe{...} types
_typemin(t::Type) = typemin(t)
_typemin(::Type{Missing}) = missing
_typemin(::Type{Maybe{T}}) where T = typemin(T)

include("segmented_sum.jl")
include("segmented_mean.jl")
include("segmented_max.jl")
include("segmented_pnorm.jl")
include("segmented_lse.jl")
# include("transformer.jl")

const names = ["Sum", "Mean", "Max", "PNorm", "LSE"]
for p in filter(!isempty, collect(powerset(collect(1:length(names)))))
    s = Symbol(lowercase.(names[p])..., "_aggregation")
    @eval begin
        """
            $($(s))([t::Type, ]d::Int)

        Construct [`Aggregation`](@ref) consisting of $($(
            join("[`Segmented" .* names[p] .* "`](@ref)", ", ", " and ")
       )) operator$($(length(p) > 1 ? "s" : "")).

        $($(
            all(in(["Sum", "Mean", "Max"]), names[p]) ? """
            # Examples
            ```jldoctest
            julia> $(s)(4)
            Aggregation{Float32}:
            $(join(" Segmented" .* names[p] .* "(ψ = Float32[0.0, 0.0, 0.0, 0.0])", "\n"))

            julia> $(s)(Int64, 2)
            Aggregation{Int64}:
            $(join(" Segmented" .* names[p] .* "(ψ = [0, 0])", "\n"))
            ```
            """ : ""
        ))

        See also: [`Aggregation`](@ref), [`AggregationOperator`](@ref), [`SegmentedSum`](@ref),
            [`SegmentedMax`](@ref), [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
        """
        function $s(d::Int)
            Aggregation($((Expr(:call, Symbol("Segmented", n), :d) for n in names[p])...))
        end
    end
    @eval function $s(::Type{T}, d::Int) where T
        Aggregation($((Expr(:call, Expr(:curly, Symbol("Segmented", n), :T), :d) for n in names[p])...))
    end
    # if length(p) > 1
    #     @eval function $s(D::Vararg{Int, $(length(p))})
    #         Aggregation($((Expr(:call, Symbol("_Segmented", n), :(D[$i]))
    #                        for (i,n) in enumerate(names[p]))...))
    #     end
    # end
    @eval export $s
end
