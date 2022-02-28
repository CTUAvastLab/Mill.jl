"""
    SegmentedMax{V <: AbstractVector{<:Number}} <: AbstractAggregation

[`AbstractAggregation`](@ref) implementing segmented max aggregation:

``
f(\\{x_1, \\ldots, x_k\\}) = \\max_{i = 1, \\ldots, k} x_i
``

Stores a vector of parameters `ψ` that are filled into the resulting matrix in case an empty bag is encountered.

See also: [`AbstractAggregation`](@ref), [`AggregationStack`](@ref),
    [`SegmentedMean`](@ref), [`SegmentedSum`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
struct SegmentedMax{V <: AbstractVector{<:Number}} <: AbstractAggregation
    ψ::V
end

Flux.@functor SegmentedMax

SegmentedMax(T::Type, d::Int) = SegmentedMax(zeros(T, d))
SegmentedMax(d::Int) = SegmentedMax(Float32, d)

Flux.@forward SegmentedMax.ψ Base.getindex, Base.length, Base.size, Base.firstindex, Base.lastindex,
        Base.first, Base.last, Base.iterate, Base.eltype

Base.vcat(as::SegmentedMax...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:SegmentedMax})
    SegmentedMax(reduce(vcat, [a.ψ for a in as]))
end

function (a::SegmentedMax)(x::Maybe{AbstractMatrix{T}}, bags::AbstractBags,
                              w::Optional{AbstractVecOrMat{T}}=nothing) where T
    _check_agg(a, x)
    segmented_max_forw(x, a.ψ, bags)
end

segmented_max_forw(::Missing, ψ::AbstractVector, bags::AbstractBags) = repeat(ψ, 1, length(bags))
function segmented_max_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags) 
    T = promote_type(eltype(x), eltype(ψ))
    y = Matrix{T}(fill(_typemin(T), size(x, 1), length(bags)))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                for i in 1:size(x, 1)
                    y[i, bi] = max(y[i, bi], x[i, j])
                end
            end
        end
    end
    y
end

function segmented_max_back(Δ, y, x, ψ, bags) 
    dx = zero(x)
    dψ = zero(ψ)
    v = similar(x, size(x, 1))
    idxs = zeros(Int, size(x, 1))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            fi = first(b)
            v .= x[:,fi]
            idxs .= fi
            for j in b
                for i in 1:size(x,1)
                    if v[i] < x[i, j]
                        idxs[i] = j
                        v[i] = x[i, j]
                    end
                end
            end
            for i in 1:size(x, 1)
                dx[i, idxs[i]] += Δ[i, bi]
            end
        end
    end
    dx, dψ, NoTangent()
end

function segmented_max_back(Δ, y, x::Missing, ψ, bags) 
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    ZeroTangent(), dψ, NoTangent()
end

function ChainRulesCore.rrule(::typeof(segmented_max_forw), args...)
    y = segmented_max_forw(args...)
    grad = Δ -> (NoTangent(), segmented_max_back(Δ, y, args...)...)
    y, grad
end
