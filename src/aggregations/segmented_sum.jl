"""
    SegmentedSum{V <: AbstractVector{<:Number}} <: AbstractAggregation

[`AbstractAggregation`](@ref) implementing segmented sum aggregation:

``
f(\\{x_1, \\ldots, x_k\\}) = \\sum_{i = 1}^{k} x_i
``

Stores a vector of parameters `ψ` that are filled into the resulting matrix in case an empty bag is encountered.

See also: [`AbstractAggregation`](@ref), [`AggregationStack`](@ref),
    [`SegmentedMax`](@ref), [`SegmentedMean`](@ref), [`SegmentedPNorm`](@ref), [`SegmentedLSE`](@ref).
"""
struct SegmentedSum{V <: AbstractVector{<:Number}} <: AbstractAggregation
    ψ::V
end

Flux.@layer :ignore SegmentedSum

SegmentedSum(T::Type, d::Integer) = SegmentedSum(zeros(T, d))
SegmentedSum(d::Integer) = SegmentedSum(Float32, d)

Flux.@forward SegmentedSum.ψ Base.getindex, Base.length, Base.size, Base.firstindex, Base.lastindex,
        Base.first, Base.last, Base.iterate, Base.eltype

Base.vcat(as::SegmentedSum...) = reduce(vcat, as |> collect)
function Base.reduce(::typeof(vcat), as::Vector{<:SegmentedSum})
    SegmentedSum(reduce(vcat, [a.ψ for a in as]))
end

function (a::SegmentedSum)(x::Maybe{AbstractMatrix{T}}, bags::AbstractBags,
                              w::Optional{AbstractVecOrMat{T}}=nothing) where T
    _check_agg(a, x)
    segmented_sum_forw(x, a.ψ, bags, w)
end

segmented_sum_forw(::Missing, ψ::AbstractVector, bags::AbstractBags, w) = repeat(ψ, 1, length(bags))
function segmented_sum_forw(x::AbstractMatrix, ψ::AbstractVector, bags::AbstractBags, w::Optional{AbstractVecOrMat}) 
    t = promote_type(eltype(x), eltype(ψ))
    y = zeros(t, size(x, 1), length(bags))
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                y[i, bi] = ψ[i]
            end
        else
            for j in b
                for i in axes(x, 1)
                    y[i, bi] += _weight(w, i, j, t) * x[i, j]
                end
            end
        end
    end
    y
end

function segmented_sum_back(Δ, y, x, ψ, bags, w) 
    dx = zero(x)
    dψ = zero(ψ)
    dw = isnothing(w) ? ZeroTangent() : zero(w)
    @inbounds for (bi, b) in enumerate(bags)
        if isempty(b)
            for i in eachindex(ψ)
                dψ[i] += Δ[i, bi]
            end
        else
            for j in b
                for i in axes(x, 1)
                    dx[i, j] += _weight(w, i, j, eltype(x)) * Δ[i, bi]
                    ∇dw_segmented_sum!(dw, Δ, x, y, w, i, j, bi)
                end
            end
        end
    end
    dx, dψ, NoTangent(), dw
end

function segmented_sum_back(Δ, y, x::Missing, ψ, bags, w::Nothing) 
    dψ = zero(ψ)
    @inbounds for (bi, b) in enumerate(bags)
        for i in eachindex(ψ)
            dψ[i] += Δ[i, bi]
        end
    end
    ZeroTangent(), dψ, NoTangent(), ZeroTangent()
end

∇dw_segmented_sum!(dw::ZeroTangent, Δ, x, y, w::Nothing, i, j, bi) = nothing
function ∇dw_segmented_sum!(dw::AbstractVector, Δ, x, y, w::AbstractVector, i, j, bi) 
    dw[j] += Δ[i, bi] * x[i, j]
end
function ∇dw_segmented_sum!(dw::AbstractMatrix, Δ, x, y, w::AbstractMatrix, i, j, bi)
    dw[i, j] += Δ[i, bi] * x[i, j]
end

function ChainRulesCore.rrule(::typeof(segmented_sum_forw), args...)
    y = segmented_sum_forw(args...)
    grad = Δ -> (NoTangent(), segmented_sum_back(unthunk(Δ), y, args...)...)
    y, grad
end
